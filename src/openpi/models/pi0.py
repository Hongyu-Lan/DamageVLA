import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import fvlmoe as _fvlmoe
from openpi.models import model as _model
from openpi.models import physical as _physical
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # === Force-awareness (ForceVLA) ===
        self.force_aware = config.force_aware
        self.force_fusion = config.force_fusion
        self.force_dim = config.force_dim
        if config.force_aware:
            if config.force_fusion == "token":
                # M1: project raw force into a single action-expert-width conditioning token.
                self.force_proj = nnx.Linear(config.force_dim, action_expert_config.width, rngs=rngs)
            else:
                # M2 (FVLMoE): project force to the VLM width, then fuse it after the frozen VLM.
                self.force_proj = nnx.Linear(config.force_dim, paligemma_config.width, rngs=rngs)
                self.fvlmoe = _fvlmoe.FVLMoE(
                    d_model=paligemma_config.width,
                    d_out=action_expert_config.width,
                    num_experts=config.fvlmoe_num_experts,
                    num_heads=config.fvlmoe_num_heads,
                    mlp_ratio=config.fvlmoe_mlp_ratio,
                    rngs=rngs,
                )

        # === Physical Interaction Token (DraftVLA) ===
        # z_phy is derived from the FVLMoE hidden, so this requires the M2 (fvlmoe) path; the config
        # enforces that. See outlines/draftvla_plan.md v2.1 §6.
        self.phy_enabled = config.phy_enabled
        if config.phy_enabled:
            self.safe_force_dim = config.safe_force_dim
            self.lambda_dist = config.lambda_dist
            self.lambda_proto = config.lambda_proto
            self.phy_proto_ramp_start = config.phy_proto_ramp_start
            self.phy_proto_ramp_steps = config.phy_proto_ramp_steps
            # Kept as plain tuples, converted to arrays at the use site. A bare jnp array assigned to
            # an nnx.Module attribute is not a valid graph node ("Arrays leaves are not supported"),
            # which breaks nnx.grad / nnx.split. These are fixed metadata, not parameters -- they must
            # never be trainable, so a tuple (static) is also the semantically correct choice.
            self.phy_label_mean = tuple(config.phy_label_mean)
            self.phy_label_scale = tuple(config.phy_label_scale)
            self.phy_proj = _physical.PhysicalProjector(paligemma_config.width, config.phy_dim, rngs=rngs)
            self.phy_action_proj = _physical.PhysicalActionProjector(
                config.phy_dim,
                action_expert_config.width,
                config.action_horizon,
                gain_init=config.phy_action_gain_init,
                rngs=rngs,
            )
            self.phy_dist_head = _physical.SafeForceDistributionHead(config.phy_dim, config.safe_force_dim, rngs=rngs)
            self.phy_proto_cls = _physical.PrototypeClassifier(config.phy_dim, config.phy_num_prototypes, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _force_guidance(
        self, prefix_out: at.Float[at.Array, "b n d"], force: at.Float[at.Array, "b f"]
    ) -> tuple[at.Float[at.Array, "b t da"], dict[str, at.Array] | None]:
        """M2 (FVLMoE) additive guidance, plus the physical-branch outputs when enabled.

        Shared by `compute_loss` and `sample_actions` so the two injection sites cannot drift apart:
        adding G_phy at only one of them would leave the physical branch untrained (plan rule 3).
        The guidance depends only on the prefix and the force reading -- not on the denoising step --
        so `sample_actions` calls this once outside its loop.
        """
        force_token = self.force_proj(force)[:, None, :]
        if not self.phy_enabled:
            fused = self.fvlmoe(prefix_out, force_token)
            return fused[:, -self.action_horizon :, :], None

        fused, hidden = self.fvlmoe(prefix_out, force_token, return_hidden=True)
        g_fvl = fused[:, -self.action_horizon :, :]
        # The last *token* of the fused hidden is the appended force token, which has attended to
        # every prefix position (plan rule 1). Not the last feature dimension.
        z_phy = self.phy_proj(hidden[:, -1, :])
        g_phy = self.phy_action_proj(z_phy)
        mu_norm, sigma_norm = self.phy_dist_head(z_phy)
        phy = {
            "z_phy": z_phy,
            "mu_norm": mu_norm,
            "sigma_norm": sigma_norm,
            "proto_logits": self.phy_proto_cls(z_phy),
            # De-normalized, for logging / serving only (plan §6.3): the [12] form is assembled at
            # the serving boundary, never carried in the model output.
            "mu_pred": jnp.asarray(self.phy_label_mean) + jnp.asarray(self.phy_label_scale) * mu_norm,
            "sigma_pred": jnp.asarray(self.phy_label_scale) * sigma_norm,
            # Diagnostics: is the physical guidance actually big enough to move the action head, or
            # is it decorative next to ForceVLA's own guidance? Compared as a ratio downstream.
            "g_fvl_rms": jnp.sqrt(jnp.mean(g_fvl.astype(jnp.float32) ** 2)),
            "g_phy_rms": jnp.sqrt(jnp.mean(g_phy.astype(jnp.float32) ** 2)),
        }
        return g_fvl + g_phy.astype(g_fvl.dtype), phy

    def probe_features(self, observation: _model.Observation) -> dict[str, at.Array]:
        """Frozen internal representations for offline probing (no sampling, no gradients).

        Used to ask the same question of every arm -- "how much of the demonstrated safe-interaction
        statistics can be read out of this model's representation?" -- WITHOUT changing the model:
        a small head is fitted on these features afterwards (examples/force/probe_train_eval.py). This
        is how the ForceVLA and no-force baselines, which carry no distribution head, get a held-out
        KL comparable to PiVLA's, and how the paper's Q5 probe ("z_phy recovers the condition where
        the vision-language token does not") is evaluated.

        Keys (all float32, [b, dim]):
          vl_prefix_mean     masked mean of the frozen VLM prefix (image + text tokens)   -- every arm
          vl_prefix_last     last valid prefix token (the final text token)               -- every arm
          force_token_raw    force_proj(contact_input), before fusion                      -- force-aware arms
          fused_force_token  the force token after FVLMoE, i.e. what z_phy is read from   -- FVLMoE arms
          z_phy              the physical token                                            -- physical-branch arm
        The VLM is frozen in all arms, so vl_prefix_* is the same function everywhere: the control.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        mask = prefix_mask[..., None].astype(jnp.float32)
        out32 = prefix_out.astype(jnp.float32)
        feats = {"vl_prefix_mean": (out32 * mask).sum(1) / jnp.maximum(mask.sum(1), 1.0)}
        last = jnp.sum(prefix_mask, axis=1) - 1
        feats["vl_prefix_last"] = jnp.take_along_axis(out32, last[:, None, None], axis=1)[:, 0]
        if self.force_aware and self.force_fusion == "fvlmoe" and observation.force is not None:
            force_token = self.force_proj(observation.force)[:, None, :]
            feats["force_token_raw"] = force_token[:, 0, :].astype(jnp.float32)
            _, hidden = self.fvlmoe(prefix_out, force_token, return_hidden=True)
            fused = hidden[:, -1, :]
            feats["fused_force_token"] = fused.astype(jnp.float32)
            if self.phy_enabled:
                feats["z_phy"] = self.phy_proj(fused).astype(jnp.float32)
        return feats

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        # M1 force fusion: inject force as a single conditioning token, analogous to the state token.
        if self.force_aware and self.force_fusion == "token" and obs.force is not None:
            force_token = self.force_proj(obs.force)[:, None, :]
            tokens.append(force_token)
            input_mask.append(jnp.ones((obs.force.shape[0], 1), dtype=jnp.bool_))
            # the force token starts its own attention block, like the state token
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _flow_loss_and_phy(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array] | None]:
        """The training forward pass: per-token flow loss, plus physical-branch outputs if enabled."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        action_hidden = suffix_out[:, -self.action_horizon :]
        # M2 force fusion (FVLMoE): fuse the frozen VLM output with the force token and add the
        # trailing action_horizon guidance tokens into the action hidden states (ForceVLA Sec. 4.2).
        # This is the TRAINING injection site; `sample_actions` has the inference one. G_phy must be
        # added at both -- adding it only at inference leaves the physical branch untrained.
        phy = None
        if self.force_aware and self.force_fusion == "fvlmoe" and observation.force is not None:
            # The trailing action_horizon fused tokens are ForceVLA's guidance ("final H_action
            # tokens from E_FVLMoE"); the force token attends to all prefix positions, so they are
            # force-dependent. Cast back to the action dtype so M2's action head runs at the same
            # precision as M1/baseline (FVLMoE computes internally in float32 for stable routing).
            guidance, phy = self._force_guidance(prefix_out, observation.force)
            action_hidden = action_hidden + guidance.astype(action_hidden.dtype)
        v_t = self.action_out_proj(action_hidden)

        return jnp.mean(jnp.square(v_t - u_t), axis=-1), phy

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        chunked_loss, _ = self._flow_loss_and_phy(rng, observation, actions, train=train)
        return chunked_loss

    @override
    def compute_train_losses(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        aux: dict[str, at.Array] | None = None,
        step: at.Int[at.Array, ""] | int | None = None,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        """Total scalar loss + per-loss metrics (plan §7.5).

        With the physical branch off this is exactly `mean(compute_loss(...))`, so every existing
        config trains bit-identically.
        """
        chunked_loss, phy = self._flow_loss_and_phy(rng, observation, actions, train=train)
        # Per-frame action-loss weight (v2 datasets, convert_draftvla_data_to_lerobot.action_loss_weight).
        # The teleop pauses during the approach and the whole reset stage are demonstrated as "hold
        # still"; at every hover-like state they outnumber the demonstrated motion, and the policy
        # learned to hover on the robot (2026-09-21). They stay in the batch -- the physical branch below
        # and the normalization statistics still see them -- but contribute their weight (0.05) of
        # gradient to the flow loss. With no weight in `aux`, or all weights 1, this is exactly the
        # old mean(chunked_loss), so every v1 config trains bit-identically.
        weight_metrics: dict[str, at.Array] = {}
        if aux and "action_loss_weight" in aux:
            weight = jnp.asarray(aux["action_loss_weight"], dtype=chunked_loss.dtype).reshape(chunked_loss.shape[0])
            per_sample = jnp.mean(chunked_loss, axis=-1)
            loss_flow = jnp.sum(per_sample * weight) / jnp.maximum(jnp.sum(weight), 1e-6)
            weight_metrics["action_weight_mean"] = jnp.mean(weight)
        else:
            loss_flow = jnp.mean(chunked_loss)
        if phy is None or not aux:
            return loss_flow, {"loss_flow": loss_flow, **weight_metrics}

        valid = aux["supervision_valid"].astype(jnp.bool_)
        num_valid = jnp.sum(valid.astype(jnp.float32))

        # --- L_dist: KL in normalized label space (identical to raw space; better conditioned) ---
        label_mean = jnp.asarray(self.phy_label_mean)
        label_scale = jnp.asarray(self.phy_label_scale)
        mu_gt_n = (aux["gt_safe_distribution"][:, : self.safe_force_dim] - label_mean) / label_scale
        sigma_gt_n = aux["gt_safe_distribution"][:, self.safe_force_dim :] / label_scale
        # Guard BEFORE the KL, not after the reduction: a sigma_gt of 0 reaching the KL makes
        # log(sigma_pred / 0) = +inf, and 0 * inf = NaN, which poisons the whole loss. The `where`
        # makes this correct regardless of what the converter wrote into the masked rows.
        mu_safe = jnp.where(valid[:, None], mu_gt_n, 0.0)
        sigma_safe = jnp.where(valid[:, None], sigma_gt_n, 1.0)
        kl_b = _physical.diagonal_gaussian_kl(mu_safe, sigma_safe, phy["mu_norm"], phy["sigma_norm"])
        loss_dist = _physical.masked_mean(kl_b, valid)

        # --- L_proto: soft-label cross-entropy over the fixed offline prototypes ---
        y_target = jnp.where(valid[:, None], aux["soft_prototype_target"], 1.0 / aux["soft_prototype_target"].shape[-1])
        ce_b = _physical.soft_label_cross_entropy(phy["proto_logits"], y_target)
        loss_proto = _physical.masked_mean(ce_b, valid)
        # The soft CE is lower-bounded by the target's own entropy, which grows with tau_q -- so raw
        # L_proto is not comparable across a tau ablation. Log the excess over that bound too.
        loss_proto_excess = loss_proto - _physical.masked_mean(_physical.target_entropy(y_target), valid)

        # lambda_proto ramp (plan §7.5 phase 2): 0 -> lambda_proto over phy_proto_ramp_steps.
        lambda_proto = jnp.asarray(self.lambda_proto, dtype=jnp.float32)
        if self.phy_proto_ramp_steps > 0:
            progress = (jnp.asarray(step if step is not None else 0, jnp.float32) - self.phy_proto_ramp_start) / float(
                self.phy_proto_ramp_steps
            )
            lambda_proto = lambda_proto * jnp.clip(progress, 0.0, 1.0)

        total = loss_flow + self.lambda_dist * loss_dist + lambda_proto * loss_proto

        proto_correct = jnp.argmax(phy["proto_logits"], axis=-1) == jnp.argmax(y_target, axis=-1)

        # --- Diagnostics (plan §9): enough to tell WHY a physical head is not learning. ---
        # The trivial predictor: in normalized space the labels are centered/scaled, so predicting
        # mu_norm=0, sigma_norm=1 IS the "constant, ignores the input" baseline. If loss_dist sits at
        # this value the head learned nothing; above it, the head is actively worse than a constant.
        kl_baseline = _physical.diagonal_gaussian_kl(
            mu_safe, sigma_safe, jnp.zeros_like(mu_safe), jnp.ones_like(sigma_safe)
        )
        # Per-dim KL: attributes the loss to a physical dim. The 1-D contract supervises the scalar
        # grip (todo_training_contract.md §B); the legacy 6-D wrench keeps its per-axis names.
        kl_dims = _physical.diagonal_gaussian_kl_per_dim(mu_safe, sigma_safe, phy["mu_norm"], phy["sigma_norm"])
        dim_names = ["grip"] if self.safe_force_dim == 1 else ["fx", "fy", "fz", "tx", "ty", "tz"][: self.safe_force_dim]

        metrics = {
            "loss_flow": loss_flow,
            "loss_dist": loss_dist,
            "loss_proto": loss_proto,
            "loss_proto_excess": loss_proto_excess,
            "proto_acc": _physical.masked_mean(proto_correct.astype(jnp.float32), valid),
            "lambda_proto": lambda_proto,
            "frac_valid": num_valid / jnp.maximum(valid.shape[0], 1),
            "num_valid": num_valid,
            "mu_pred_mean": jnp.mean(phy["mu_pred"]),
            "sigma_pred_mean": jnp.mean(phy["sigma_pred"]),
            "z_phy_norm": jnp.mean(jnp.linalg.norm(phy["z_phy"], axis=-1)),
            # Is z_phy collapsed? ~1.0 => every sample maps to one direction => heads cannot learn.
            "z_phy_cos": _physical.mean_pairwise_cosine(phy["z_phy"]),
            # Does the safe-dist head beat a constant predictor? loss_dist / this should go < 1.
            "loss_dist_baseline": _physical.masked_mean(kl_baseline, valid),
            # Is the prototype classifier collapsed to uniform? ~log(K) => yes.
            "proto_entropy": _physical.masked_mean(_physical.categorical_entropy(phy["proto_logits"]), valid),
            # Does the physical branch actually move the action head, vs ForceVLA's own guidance?
            "g_phy_rms": phy["g_phy_rms"],
            "g_fvl_rms": phy["g_fvl_rms"],
            "g_phy_rel": phy["g_phy_rms"] / (phy["g_fvl_rms"] + 1e-9),
            # The learnable gain on G_phy. Diagnostic only: gain and the projector weights multiply,
            # so read g_phy_rel for the actual magnitude; gain alone is not interpretable.
            "phy_alpha": self.phy_action_proj.gain.value,
        }
        metrics |= {f"kl_{name}": _physical.masked_mean(kl_dims[:, i], valid) for i, name in enumerate(dim_names)}
        metrics |= weight_metrics  # action_weight_mean when the v2 per-frame weight is present
        return total, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        actions, _ = self._sample_actions_and_physical(rng, observation, num_steps=num_steps, noise=noise)
        return actions

    def sample_actions_with_physical(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        """`sample_actions` plus the physical-branch readouts of the SAME forward pass.

        Returns `(actions, phy)`. With the physical branch enabled, `phy` holds `mu_pred` /
        `sigma_pred` [b, safe_force_dim] (de-normalized), `proto_probs` [b, K] and `z_phy` [b, phy_dim];
        otherwise it is `{}`. The readouts come from the very `_force_guidance` call whose G_phy shaped
        these actions, so what gets logged is what acted -- no second forward pass, no drift between
        the two. Serving (policy.py) samples through this so evaluation rollouts can record mu_hat,
        sigma_hat, proto_prob and z_phy per frame (notes/eval_logging_spec.md §4). The readouts are
        diagnostics, never control outputs.
        """
        return self._sample_actions_and_physical(rng, observation, num_steps=num_steps, noise=noise)

    def _sample_actions_and_physical(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # M2 force fusion (FVLMoE): the fused guidance depends only on the (denoising-step-invariant)
        # prefix output and force reading, so compute it once here and add it inside `step`. This is
        # the INFERENCE injection site; `compute_loss` has the training one. Both go through
        # `_force_guidance`, so G_phy cannot be added to one and forgotten at the other.
        guidance = None
        physical: dict[str, at.Array] = {}
        if self.force_aware and self.force_fusion == "fvlmoe" and observation.force is not None:
            guidance, phy = self._force_guidance(prefix_out, observation.force)
            if phy is not None:
                # float32 so a bfloat16 serving model still logs full-precision readouts.
                physical = {
                    "mu_pred": phy["mu_pred"].astype(jnp.float32),
                    "sigma_pred": phy["sigma_pred"].astype(jnp.float32),
                    "proto_probs": jax.nn.softmax(phy["proto_logits"].astype(jnp.float32), axis=-1),
                    "z_phy": phy["z_phy"].astype(jnp.float32),
                }

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            action_hidden = suffix_out[:, -self.action_horizon :]
            if guidance is not None:
                action_hidden = action_hidden + guidance.astype(action_hidden.dtype)
            v_t = self.action_out_proj(action_hidden)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, physical
