import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # === Force-awareness (ForceVLA) ===
    # When False, the model is exactly vanilla pi0 (full backward compatibility).
    force_aware: bool = False
    # Dimensionality of the force-conditioning input. Legacy ForceVLA uses one 6-D flange wrench;
    # DraftVLA uses 12-D [two-finger mean wrench, signed half-difference wrench].
    force_dim: int = 6
    # "token": M1, inject force as a single conditioning token in the action-expert suffix.
    # "fvlmoe": M2, faithful ForceVLA late fusion via the FVLMoE module (after the frozen VLM).
    force_fusion: Literal["token", "fvlmoe"] = "token"
    # Freeze the PaliGemma VLM (SigLIP image tower + Gemma backbone), training only the action
    # expert + force modules. Decoupled from force_aware so a no-force baseline can match the freeze.
    freeze_vlm: bool = False
    # FVLMoE (M2) hyperparameters (ForceVLA Table 4).
    fvlmoe_num_experts: int = 4
    fvlmoe_num_heads: int = 8
    fvlmoe_mlp_ratio: float = 1.0

    # === Physical Interaction Token (DraftVLA, outlines/draftvla_plan.md v2.1) ===
    # When False, the model is exactly today's force-aware pi0 (full backward compatibility).
    # Requires force_aware=True and force_fusion="fvlmoe": z_phy is derived from the FVLMoE hidden.
    phy_enabled: bool = False
    phy_dim: int = 128
    phy_num_prototypes: int = 4
    # Dimensionality of the physical quantity whose safe distribution is supervised. Intentionally
    # independent from force_dim. 2026-09-16 contract: the model conditions on the 57-D contact
    # input while the target is the 1-D scalar grip -> safe_force_dim=1, [mu, sigma] = 2 values.
    # The legacy 6-D wrench target (12 values) remains valid for the retired configs.
    safe_force_dim: int = 6
    # Loss weights (plan §7.5). lambda_proto is ramped 0 -> target over `phy_proto_ramp_steps`.
    lambda_dist: float = 1.0
    lambda_proto: float = 0.05
    lambda_nll: float = 0.0
    use_force_nll: bool = False
    phy_proto_ramp_start: int = 0
    phy_proto_ramp_steps: int = 0
    # Init of the learnable scalar gain on G_phy (todo_training_contract.md §0c, decided 2026-09-16:
    # 13 lifts the initial g_phy_rel from ~0.023 to ~0.3). 1.0 preserves the legacy behavior. If the
    # first ~500 steps show loss_flow clearly worse than the forcevla baseline (~0.028), drop to 5-8.
    phy_action_gain_init: float = 1.0
    # === Ablation switches (2026-09-22). THE DEFAULTS ARE PiVLA. ===
    # Leaving both untouched reproduces the full method bit-for-bit, so the three main arms need no
    # edit; setting a switch back to its default here is the whole "restore PiVLA" operation.
    #
    # B2 "no extra guidance": False drops G_phy from the action path. z_phy, both heads and both
    # auxiliary losses stay on -- the supervision still reaches the action expert through the shared
    # FVLMoE (G_fvl), so this variant subtracts the direct pathway alone, not the supervision.
    phy_guidance: bool = True
    # B3 "vision-language token": True reads z_phy from the last VALID VLM prefix token, taken
    # BEFORE FVLMoE fusion, instead of the fused force token. FVLMoE and G_fvl are untouched, so
    # force still reaches the action expert; only the origin of the representation moves. Any
    # post-fusion token would already have attended to the force token and prove nothing.
    phy_source_vl: bool = False
    # Safe-distribution label normalizer (plan §6.2). ONE scale per physical dim, shared by mu and
    # sigma -- that shared scale is what keeps the KL identical to raw space. `mean` shifts mu only.
    #
    # Defaults: mean of the 12 group mus / median of the 12 group sigmas, over the shipped labels of
    # DamageVLA_training_post_process_20260821 (all 22 successful contributing episodes).
    # NOTE: for the real run these MUST be recomputed from the training split only (plan §4.3.1) --
    # as shipped they are computed over episodes that a val split would hold out.
    phy_label_mean: tuple[float, ...] = (-2.134591, -0.709413, 2.234035, 18.499234, 21.234861, 33.975909)
    phy_label_scale: tuple[float, ...] = (4.310663, 1.925379, 2.313801, 24.235823, 40.294040, 26.070025)

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if (not self.phy_guidance or self.phy_source_vl) and not self.phy_enabled:
            raise ValueError(
                "phy_guidance / phy_source_vl are ablations OF the physical branch and require "
                f"phy_enabled=True (got phy_enabled={self.phy_enabled})."
            )
        if self.phy_enabled and not (self.force_aware and self.force_fusion == "fvlmoe"):
            raise ValueError(
                "phy_enabled requires force_aware=True and force_fusion='fvlmoe': the physical token is "
                f"derived from the FVLMoE hidden (got force_aware={self.force_aware}, "
                f"force_fusion={self.force_fusion!r})."
            )
        if self.phy_enabled and not (len(self.phy_label_mean) == len(self.phy_label_scale) == self.safe_force_dim):
            raise ValueError(
                "phy_label_mean/phy_label_scale must both have "
                f"safe_force_dim={self.safe_force_dim} entries, got "
                f"{len(self.phy_label_mean)} / {len(self.phy_label_scale)}."
            )
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                force=(jax.ShapeDtypeStruct([batch_size, self.force_dim], jnp.float32) if self.force_aware else None),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        # Freeze the PaliGemma VLM (SigLIP image tower + the Gemma backbone expert), leaving the
        # action expert (the "_1" LLM params), force_proj, and fvlmoe trainable. This is the
        # ForceVLA fine-tuning regime: only the action expert + force modules learn.
        if self.freeze_vlm and not has_lora:
            img_params_filter = nnx_utils.PathRegex(".*img.*")
            filters.append(
                nnx.Any(
                    img_params_filter,
                    nnx.All(gemma_params_filter, nnx.Not(action_expert_params_filter)),
                )
            )

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
