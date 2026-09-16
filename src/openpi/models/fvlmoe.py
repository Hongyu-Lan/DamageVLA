"""FVLMoE: Force-aware Vision-Language Mixture-of-Experts fusion module.

Implements the ForceVLA "FVLMoE" late-fusion block (ForceVLA paper, Sec. 4.2 / Table 4) for the
JAX/Flax NNX pi0 model. The module fuses the *frozen* VLM's prefix output (the image + language
tokens, ``prefix_out``) with a projected force token and produces a guidance sequence that is added
into the flow-matching action head. The caller takes the trailing ``action_horizon`` fused tokens
as the guidance (ForceVLA's "final H_action tokens from E_FVLMoE"); the appended force token
attends to every prefix position, so those guidance tokens are force-dependent.

Pipeline (run in float32 for numerically stable attention / softmax / routing):

    E_in    = concat([vl_tokens, force_token], axis=1)              # (b, N+1, d_model)
    (1) pre-norm multi-head self-attention + residual               # force attends to V-L context
    (2) pre-norm FFN + residual
    (3) pre-norm sparse MoE (E experts, top-1 routing) + residual    # the "FVLMoE" routing
    E_fused = out_proj(x)                                           # (b, N+1, d_out)

`d_out` is the action-expert width, so the trailing `action_horizon` fused tokens can be added
element-wise to the action hidden states inside the denoising loop (ForceVLA's additive guidance
injection). Late fusion (after the VLM) is essential: ForceVLA's ablation shows fusing force
before the VLM collapses to 0% success, while this late MoE fusion is their best configuration.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class FVLMoE(nnx.Module):
    """Force-aware Vision-Language Mixture-of-Experts fusion (ForceVLA Sec. 4.2)."""

    def __init__(
        self,
        d_model: int,
        d_out: int,
        *,
        num_experts: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 1.0,
        rngs: nnx.Rngs,
    ):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_experts = num_experts
        hidden = int(mlp_ratio * d_model)

        # (1) Transformer encoder block: pre-norm multi-head self-attention.
        self.norm_attn = nnx.LayerNorm(d_model, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.o_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        # (1b) Pre-norm feed-forward network.
        self.norm_ffn = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn_in = nnx.Linear(d_model, hidden, rngs=rngs)
        self.ffn_out = nnx.Linear(hidden, d_model, rngs=rngs)
        # (2) Sparse Mixture-of-Experts: router + stacked expert MLPs (top-1).
        self.norm_moe = nnx.LayerNorm(d_model, rngs=rngs)
        self.router = nnx.Linear(d_model, num_experts, rngs=rngs)
        key1, key2 = jax.random.split(rngs.params())
        self.exp_w1 = nnx.Param(jax.random.normal(key1, (num_experts, d_model, hidden)) * (d_model**-0.5))
        self.exp_w2 = nnx.Param(jax.random.normal(key2, (num_experts, hidden, d_model)) * (hidden**-0.5))
        # (3) Output projection to the action-expert width.
        self.out_proj = nnx.Linear(d_model, d_out, rngs=rngs)

    def _self_attention(self, x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n d"]:
        b, n, _ = x.shape
        h = self.norm_attn(x)
        q = self.q_proj(h).reshape(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(b, n, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(b, n, self.num_heads, self.head_dim)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (self.head_dim**-0.5)
        probs = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(b, n, self.d_model)
        return x + self.o_proj(ctx)  # full (non-causal) self-attention with residual

    def _ffn(self, x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n d"]:
        h = self.norm_ffn(x)
        h = jax.nn.gelu(self.ffn_in(h))
        return x + self.ffn_out(h)

    def _moe(self, x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n d"]:
        b, n, d = x.shape
        h = self.norm_moe(x)
        flat = h.reshape(b * n, d)
        gate = jax.nn.softmax(self.router(flat), axis=-1)  # (b*n, E)
        top1 = jnp.argmax(gate, axis=-1)  # (b*n,)
        top1_weight = jnp.max(gate, axis=-1, keepdims=True)  # (b*n, 1)
        # Dense-compute every expert, then gather the top-1 expert per token. This is
        # mathematically identical to sparse top-1 routing in the forward pass, and is simple and
        # jit/scan-safe (no ragged dispatch). Cost is ~num_experts x expert FLOPs, negligible here.
        expert_hidden = jax.nn.gelu(jnp.einsum("nd,edh->enh", flat, self.exp_w1.value))  # (E, b*n, hidden)
        expert_out = jnp.einsum("enh,ehd->end", expert_hidden, self.exp_w2.value)  # (E, b*n, d)
        selected = jnp.take_along_axis(expert_out, top1[None, :, None], axis=0)[0]  # (b*n, d)
        routed = selected * top1_weight  # weight the selected expert by its gate score (paper's g_i(x))
        return x + routed.reshape(b, n, d)  # residual around the MoE layer

    @at.typecheck
    def __call__(
        self,
        vl_tokens: at.Float[at.Array, "b n d"],
        force_token: at.Float[at.Array, "b 1 d"],
        *,
        return_hidden: bool = False,
    ) -> at.Float[at.Array, "b n1 dout"] | tuple[at.Float[at.Array, "b n1 dout"], at.Float[at.Array, "b n1 d"]]:
        """Fuse the VLM prefix with the force token.

        Args:
            return_hidden: also return `h`, the fused hidden BEFORE `out_proj`, at `d_model` (the VLM
                width). DraftVLA takes `h[:, -1, :]` -- the last *token*, i.e. the appended force
                token, which has attended to every prefix position -- as the contextualized force
                token feeding `z_phy` (draftvla_plan.md v2.1 §6.1, rule 1). Note the returned value
                is the pre-projection hidden: `out_proj` maps to the action-expert width, which is a
                different space.
        """
        e_in = jnp.concatenate([vl_tokens.astype(jnp.float32), force_token.astype(jnp.float32)], axis=1)
        x = self._self_attention(e_in)
        x = self._ffn(x)
        h = self._moe(x)
        out = self.out_proj(h)
        if return_hidden:
            return out, h
        return out
