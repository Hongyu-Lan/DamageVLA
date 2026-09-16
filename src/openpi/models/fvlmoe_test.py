import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models.fvlmoe import FVLMoE


def test_fvlmoe_shapes_and_force_influence():
    model = FVLMoE(d_model=16, d_out=8, num_experts=4, num_heads=4, mlp_ratio=1.0, rngs=nnx.Rngs(0))
    vl_tokens = jnp.ones((2, 5, 16))
    force_a = jnp.ones((2, 1, 16))
    force_b = jnp.zeros((2, 1, 16))

    out_a = model(vl_tokens, force_a)
    out_b = model(vl_tokens, force_b)

    # Output is (batch, num_vl_tokens + 1, d_out).
    assert out_a.shape == (2, 6, 8)
    # The force token must change the fused output -- guards against force being silently ignored.
    assert not jnp.allclose(out_a, out_b)
