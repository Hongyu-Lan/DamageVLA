"""v2 (2026-09-22): per-frame `action_loss_weight` on the flow loss (pi0.compute_train_losses).

Contract: with no weight in `aux`, or all weights 1, the flow loss is bit-for-bit the old mean, so every
v1 config trains identically; with weights it is the weight-normalised mean of per-sample flow losses,
for every arm (physical branch on or off).
"""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import physical_test as _pt
from openpi.models import pi0_config


def _per_sample_flow(model, rng, observation, actions):
    return jnp.mean(model.compute_loss(rng, observation, actions, train=True), axis=-1)


def test_all_ones_weight_is_the_plain_mean():
    config = _pt._config()
    model = config.create(jax.random.key(0))
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    aux = {**_pt._aux(2, valid=True), "action_loss_weight": jnp.ones((2, 1))}
    _, metrics = model.compute_train_losses(jax.random.key(3), obs, act, train=True, aux=aux)
    expected = jnp.mean(_per_sample_flow(model, jax.random.key(3), obs, act))
    np.testing.assert_allclose(float(metrics["loss_flow"]), float(expected), rtol=1e-6)
    assert float(metrics["action_weight_mean"]) == 1.0


def test_no_weight_key_keeps_the_v1_loss_and_metrics():
    config = _pt._config()
    model = config.create(jax.random.key(0))
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    _, metrics = model.compute_train_losses(jax.random.key(3), obs, act, train=True, aux=_pt._aux(2, valid=True))
    expected = jnp.mean(_per_sample_flow(model, jax.random.key(3), obs, act))
    np.testing.assert_allclose(float(metrics["loss_flow"]), float(expected), rtol=1e-6)
    assert "action_weight_mean" not in metrics


def test_weights_reweight_per_sample_flow_losses():
    config = _pt._config()
    model = config.create(jax.random.key(0))
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    w = jnp.asarray([[1.0], [0.05]])
    aux = {**_pt._aux(2, valid=True), "action_loss_weight": w}
    _, metrics = model.compute_train_losses(jax.random.key(3), obs, act, train=True, aux=aux)
    per = _per_sample_flow(model, jax.random.key(3), obs, act)
    expected = (per[0] * 1.0 + per[1] * 0.05) / 1.05
    np.testing.assert_allclose(float(metrics["loss_flow"]), float(expected), rtol=1e-6)
    np.testing.assert_allclose(float(metrics["action_weight_mean"]), 0.525, rtol=1e-6)
    # And it is genuinely different from the plain mean unless the two samples happen to tie.
    if abs(float(per[0] - per[1])) > 1e-6:
        assert not np.isclose(float(metrics["loss_flow"]), float(jnp.mean(per)), rtol=1e-6)


def test_weight_applies_with_the_physical_branch_off():
    """ForceVLA-style and no-force arms share the v2 data, so the weight must apply to them too."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=4,
        force_aware=True,
        force_fusion="fvlmoe",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(jax.random.key(0))
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    w = jnp.asarray([[1.0], [0.05]])
    total, metrics = model.compute_train_losses(jax.random.key(3), obs, act, train=True, aux={"action_loss_weight": w})
    per = _per_sample_flow(model, jax.random.key(3), obs, act)
    expected = (per[0] * 1.0 + per[1] * 0.05) / 1.05
    np.testing.assert_allclose(float(total), float(expected), rtol=1e-6)
    assert set(metrics) == {"loss_flow", "action_weight_mean"}
