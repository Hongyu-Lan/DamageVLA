"""Tests for the DraftVLA physical branch (outlines/draftvla_plan.md v2.1 §10).

Each test here is a gate on a specific failure the plan calls out by name; the docstrings say which.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import physical as _physical
from openpi.models import pi0_config


def _config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=4,
        force_aware=True,
        force_fusion="fvlmoe",
        phy_enabled=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        **kwargs,
    )


def _aux(batch_size: int, *, valid: bool, num_prototypes: int = 4) -> dict:
    return {
        "gt_safe_distribution": jnp.concatenate([jnp.zeros((batch_size, 6)), jnp.ones((batch_size, 6))], axis=-1),
        "soft_prototype_target": jnp.full((batch_size, num_prototypes), 1.0 / num_prototypes),
        "supervision_valid": jnp.full((batch_size,), valid, dtype=jnp.bool_),
    }


def test_kl_is_invariant_under_shared_per_dim_scaling():
    """Plan §6.2 / rule 10: the KL must be identical in raw and normalized space.

    This is the property v2's normalizer silently broke by giving mu and sigma different scales.
    """
    rng = np.random.default_rng(0)
    mu_gt, mu_pred = jnp.asarray(rng.normal(size=(3, 6))), jnp.asarray(rng.normal(size=(3, 6)))
    sigma_gt = jnp.asarray(np.abs(rng.normal(size=(3, 6))) + 0.5)
    sigma_pred = jnp.asarray(np.abs(rng.normal(size=(3, 6))) + 0.5)
    # A per-dim affine change of variable on the wrench: x' = (x - m) / s.
    m = jnp.asarray(rng.normal(size=(6,)))
    s = jnp.asarray(np.abs(rng.normal(size=(6,))) + 1.0)

    raw = _physical.diagonal_gaussian_kl(mu_gt, sigma_gt, mu_pred, sigma_pred)
    normalized = _physical.diagonal_gaussian_kl((mu_gt - m) / s, sigma_gt / s, (mu_pred - m) / s, sigma_pred / s)
    np.testing.assert_allclose(raw, normalized, rtol=1e-5)


def test_kl_with_unshared_scaling_changes_the_objective():
    """The negative control for the test above: different mu/sigma scales are NOT equivalent."""
    mu_gt = jnp.zeros((1, 6))
    sigma_gt = jnp.full((1, 6), 2.0)
    mu_pred = jnp.full((1, 6), 0.5)
    sigma_pred = jnp.full((1, 6), 1.0)
    s_mu, s_sigma = 13.4853, 33.0706  # the real gap on the `ty` dim (plan §6.2)

    raw = _physical.diagonal_gaussian_kl(mu_gt, sigma_gt, mu_pred, sigma_pred)
    unshared = _physical.diagonal_gaussian_kl(mu_gt / s_mu, sigma_gt / s_sigma, mu_pred / s_mu, sigma_pred / s_sigma)
    assert not np.allclose(raw, unshared)


def test_masked_mean_is_zero_and_finite_when_nothing_is_valid():
    """Plan §7.2: the effective batch can be empty; the max(.,1) guard must hold."""
    values = jnp.asarray([1.0, 2.0, 3.0])
    valid = jnp.zeros((3,), dtype=jnp.bool_)
    out = _physical.masked_mean(values, valid)
    assert float(out) == 0.0
    assert jnp.isfinite(out)


def test_zero_sigma_gt_would_nan_without_the_guard():
    """Plan §5/§7.2: a sigma_gt of 0 reaching the KL gives log(x/0)=inf, and 0*inf=NaN.

    Documents *why* the `where` guard exists: with the floor removed this is exactly the trap.
    """
    kl = jnp.log(jnp.asarray(1.0) / jnp.asarray(0.0))  # the unguarded log(sigma_pred / sigma_gt)
    assert jnp.isinf(kl)
    assert jnp.isnan(jnp.asarray(0.0) * kl)  # masking AFTER the reduction cannot rescue it
    # With the guard, the same input is finite.
    guarded = _physical.diagonal_gaussian_kl(jnp.zeros((1, 6)), jnp.zeros((1, 6)), jnp.zeros((1, 6)), jnp.ones((1, 6)))
    assert jnp.all(jnp.isfinite(guarded))


def test_all_masked_batch_gives_zero_finite_losses_and_zero_grads():
    """Plan §10.5: assert on the LOSS VALUE, not only on gradients.

    A test that checks only "finite gradients" half-passes while the loss is NaN, because the
    gradient of the NaN path is coincidentally 0.
    """
    config = _config()
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    total, metrics = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux(2, valid=False)
    )
    assert jnp.isfinite(total), "all-masked batch must not produce NaN/inf"
    assert float(metrics["loss_dist"]) == 0.0
    assert float(metrics["loss_proto"]) == 0.0
    assert float(metrics["frac_valid"]) == 0.0
    # The total must equal the flow loss alone: the masked terms contribute exactly nothing.
    np.testing.assert_allclose(float(total), float(metrics["loss_flow"]), rtol=1e-6)


def test_valid_batch_produces_nonzero_physical_losses():
    config = _config()
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    total, metrics = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux(2, valid=True)
    )
    assert jnp.isfinite(total)
    assert float(metrics["loss_dist"]) > 0.0
    assert float(metrics["frac_valid"]) == 1.0
    assert set(metrics) >= {"loss_flow", "loss_dist", "loss_proto", "loss_proto_excess", "proto_acc"}


def test_12d_force_input_keeps_12d_safe_distribution_target():
    """DraftVLA conditions on [mean(6), difference(6)] without expanding the aggregate safe target."""
    config = _config(force_dim=12, safe_force_dim=6)
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    assert observation.force is not None
    assert observation.force.shape == (2, 12)
    assert model.force_proj.in_features == 12
    assert model.phy_dist_head.fc_out.out_features == 12

    total, metrics = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux(2, valid=True)
    )
    assert jnp.isfinite(total)
    assert float(metrics["loss_dist"]) > 0.0


def test_safe_label_normalizer_is_validated_against_safe_output_not_input():
    config = _config(force_dim=12, safe_force_dim=6)
    assert len(config.phy_label_mean) == len(config.phy_label_scale) == config.safe_force_dim

    with pytest.raises(ValueError, match="safe_force_dim=12"):
        _config(force_dim=12, safe_force_dim=12)


def test_g_phy_has_gradient_at_the_training_site():
    """Plan §10.6 / rule 3: G_phy must be added in compute_loss, not only in sample_actions.

    This is the gate on the review's finding: adding guidance only at the inference site leaves the
    physical action branch with no gradient, so it looks alive and never trains.
    """
    config = _config()
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    def loss_fn(m):
        # aux=None => only the flow loss. If G_phy were absent from the training forward pass, the
        # action projector could not influence it and its gradient would be exactly zero.
        return jnp.mean(m.compute_loss(jax.random.key(0), observation, actions, train=True))

    grads = nnx.grad(loss_fn)(model)
    g_phy_grads = jax.tree.leaves(grads["phy_action_proj"])
    assert g_phy_grads, "phy_action_proj has no parameters in the gradient tree"
    assert any(jnp.any(g != 0) for g in g_phy_grads), (
        "phy_action_proj got a zero gradient from compute_loss: G_phy is missing from the TRAINING "
        "injection site (pi0.py compute_loss), so the physical action branch never trains."
    )


def test_phy_disabled_is_bit_identical_to_forcevla():
    """Plan §10.2: with phy_enabled=False the model must be exactly today's pi0_force_fvlmoe."""
    base_kwargs = {
        "action_dim": 32,
        "action_horizon": 4,
        "force_aware": True,
        "force_fusion": "fvlmoe",
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
    }
    off = pi0_config.Pi0Config(**base_kwargs, phy_enabled=False)
    model = off.create(jax.random.key(0))
    observation, actions = off.fake_obs(batch_size=2), off.fake_act(batch_size=2)

    total, metrics = model.compute_train_losses(jax.random.key(0), observation, actions, train=True, aux={})
    chunked = model.compute_loss(jax.random.key(0), observation, actions, train=True)
    np.testing.assert_allclose(float(total), float(jnp.mean(chunked)), rtol=1e-6)
    assert metrics == {"loss_flow": metrics["loss_flow"]}, "physical metrics must be absent when disabled"


def test_phy_requires_fvlmoe_fusion():
    """Plan §6: z_phy is derived from the FVLMoE hidden, so the M2 path is required."""
    with pytest.raises(ValueError, match="phy_enabled requires"):
        pi0_config.Pi0Config(force_aware=True, force_fusion="token", phy_enabled=True)
    with pytest.raises(ValueError, match="phy_enabled requires"):
        pi0_config.Pi0Config(force_aware=False, phy_enabled=True)


def test_fvlmoe_return_hidden_shape_and_last_token_is_the_force_token():
    """Plan rule 1: h[:, -1, :] is the appended force TOKEN, at the VLM width."""
    from openpi.models import fvlmoe as _fvlmoe

    d_model, d_out, n = 16, 8, 5
    module = _fvlmoe.FVLMoE(d_model=d_model, d_out=d_out, num_heads=2, rngs=nnx.Rngs(0))
    vl = jnp.zeros((2, n, d_model))
    force = jnp.ones((2, 1, d_model))

    out, hidden = module(vl, force, return_hidden=True)
    assert out.shape == (2, n + 1, d_out)
    assert hidden.shape == (2, n + 1, d_model), "hidden must be pre-out_proj, at the VLM width"
    # Same call without the flag returns just the projection, unchanged.
    np.testing.assert_allclose(np.asarray(module(vl, force)), np.asarray(out), rtol=1e-6)


def test_group_id_never_reaches_the_model():
    """Plan rule 9 / §10.3: group_id is supervision-only.

    Structural, not a convention: Observation.from_dict ignores unknown keys, so a group_id in the
    batch dict cannot enter the model's input PyTree.
    """
    batch = {
        "image": {"base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((1,), dtype=bool)},
        "state": np.zeros((1, 32), dtype=np.float32),
        "force": np.zeros((1, 12), dtype=np.float32),
        "group_id": np.asarray([7], dtype=np.int32),
        "gt_safe_distribution": np.zeros((1, 12), dtype=np.float32),
    }
    observation = _model.Observation.from_dict(batch)
    leaves = jax.tree.leaves(observation)
    assert not any(np.array_equal(np.asarray(x), np.asarray([7])) for x in leaves), "group_id leaked into Observation"
    assert "group_id" not in observation.to_dict()


def test_aux_keys_agree_between_loader_and_policy():
    """The two AUX_KEYS lists must not drift apart."""
    from openpi.policies import draftvla_policy
    from openpi.training import data_loader

    assert set(data_loader.AUX_KEYS) == set(draftvla_policy.AUX_KEYS)


# === 2026-09-16 contract (Plan B): 57-D contact input, 1-D grip target, learnable gain ============


def _config_task12(**kwargs) -> pi0_config.Pi0Config:
    return _config(
        force_dim=57,
        safe_force_dim=1,
        phy_num_prototypes=6,
        phy_label_mean=(1.030686,),
        phy_label_scale=(0.404850,),
        **kwargs,
    )


def _aux_task12(batch_size: int, *, valid: bool) -> dict:
    return {
        "gt_safe_distribution": jnp.concatenate(
            [jnp.zeros((batch_size, 1)), jnp.ones((batch_size, 1))], axis=-1
        ),
        "soft_prototype_target": jnp.full((batch_size, 6), 1.0 / 6),
        "supervision_valid": jnp.full((batch_size,), valid, dtype=jnp.bool_),
    }


def test_57d_contact_input_keeps_1d_grip_target():
    """Contract §A/§B: force_dim=57 conditions the model; the supervised target is [mu, sigma] of
    the scalar grip -- 2 numbers, so the dist head's output is 2 wide and the KL is 1-D."""
    config = _config_task12()
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    assert observation.force.shape == (2, 57)
    assert model.force_proj.in_features == 57
    assert model.phy_dist_head.fc_out.out_features == 2
    loss, metrics = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux_task12(2, valid=True)
    )
    assert jnp.isfinite(loss)
    assert "kl_grip" in metrics and "kl_fx" not in metrics


def test_learnable_gain_scales_g_phy_and_is_logged():
    """Contract §0c: the gain multiplies G_phy (g_phy_rel scales with it) and is logged as
    phy_alpha. gain=0 must silence the physical guidance entirely."""
    config = _config_task12(phy_action_gain_init=13.0)
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    _, metrics = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux_task12(2, valid=True)
    )
    assert float(metrics["phy_alpha"]) == pytest.approx(13.0)

    model13_rel = float(metrics["g_phy_rel"])
    model.phy_action_proj.gain.value = jnp.asarray(1.0)
    _, metrics1 = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux_task12(2, valid=True)
    )
    assert model13_rel == pytest.approx(13.0 * float(metrics1["g_phy_rel"]), rel=1e-3)

    model.phy_action_proj.gain.value = jnp.asarray(0.0)
    _, metrics0 = model.compute_train_losses(
        jax.random.key(0), observation, actions, train=True, aux=_aux_task12(2, valid=True)
    )
    assert float(metrics0["g_phy_rel"]) == pytest.approx(0.0, abs=1e-9)


def test_gain_default_preserves_legacy_behavior():
    """phy_action_gain_init defaults to 1.0 so every pre-contract config is bit-identical."""
    assert pi0_config.Pi0Config(action_dim=32, action_horizon=4).phy_action_gain_init == 1.0
