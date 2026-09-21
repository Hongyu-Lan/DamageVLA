import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


# === Force-awareness (ForceVLA) ===


def _all_param_paths(config: _pi0_config.Pi0Config) -> list[str]:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    state = nnx.state(abstract_model, nnx.Param).flat_state()
    return ["/".join(str(k) for k in path) for path in state]


def test_forcevla_modules_created_when_enabled():
    paths = _all_param_paths(_pi0_config.Pi0Config(force_aware=True, force_fusion="fvlmoe"))
    assert any("fvlmoe" in p for p in paths)
    assert any("force_proj" in p for p in paths)


def test_forcevla_no_modules_when_disabled():
    # Backward compatibility: vanilla pi0 must not carry any force modules.
    paths = _all_param_paths(_pi0_config.Pi0Config())
    assert not any("fvlmoe" in p for p in paths)
    assert not any("force_proj" in p for p in paths)


def test_forcevla_freeze_vlm_filter():
    config = _pi0_config.Pi0Config(force_aware=True, force_fusion="fvlmoe", freeze_vlm=True)
    state = _get_frozen_state(config)
    frozen = ["/".join(str(k) for k in path) for path in state]
    assert len(frozen) > 0
    # Every frozen param belongs to the VLM: the SigLIP image tower or the non-action-expert LLM.
    assert all(("img" in p) or ("llm" in p and "_1" not in p) for p in frozen)
    # The action expert (LLM "_1" params) and the force modules must stay trainable (not frozen).
    # Note: SigLIP image params can contain "_1" in layer names (e.g. "LayerNorm_1"), so we only
    # treat "_1" *within an llm path* as the action expert.
    assert all(not ("llm" in p and "_1" in p) for p in frozen)
    assert all("force_proj" not in p for p in frozen)
    assert all("fvlmoe" not in p for p in frozen)


@pytest.mark.manual
def test_pi0_force_token_forward():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        force_aware=True,
        force_fusion="token",
    )
    model = config.create(jax.random.key(0))
    obs, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    assert obs.force is not None
    assert obs.force.shape == (2, config.force_dim)
    assert model.compute_loss(jax.random.key(1), obs, actions, train=True).shape == (2, config.action_horizon)
    out = model.sample_actions(jax.random.key(2), obs, num_steps=2)
    assert out.shape == (2, config.action_horizon, config.action_dim)


@pytest.mark.manual
def test_pi0_force_fvlmoe_forward():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        force_aware=True,
        force_fusion="fvlmoe",
    )
    model = config.create(jax.random.key(0))
    obs, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    loss = model.compute_loss(jax.random.key(1), obs, actions, train=True)
    assert loss.shape == (2, config.action_horizon)
    out = model.sample_actions(jax.random.key(2), obs, num_steps=2)
    assert out.shape == (2, config.action_horizon, config.action_dim)


# === Physical branch readouts at inference (DraftVLA) ===


def phy_dummy_config(**overrides) -> _pi0_config.Pi0Config:
    """The task12 physical branch on the dummy backbones: 57-D contact input, 1-D grip target, K=6."""
    kwargs = {
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_horizon": 4,
        "force_aware": True,
        "force_fusion": "fvlmoe",
        "force_dim": 57,
        "phy_enabled": True,
        "safe_force_dim": 1,
        "phy_num_prototypes": 6,
        "phy_label_mean": (1.0,),
        "phy_label_scale": (0.4,),
    }
    kwargs.update(overrides)
    return _pi0_config.Pi0Config(**kwargs)


@pytest.mark.manual
def test_pi0_phy_sample_actions_with_physical_returns_readouts_of_the_same_pass():
    config = phy_dummy_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    noise = jax.random.normal(jax.random.key(3), (2, config.action_horizon, config.action_dim))

    actions, phy = model.sample_actions_with_physical(jax.random.key(2), obs, num_steps=2, noise=noise)
    plain = model.sample_actions(jax.random.key(2), obs, num_steps=2, noise=noise)

    # Same pass, same actions: exposing the readouts must not change what is sampled.
    assert actions.shape == (2, config.action_horizon, config.action_dim)
    np.testing.assert_array_equal(np.asarray(actions), np.asarray(plain))

    assert set(phy) == {"mu_pred", "sigma_pred", "proto_probs", "z_phy"}
    assert phy["mu_pred"].shape == (2, 1)
    assert phy["sigma_pred"].shape == (2, 1)
    assert bool(jnp.all(phy["sigma_pred"] > 0)), "softplus sigma must be strictly positive"
    assert phy["proto_probs"].shape == (2, 6)
    np.testing.assert_allclose(np.asarray(phy["proto_probs"]).sum(-1), 1.0, atol=1e-5)
    assert phy["z_phy"].shape == (2, config.phy_dim)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(phy["z_phy"]), axis=-1), 1.0, atol=1e-3)
    assert all(v.dtype == jnp.float32 for v in phy.values())


@pytest.mark.manual
def test_pi0_probe_features_keys_follow_the_arm():
    """Every arm exposes the frozen VL prefix; force-aware arms add the force path; only the
    physical-branch arm adds z_phy. Shapes are [b, dim], float32, and finite."""
    full = phy_dummy_config()
    forcevla = _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4, force_aware=True, force_fusion="fvlmoe", force_dim=57
    )
    noforce = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4)
    expected = {
        "full": {"vl_prefix_mean", "vl_prefix_last", "force_token_raw", "fused_force_token", "z_phy"},
        "forcevla": {"vl_prefix_mean", "vl_prefix_last", "force_token_raw", "fused_force_token"},
        "noforce": {"vl_prefix_mean", "vl_prefix_last"},
    }
    for name, config in (("full", full), ("forcevla", forcevla), ("noforce", noforce)):
        model = config.create(jax.random.key(0))
        feats = model.probe_features(config.fake_obs(batch_size=2))
        assert set(feats) == expected[name], (name, set(feats))
        for k, v in feats.items():
            assert v.shape[0] == 2 and v.ndim == 2, (name, k, v.shape)
            assert v.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(v))), (name, k)
    assert model.probe_features(noforce.fake_obs(batch_size=2))["vl_prefix_mean"].shape[1] > 0


@pytest.mark.manual
def test_pi0_forcevla_sample_actions_with_physical_has_no_readouts():
    """Physical branch OFF (the ForceVLA arm): same API, empty readouts, actions unchanged."""
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        force_aware=True,
        force_fusion="fvlmoe",
        force_dim=57,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    noise = jax.random.normal(jax.random.key(3), (2, config.action_horizon, config.action_dim))
    actions, phy = model.sample_actions_with_physical(jax.random.key(2), obs, num_steps=2, noise=noise)
    assert phy == {}
    plain = model.sample_actions(jax.random.key(2), obs, num_steps=2, noise=noise)
    np.testing.assert_array_equal(np.asarray(actions), np.asarray(plain))
