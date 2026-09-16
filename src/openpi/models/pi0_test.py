import flax.nnx as nnx
import jax
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
