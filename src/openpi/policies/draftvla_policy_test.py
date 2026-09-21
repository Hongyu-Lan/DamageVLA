"""DraftVLA serving path: the physical readouts must reach the client through Policy + DraftVLAOutputs.

The model-level contract is tested in models/pi0_test.py; this covers the plumbing above it, which
is what a policy server actually runs: Policy.infer -> output transforms -> response dict.
"""

import jax
import numpy as np
import pytest

from openpi.models import pi0_config as _pi0_config
from openpi.models import pi0_test as _pi0_test
from openpi.policies import draftvla_policy
from openpi.policies import policy as _policy


def _model_ready_observation(config: _pi0_config.Pi0Config) -> dict:
    """One unbatched observation in the model's own key layout (no robot transforms needed)."""
    obs = config.fake_obs(batch_size=1)
    return jax.tree.map(lambda x: np.asarray(x)[0], obs.to_dict())


@pytest.mark.manual
def test_policy_infer_returns_physical_readouts_for_draftvla():
    config = _pi0_test.phy_dummy_config()
    model = config.create(jax.random.key(0))
    policy = _policy.Policy(
        model, output_transforms=[draftvla_policy.DraftVLAOutputs()], sample_kwargs={"num_steps": 2}
    )

    out = policy.infer(_model_ready_observation(config))

    assert out["actions"].shape == (config.action_horizon, 7)
    # [mu_hat, sigma_hat] under the 1-D grip contract, K=6 prototype probabilities, the 128-D token.
    assert out["safe_force_distribution"].shape == (2,)
    assert out["safe_force_distribution"][1] > 0
    assert out["prototype_probs"].shape == (6,)
    np.testing.assert_allclose(out["prototype_probs"].sum(), 1.0, atol=1e-5)
    assert out["z_phy"].shape == (config.phy_dim,)
    # Internal keys never leak to the wire.
    assert not {"mu_pred", "sigma_pred", "proto_probs", "force", "state"} & set(out)


@pytest.mark.manual
def test_policy_infer_forcevla_has_no_physical_keys():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        force_aware=True,
        force_fusion="fvlmoe",
        force_dim=57,
    )
    model = config.create(jax.random.key(0))
    policy = _policy.Policy(
        model, output_transforms=[draftvla_policy.DraftVLAOutputs()], sample_kwargs={"num_steps": 2}
    )

    out = policy.infer(_model_ready_observation(config))

    assert out["actions"].shape == (config.action_horizon, 7)
    assert not {"safe_force_distribution", "prototype_probs", "z_phy"} & set(out)
