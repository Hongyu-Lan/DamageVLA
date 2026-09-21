import numpy as np

from openpi.models import model as _model
from openpi.policies import draftvla_policy
from openpi.policies import force_policy


def test_force_inputs():
    example = force_policy.make_force_example()
    out = force_policy.ForceInputs(model_type=_model.ModelType.PI0)(example)

    assert np.asarray(out["state"]).shape == (7,)
    # The 6-axis force/torque reading is emitted under the top-level "force" key.
    assert np.asarray(out["force"]).shape == (6,)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    # The (absent) right wrist image is padded and masked out for pi0.
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is False
    assert out["prompt"] == "grasp the banana"


def test_force_outputs_slices_to_seven():
    model_actions = np.zeros((8, 32), dtype=np.float32)
    out = force_policy.ForceOutputs()({"actions": model_actions})
    assert out["actions"].shape == (8, 7)


def test_draftvla_inputs_use_explicit_contact_input_key():
    # 2026-09-16 contract: the 57-D contact input arrives under its own key and is adapted to the
    # model's generic "force" tensor at this boundary; the legacy flange key is never expected.
    example = draftvla_policy.make_draftvla_example()
    expected = np.asarray(example["observation/contact_input"])
    out = draftvla_policy.DraftVLAInputs(model_type=_model.ModelType.PI0)(example)

    np.testing.assert_array_equal(out["force"], expected)
    assert np.asarray(out["force"]).shape == (57,)
    assert "observation/force_torque" not in example
    assert "observation/gripper_wrench" not in example
