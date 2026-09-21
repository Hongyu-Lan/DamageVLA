"""DraftVLA (Damage-Aware ForceVLA) robot transforms.

Extends the force-aware transforms with the physical-branch supervision (2026-09-16 contract,
outlines/todo_training_contract.md). The model inputs are two RGB views, the prompt, state(7), and
the 57-D contact input ``[left_data_zeroed(25), right_data_zeroed(25), gripper_width(1),
force_torque_zeroed(6)]`` (contract §A). The extra keys carried here are *labels*, present during
training only and never during inference:

    gt_safe_distribution   [2]   group-level safe-grip distribution, [mu_grip, sigma_grip]
    soft_prototype_target  [6]   soft prototype target (K=6)
    supervision_valid      ()    bool mask -- prepare/reset frames carry no physical label
    group_id               ()    supervision-only, for logging (plan rule 9)

These ride through the transform pipeline as plain dict keys and are split off into the loader's
`aux` dict (`data_loader.AUX_KEYS`). `Observation.from_dict` ignores unknown keys, so `group_id`
provably never enters the model's input PyTree -- that is the point of routing them this way rather
than adding `Observation` fields.

For DraftVLA, ``observation/contact_input`` must follow examples/force/draftvla_contact.py exactly:
the fingertip `_data` voltages zeroed per episode over the leading prepare window, the MEASURED
gripper width in metres, and the pipeline's `force_torque_zeroed`. Training and inference must use
the same zeroing, ordering, and units. The generic model still names the resulting tensor
``force``. The supervised safe distribution is the scalar-grip ``[mu, sigma]`` target.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import force_policy

# Label keys that are supervision, not model input. Kept in sync with data_loader.AUX_KEYS.
# `action_loss_weight` (v2, 2026-09-22) weights the flow loss per frame; v1 datasets do not carry it.
AUX_KEYS = ("gt_safe_distribution", "soft_prototype_target", "supervision_valid", "group_id", "action_loss_weight")


def make_draftvla_example() -> dict:
    """Creates a random input example for the DraftVLA policy (inference-shaped: no labels)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/contact_input": np.random.rand(57),
        "prompt": "grasp the banana from the table and place it into the box",
    }


@dataclasses.dataclass(frozen=True)
class DraftVLAInputs(transforms.DataTransformFn):
    """`ForceInputs` plus pass-through of the physical-branch supervision keys."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # ForceInputs is shared with legacy flange-F/T configs. Adapt DraftVLA's explicit contact
        # API key at this boundary so a deployment cannot silently send the old flange signal.
        force_data = dict(data)
        force_data["observation/force_torque"] = data["observation/contact_input"]
        inputs = force_policy.ForceInputs(model_type=self.model_type)(force_data)
        # Training only: absent at inference, which is why each key is optional.
        for key in AUX_KEYS:
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class DraftVLAOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the dataset action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # First 7 action dims (TCP velocity x6 + gripper target); the rest is padding.
        outputs = {"actions": np.asarray(data["actions"][:, :7])}
        # Physical-branch readouts, present when the policy sampled through
        # `sample_actions_with_physical` (policy.py). Assemble [mu, sigma] at the serving boundary
        # only -- the model carries mu and sigma separately so the two forms cannot drift out of sync
        # (plan §6.3). Under the 1-D grip contract this is 2 numbers: [mu_hat, sigma_hat].
        if "mu_pred" in data and "sigma_pred" in data:
            outputs["safe_force_distribution"] = np.concatenate(
                [np.asarray(data["mu_pred"]), np.asarray(data["sigma_pred"])], axis=-1
            )
        if "proto_probs" in data:
            outputs["prototype_probs"] = np.asarray(data["proto_probs"])
        if "z_phy" in data:
            # The 128-D physical token itself, logged per frame: the offline representation analyses
            # (linear probe, retrieval, cross-task prototypes) can only be run from it, and a rollout
            # that did not store it has to be repeated (notes/eval_logging_spec.md §4.2).
            outputs["z_phy"] = np.asarray(data["z_phy"])
        return outputs
