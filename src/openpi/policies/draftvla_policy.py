"""DraftVLA (Damage-Aware ForceVLA) robot transforms.

Extends the force-aware transforms with the physical-branch supervision (outlines/draftvla_plan.md
v2.1). The model inputs are unchanged from `ForceInputs` -- two RGB views, the prompt, state(7), and
the 12-D estimated gripper wrench ``[two-finger mean(6), signed half-difference(6)]``. The extra keys
carried here are *labels*, present during training only and never during inference:

    gt_safe_distribution   [12]  stage-level tactile safe distribution, [mu x6, sigma x6]
    soft_prototype_target  [4]   soft prototype target
    supervision_valid      ()    bool mask -- 56.4% of kept frames have a physical label
    group_id               ()    supervision-only, for logging (plan rule 9)

These ride through the transform pipeline as plain dict keys and are split off into the loader's
`aux` dict (`data_loader.AUX_KEYS`). `Observation.from_dict` ignores unknown keys, so `group_id`
provably never enters the model's input PyTree -- that is the point of routing them this way rather
than adding `Observation` fields.

For DraftVLA, ``observation/gripper_wrench`` must be ``concat(0.5*(left+right), 0.5*(left-right))``.
Training and inference must use the same estimator, left/right order, component order, coordinate
convention, and units. The generic model still names the resulting tensor ``force``. The supervised
safe distribution remains the 12-D ``[mu_mean(6), sigma_mean(6)]`` aggregate target.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import force_policy

# Label keys that are supervision, not model input. Kept in sync with data_loader.AUX_KEYS.
AUX_KEYS = ("gt_safe_distribution", "soft_prototype_target", "supervision_valid", "group_id")


def make_draftvla_example() -> dict:
    """Creates a random input example for the DraftVLA policy (inference-shaped: no labels)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/gripper_wrench": np.random.rand(12),
        "prompt": "grasp the banana from the table and place it into the box",
    }


@dataclasses.dataclass(frozen=True)
class DraftVLAInputs(transforms.DataTransformFn):
    """`ForceInputs` plus pass-through of the physical-branch supervision keys."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # ForceInputs is shared with legacy flange-F/T configs. Adapt DraftVLA's explicit tactile
        # API key at this boundary so a deployment cannot silently send the old flange signal.
        force_data = dict(data)
        force_data["observation/force_torque"] = data["observation/gripper_wrench"]
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
        # Assemble the [12] safe-distribution at the serving boundary only -- the model carries mu
        # and sigma separately so the two forms cannot drift out of sync (plan §6.3).
        if "mu_pred" in data and "sigma_pred" in data:
            outputs["safe_force_distribution"] = np.concatenate(
                [np.asarray(data["mu_pred"]), np.asarray(data["sigma_pred"])], axis=-1
            )
        if "proto_probs" in data:
            outputs["prototype_probs"] = np.asarray(data["proto_probs"])
        return outputs
