import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_force_example() -> dict:
    """Creates a random input example for the force-aware policy (UR5e + 6-axis F/T sensor)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/force_torque": np.random.rand(6),
        "prompt": "grasp the banana",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ForceInputs(transforms.DataTransformFn):
    """Converts force-aware dataset inputs into the model's expected format (training + inference).

    Mirrors LiberoInputs, with one extra modality: the 6-axis force/torque reading is read from
    "observation/force_torque" and emitted under the top-level "force" key, which flows into
    Observation.force and is z-score normalized upstream via norm stats keyed by "force".

    For your own dataset, copy this class and adjust the keys/dims below.
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot stores images as float32 (C,H,W); parse back to uint8 (H,W,C). Skipped at inference.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # No second wrist camera in this dataset: pad with zeros (masked out for pi0 below).
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0, not pi0-FAST. Do not change this.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            # 6-axis force/torque -> model's Observation.force (normalized upstream like state).
            "force": data["observation/force_torque"],
        }

        # Actions are only available during training. Padded to the model action dim downstream.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ForceOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the dataset action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Return the first 7 actions (TCP velocity x6 + gripper target); the rest is padding.
        return {"actions": np.asarray(data["actions"][:, :7])}
