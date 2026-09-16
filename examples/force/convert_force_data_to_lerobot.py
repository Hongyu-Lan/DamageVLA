"""Convert the custom force/torque dataset (JSONL + JPEG frames) to LeRobot format.

The raw dataset is a directory (or directories) each containing:
  - observations.jsonl : one JSON record per timestep
  - rgb/000000.jpg ...  : base camera frames (224x224)
  - wrist/000000.jpg .. : wrist camera frames (224x224)

Each record provides ``tcp_pose`` (position_xyz + rotation_vector), ``gripper_width``,
``force_torque`` (6-axis wrench), ``action_7`` (TCP velocity x6 + gripper target), and a ``prompt``.
We assemble the model inputs as:
  state   = position_xyz(3) + rotation_vector(3) + gripper_width(1)   -> (7,)
  force_torque                                                        -> (6,)
  actions = action_7                                                  -> (7,)   (no delta conversion)

Usage:
  uv run examples/force/convert_force_data_to_lerobot.py --data_dir data --repo_id force/banana

Every ``observations.jsonl`` found under --data_dir (recursively) becomes one LeRobot episode, so
adding more episode folders later requires no code change. The output is written to
$HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot/<repo_id>), which the training and
norm-stats pipelines read back via the same repo_id (no Hugging Face Hub round-trip needed).
"""

import json
import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro


def _load_records(episode_dir: pathlib.Path) -> list[dict]:
    with (episode_dir / "observations.jsonl").open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["index"])
    return records


def main(data_dir: str = "data", repo_id: str = "force/banana", *, fps: int = 5, push_to_hub: bool = False):
    root = pathlib.Path(data_dir)
    episode_dirs = sorted({p.parent for p in root.glob("**/observations.jsonl")})
    if not episode_dirs:
        raise FileNotFoundError(f"No observations.jsonl found under {root}")
    print(f"Found {len(episode_dirs)} episode(s): {[str(d) for d in episode_dirs]}")

    # Clean up any existing dataset in the output directory.
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # OpenPi assumes proprio is stored in `state` and actions in `actions`; LeRobot stores RGB as
    # the `image` dtype. `force_torque` is an extra plain float feature (like `state`).
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=fps,
        features={
            "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "force_torque": {"dtype": "float32", "shape": (6,), "names": ["force_torque"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Each observations.jsonl is one episode; save_episode() delimits episodes.
    for episode_dir in episode_dirs:
        records = _load_records(episode_dir)
        for r in records:
            tcp = r["tcp_pose"]
            state = np.asarray([*tcp["position_xyz"], *tcp["rotation_vector"], r["gripper_width"]], dtype=np.float32)
            base_image = np.asarray(Image.open(episode_dir / r["image_path"]).convert("RGB"))
            wrist_image = np.asarray(Image.open(episode_dir / r["wrist_image_path"]).convert("RGB"))
            dataset.add_frame(
                {
                    "image": base_image,
                    "wrist_image": wrist_image,
                    "state": state,
                    "force_torque": np.asarray(r["force_torque"], dtype=np.float32),
                    "actions": np.asarray(r["action_7"], dtype=np.float32),
                    "task": r["prompt"],
                }
            )
        dataset.save_episode()
        print(f"Saved episode from {episode_dir} ({len(records)} frames)")

    if push_to_hub:
        dataset.push_to_hub(tags=["force", "ur5e", "forcevla"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
