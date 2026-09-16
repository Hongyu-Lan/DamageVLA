"""Convert the draftVLA fruit pick-and-place dataset (JSONL + JPEG frames) to LeRobot format.

Source: ``../DamageVLA_training_post_process_20260821`` — 26 episodes, 20,344 frames, UR5e pick &
place over 3 fruits, with per-frame damage-aware supervision labels (see the dataset README).

Two things this converter does that the plain force converter does not:

  1. **Episode denylist** (plan §4.1). All 26 episodes carry non-null physical labels, including the
     4 failed episodes — masking on ``prototype_supervision_valid`` does NOT exclude them because
     they carry valid group labels. The denylist is the only thing that does; denylisted episodes
     are never written.
  2. **Gripper-wrench input**. The model's 12-D force input is ``[mean, signed half-difference]`` of
     ``tactile_estimated_wrenches.left_estimated`` and ``.right_estimated``. The first six values are
     the same mean signal used to build the unchanged 12-D ``gt_safe_distribution`` and soft physical
     prototypes; the last six preserve finger imbalance for action conditioning. The UR flange
     ``force_torque`` is retained in the raw JSONL but is not used by this training dataset.

Images in ``rgb/`` and ``wrist/`` are ALREADY 224x224 — do not re-crop or resize (plan §4, README §9.9).

Usage:
  # Fast smoke conversion (2 episodes) before committing to the full run.
  uv run examples/force/convert_draftvla_data_to_lerobot.py --max-episodes 2 --repo-id draftvla/tactile_smoke

  # Full run (22 successful episodes; the 4 failed ones are skipped by default).
  uv run examples/force/convert_draftvla_data_to_lerobot.py

  # Train-only labels from build_safe_group_prototypes.py --val-episodes (plan §4.3.1 / §8.1).
  uv run examples/force/convert_draftvla_data_to_lerobot.py --labels-dir prototype_metadata_trainonly

The output goes to $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot/<repo_id>), which
the training and norm-stats pipelines read back via the same repo_id.
"""

import argparse
import collections
import json
import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image

# Resolve the complete uploaded dataset next to the code repository. The similarly named directory
# inside the repository is an incomplete transfer and must never be selected implicitly.
DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "DamageVLA_training_post_process_20260821"

# The 4 failed episodes are excluded from every loss. They still carry non-null physical labels, so
# supervision_valid alone cannot remove them from L_flow / L_dist / L_proto.
DEFAULT_EXCLUDE = (
    "pi0_train_20260821_152043",  # pear, failed grasp/place
    "pi0_train_20260821_152329",  # pear, failed grasp/place
    "pi0_train_20260821_155925",  # banana, failed grasp/place
    "pi0_train_20260821_161238",  # banana, failed grasp/place
)

K_PROTOTYPES = 4
FORCE_INPUT_SIGNAL = "tactile_estimated_wrench_mean_plus_signed_half_difference"
WRENCH_LAYOUT = ("fx", "fy", "fz", "tx", "ty", "tz")

# Null-label dummies (plan §5). sigma := 1, NEVER 0: a zero sigma makes the KL's log(sigma_pred/sigma_gt)
# term log(x/0) = +inf, and 0 * inf = NaN, which masking after the reduction cannot rescue.
NULL_SAFE_DISTRIBUTION = np.concatenate([np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)])
NULL_SOFT_PROTOTYPE_TARGET = np.full(K_PROTOTYPES, 1.0 / K_PROTOTYPES, dtype=np.float32)


def _load_records(episode_dir: pathlib.Path) -> list[dict]:
    with (episode_dir / "observations.jsonl").open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["index"])
    return records


def _load_label_sidecar(labels_dir: pathlib.Path, episode_name: str) -> dict[int, dict]:
    """Read a build_safe_group_prototypes.py --val-episodes sidecar: frame index -> label row."""
    path = labels_dir / episode_name / "labels.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"--labels-dir given but {path} does not exist")
    with path.open() as f:
        return {row["index"]: row for row in (json.loads(line) for line in f if line.strip())}


def _two_finger_mean_difference(record: dict, episode_name: str) -> np.ndarray:
    """Build the 12-D model input [two-finger mean, signed left-minus-right half-difference]."""
    try:
        wrenches = record["tactile_estimated_wrenches"]
        left = np.asarray(wrenches["left_estimated"], dtype=np.float64)
        right = np.asarray(wrenches["right_estimated"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{episode_name} frame {record.get('index', '?')}: invalid tactile_estimated_wrenches"
        ) from exc
    if left.shape != (6,) or right.shape != (6,):
        raise ValueError(
            f"{episode_name} frame {record.get('index', '?')}: expected left/right wrench shape (6,), "
            f"got {left.shape} / {right.shape}"
        )
    mean = 0.5 * (left + right)
    difference = 0.5 * (left - right)
    force = np.concatenate([mean, difference])
    if not np.all(np.isfinite(force)):
        raise ValueError(f"{episode_name} frame {record.get('index', '?')}: non-finite tactile wrench")
    return force.astype(np.float32)


def _frame_labels(record: dict, sidecar: dict | None) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """(gt_safe_distribution [12], soft_prototype_target [4], supervision_valid, group_id)."""
    src = sidecar if sidecar is not None else record
    dist = src.get("gt_safe_distribution")
    target = src.get("soft_prototype_target")
    valid = bool(src.get("prototype_supervision_valid", False))
    group_id = src.get("group_id")
    group_id = -1 if group_id is None else int(group_id)

    safe = NULL_SAFE_DISTRIBUTION if dist is None else np.asarray(dist, dtype=np.float32)
    proto = NULL_SOFT_PROTOTYPE_TARGET if target is None else np.asarray(target, dtype=np.float32)
    if safe.shape != (12,):
        raise ValueError(f"gt_safe_distribution has shape {safe.shape}, expected (12,)")
    if proto.shape != (K_PROTOTYPES,):
        raise ValueError(f"soft_prototype_target has shape {proto.shape}, expected ({K_PROTOTYPES},)")
    return safe, proto, valid, group_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Root holding pi0_train_*/ dirs")
    p.add_argument(
        "--repo-id", default="draftvla/fruits_tactile", help="Output LeRobot repo_id (under $HF_LEROBOT_HOME)"
    )
    p.add_argument(
        "--exclude",
        action="append",
        metavar="EP",
        help="Episode denylist, repeatable — never written. Defaults to the 4 failed episodes in the 20260821 batch.",
    )
    p.add_argument(
        "--episodes", action="append", metavar="EP", help="Repeatable; convert only these (denylist still applies)"
    )
    p.add_argument("--max-episodes", type=int, default=None, help="Stop after N episodes (fast smoke conversion)")
    p.add_argument(
        "--labels-dir",
        default=None,
        help="Read per-frame labels from a build_safe_group_prototypes.py --val-episodes sidecar "
        "instead of the shipped in-jsonl labels",
    )
    p.add_argument("--fps", type=int, default=10, help="Nominal sample rate (10 Hz nominal / ~8.7 Hz effective)")
    p.add_argument("--push-to-hub", action="store_true", help="Push the converted dataset to the Hugging Face Hub")
    return p.parse_args()


def main(
    data_dir: str = str(DEFAULT_DATA_DIR),
    repo_id: str = "draftvla/fruits_tactile",
    *,
    exclude: list[str] | None = None,
    episodes: list[str] | None = None,
    max_episodes: int | None = None,
    labels_dir: str | None = None,
    fps: int = 10,
    push_to_hub: bool = False,
) -> None:
    """Convert the uploaded 20260821 DamageVLA batch to a local LeRobot dataset."""
    exclude = list(DEFAULT_EXCLUDE) if exclude is None else exclude
    episodes = episodes or []
    root = pathlib.Path(data_dir)
    all_dirs = sorted(p.parent for p in root.glob("*/observations.jsonl"))
    if not all_dirs:
        raise FileNotFoundError(f"No <episode>/observations.jsonl under {root}")

    denylist = set(exclude)
    unknown = denylist - {d.name for d in all_dirs}
    if unknown:
        raise ValueError(f"Denylisted episode(s) not found under {root}: {sorted(unknown)}")

    selected = all_dirs
    if episodes:
        wanted = set(episodes)
        unknown = wanted - {d.name for d in all_dirs}
        if unknown:
            raise ValueError(f"--episodes not found under {root}: {sorted(unknown)}")
        selected = [d for d in all_dirs if d.name in wanted]

    skipped = [d.name for d in selected if d.name in denylist]
    selected = [d for d in selected if d.name not in denylist]
    if max_episodes is not None:
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("No episodes left to convert after applying the denylist / --episodes / --max-episodes.")

    labels_root = pathlib.Path(labels_dir) if labels_dir else None
    print(f"Data root:   {root}")
    print(f"Labels:      {'sidecar ' + str(labels_root) if labels_root else 'shipped observations.jsonl'}")
    print(f"Denylisted:  {len(skipped)} -> {skipped}")
    print(f"Converting:  {len(selected)} episode(s) -> {repo_id}\n")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=fps,
        features={
            # rgb/ and wrist/ are already 224x224 — the converter must not re-crop (plan §4).
            "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "gripper_wrench": {
                "dtype": "float32",
                "shape": (12,),
                "names": [f"mean_{name}" for name in WRENCH_LAYOUT] + [f"difference_{name}" for name in WRENCH_LAYOUT],
            },
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "gt_safe_distribution": {"dtype": "float32", "shape": (12,), "names": ["gt_safe_distribution"]},
            "soft_prototype_target": {"dtype": "float32", "shape": (4,), "names": ["soft_prototype_target"]},
            # LeRobot represents a scalar as shape (1,) -> datasets.Value (not a Sequence).
            "supervision_valid": {"dtype": "bool", "shape": (1,), "names": ["supervision_valid"]},
            "group_id": {"dtype": "int32", "shape": (1,), "names": ["group_id"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    valid_counts: collections.Counter = collections.Counter()
    for episode_dir in selected:
        records = _load_records(episode_dir)
        sidecar = _load_label_sidecar(labels_root, episode_dir.name) if labels_root else None

        for r in records:
            tcp = r["tcp_pose"]
            state = np.asarray([*tcp["position_xyz"], *tcp["rotation_vector"], r["gripper_width"]], dtype=np.float32)
            gripper_wrench = _two_finger_mean_difference(r, episode_dir.name)
            safe, proto, valid, group_id = _frame_labels(r, sidecar[r["index"]] if sidecar else None)
            valid_counts[valid] += 1
            dataset.add_frame(
                {
                    "image": np.asarray(Image.open(episode_dir / r["image_path"]).convert("RGB")),
                    "wrist_image": np.asarray(Image.open(episode_dir / r["wrist_image_path"]).convert("RGB")),
                    "state": state,
                    "gripper_wrench": gripper_wrench,
                    "actions": np.asarray(r["action_7"], dtype=np.float32),
                    "gt_safe_distribution": safe,
                    "soft_prototype_target": proto,
                    "supervision_valid": np.asarray([valid], dtype=bool),
                    "group_id": np.asarray([group_id], dtype=np.int32),
                    "task": r["prompt"],
                }
            )
        dataset.save_episode()

        fruit = next((r["group_fruit"] for r in records if r.get("group_fruit")), "?")
        total_frames += len(records)
        print(f"Saved {episode_dir.name}  fruit={fruit:<7} frames={len(records):>5}")

    print("\n" + "=" * 100)
    print(f"Episodes converted: {len(selected)}   skipped (denylist): {len(skipped)}   frames: {total_frames}")
    print(f"supervision_valid:  True={valid_counts[True]}  False={valid_counts[False]}")
    print(f"force input:        {FORCE_INPUT_SIGNAL}")
    print("=" * 100)

    # Written last: its presence proves the process completed every selected episode. `info.json`
    # is updated after each episode and therefore cannot distinguish a complete conversion from a
    # process killed halfway through on a login node.
    completion = {
        "repo_id": repo_id,
        "source": str(root.resolve()),
        "episodes": len(selected),
        "frames": total_frames,
        "excluded_episodes": skipped,
        "fps": fps,
        "force_input_signal": FORCE_INPUT_SIGNAL,
    }
    (output_path / ".draftvla_conversion_complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(f"Completion marker: {output_path / '.draftvla_conversion_complete.json'}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["force", "tactile", "ur5e", "forcevla", "draftvla"], private=False, license="apache-2.0"
        )


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        exclude=args.exclude,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        labels_dir=args.labels_dir,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
    )
