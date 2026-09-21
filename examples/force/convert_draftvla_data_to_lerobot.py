"""Convert the DamageVLA task1+2 dataset (JSONL + JPEG frames) to LeRobot format.

2026-09-16 contract (Plan B, outlines/todo_training_contract.md): source is the merged
``DamageVLA_training_post_process_20260916`` batch — 83 episodes over 8 conditions (6 fruits +
carton_{empty,full}); 6 are denylisted (draftvla_contact.DEFAULT_EXCLUDE) and never written.

What this converter produces per frame:

  1. **contact_input (57)** — ``[left_data_zeroed(25), right_data_zeroed(25), gripper_width(1),
     force_torque_zeroed(6)]`` (contract §A). Fingertip voltages are zeroed per episode over the
     leading prepare window (draftvla_contact.tactile_zeroing); the flange wrench comes pre-zeroed
     from the post-process pipeline. The estimated tactile wrench is NOT used anywhere any more.
  2. **gt_safe_distribution (2)** = [mu_grip, sigma_grip] and **soft_prototype_target (K=6)** — read
     from the --labels-dir sidecars written by build_safe_group_prototypes.py. The sidecars are
     REQUIRED: the labels baked into observations.jsonl are the retired 12-D wrench contract, and
     split-first (contract §F) demands train-only-fitted tables anyway.

Images in ``rgb/`` and ``wrist/`` are ALREADY 224x224 — do not re-crop or resize.

Usage (always convert train and val separately, from the same labels dir):
  uv run examples/force/convert_draftvla_data_to_lerobot.py \
      --labels-dir prototype_metadata_task12_trainonly \
      --episodes-file <train list> --repo-id draftvla/task12_tactile_train
  uv run examples/force/convert_draftvla_data_to_lerobot.py \
      --labels-dir prototype_metadata_task12_trainonly \
      --episodes-file examples/force/val_episodes_20260917.txt --repo-id draftvla/task12_tactile_val

The output goes to $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot/<repo_id>), which
the training and norm-stats pipelines read back via the same repo_id.
"""

import argparse
import collections
import json
import pathlib
import shutil
import sys

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import draftvla_contact as contact  # noqa: E402

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "DamageVLA_training_post_process_20260916"

DEFAULT_EXCLUDE = contact.DEFAULT_EXCLUDE

K_PROTOTYPES = 6
FORCE_INPUT_SIGNAL = contact.CONTACT_INPUT_SIGNAL

# Null-label dummies for prepare/reset frames (plan §5). sigma := 1, NEVER 0: a zero sigma makes the
# KL's log(sigma_pred/sigma_gt) term log(x/0) = +inf, and 0 * inf = NaN, which masking after the
# reduction cannot rescue.
NULL_SAFE_DISTRIBUTION = np.asarray([0.0, 1.0], dtype=np.float32)
NULL_SOFT_PROTOTYPE_TARGET = np.full(K_PROTOTYPES, 1.0 / K_PROTOTYPES, dtype=np.float32)

# --- v2 (2026-09-22): the gripper action is the teleoperator's BUTTON, not a position -----------
# Teleoperation had exactly three states, and `gripper_action_target` records them as a position:
# holding "close" writes 0.0040, holding "open" writes 0.0650, and with no button held the field is
# a READBACK of the measured width.  Measured over all 62 episodes: 558 close frames (1.6%),
# 719 open frames (2.0%), 34,272 readback frames (96.4%, |target - width| <= 1.67 mm, never above
# 2 mm).  Regressing that as an absolute position teaches the network to echo state[6], and on a
# position-controlled gripper the echo error becomes real motion that accumulates: on 2026-09-21 the
# carry aperture ratcheted open 8.9 mm in 19 s and the filled carton fell out (both conditions
# reproduced it).  Re-encoding the channel as the button removes the regression from "hold"
# entirely -- the client sends no command at all -- so no output error can move the gripper.
GRIPPER_CLOSE_TARGET_M = 0.004
GRIPPER_OPEN_TARGET_M = 0.065
GRIPPER_ANCHOR_TOL_M = 1e-4  # the two commanded values are written bit-exact; the readback is not
GRIPPER_READBACK_MAX_M = 0.002  # readback lags the width by at most 1.67 mm; beyond 2 mm is a data fault
GRIPPER_CLOSE, GRIPPER_HOLD, GRIPPER_OPEN = -1.0, 0.0, 1.0

# Frames whose ACTION is "nothing happens" and whose state is indistinguishable from frames where
# something does: the teleop pauses during the approach, and the whole reset stage (park over the box
# after release -- not part of the task at all).  They are 39% of the dataset and they outnumber the
# demonstrated motion at every hover-like state, which is why the policy hovers at z~0.57 and z~0.40
# on the robot instead of descending.  They stay in the batch -- the physical branch and the
# normalization statistics still see them -- but contribute 5% of their gradient to the action loss.
# The grasp stage is deliberately NOT down-weighted even though 96% of its frames are cartesian-still:
# that stillness is the arm holding position while the gripper closes, and removing it would teach
# the policy to keep descending into the table.
ACTION_WEIGHT_FULL = 1.0
ACTION_WEIGHT_IDLE = 0.05
IDLE_STAGES = ("reset",)

_load_records = contact.load_records


def gripper_button_action(record: dict, episode_name: str) -> float:
    """`gripper_action_target` -> {-1 close, 0 hold, +1 open}: the button the operator was holding."""
    target = float(record["gripper_action_target"])
    if abs(target - GRIPPER_CLOSE_TARGET_M) <= GRIPPER_ANCHOR_TOL_M:
        return GRIPPER_CLOSE
    if abs(target - GRIPPER_OPEN_TARGET_M) <= GRIPPER_ANCHOR_TOL_M:
        return GRIPPER_OPEN
    width = float(record["gripper_width"])
    if abs(target - width) > GRIPPER_READBACK_MAX_M:
        raise ValueError(
            f"{episode_name} frame {record.get('index')}: gripper_action_target {target:.5f} is neither "
            f"a button anchor nor a readback of gripper_width {width:.5f} (delta "
            f"{abs(target - width) * 1e3:.2f} mm > {GRIPPER_READBACK_MAX_M * 1e3:.1f} mm)"
        )
    return GRIPPER_HOLD


def action_loss_weight(record: dict, button: float) -> float:
    """1.0, or ACTION_WEIGHT_IDLE for reset frames and for approach frames where nothing moves."""
    stage = record.get("stage")
    if stage in IDLE_STAGES:
        return ACTION_WEIGHT_IDLE
    if stage == "prepare" and button == GRIPPER_HOLD:
        speeds = record.get("cmd_speed_l") or [0.0] * 6
        if max(abs(float(v)) for v in speeds) < 1e-6:
            return ACTION_WEIGHT_IDLE
    return ACTION_WEIGHT_FULL


def frame_action_7(record: dict, episode_name: str) -> tuple[np.ndarray, float]:
    """The 7-D action with dim 6 re-encoded as the button, plus this frame's action-loss weight."""
    action = np.asarray(record["action_7"], dtype=np.float32).copy()
    if action.shape != (7,):
        raise ValueError(f"{episode_name}: action_7 has shape {action.shape}, expected (7,)")
    button = gripper_button_action(record, episode_name)
    action[6] = button
    return action, action_loss_weight(record, button)


def _load_label_sidecar(labels_dir: pathlib.Path, episode_name: str) -> dict[int, dict]:
    """Read a build_safe_group_prototypes.py --val-episodes sidecar: frame index -> label row."""
    path = labels_dir / episode_name / "labels.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"--labels-dir given but {path} does not exist")
    with path.open() as f:
        return {row["index"]: row for row in (json.loads(line) for line in f if line.strip())}


def _frame_labels(sidecar_row: dict) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """(gt_safe_distribution [2], soft_prototype_target [6], supervision_valid, group_id)."""
    dist = sidecar_row.get("gt_safe_distribution")
    target = sidecar_row.get("soft_prototype_target")
    valid = bool(sidecar_row.get("prototype_supervision_valid", False))
    group_id = sidecar_row.get("group_id")
    group_id = -1 if group_id is None else int(group_id)

    safe = NULL_SAFE_DISTRIBUTION if dist is None else np.asarray(dist, dtype=np.float32)
    proto = NULL_SOFT_PROTOTYPE_TARGET if target is None else np.asarray(target, dtype=np.float32)
    if safe.shape != (2,):
        raise ValueError(f"gt_safe_distribution has shape {safe.shape}, expected (2,)")
    if proto.shape != (K_PROTOTYPES,):
        raise ValueError(f"soft_prototype_target has shape {proto.shape}, expected ({K_PROTOTYPES},)")
    return safe, proto, valid, group_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Root holding pi0_train_*/ dirs")
    p.add_argument(
        "--repo-id", default="draftvla/task12_tactile", help="Output LeRobot repo_id (under $HF_LEROBOT_HOME)"
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
    p.add_argument(
        "--episodes-file", default=None, help="File with one episode name per line (adds to --episodes)"
    )
    p.add_argument("--max-episodes", type=int, default=None, help="Stop after N episodes (fast smoke conversion)")
    p.add_argument(
        "--labels-dir",
        required=True,
        help="Sidecar directory written by build_safe_group_prototypes.py --write. REQUIRED: the "
        "labels baked into observations.jsonl are the retired 12-D wrench contract.",
    )
    p.add_argument("--fps", type=int, default=10, help="Nominal sample rate (10 Hz nominal / ~8.7 Hz effective)")
    p.add_argument("--push-to-hub", action="store_true", help="Push the converted dataset to the Hugging Face Hub")
    return p.parse_args()


def main(
    data_dir: str = str(DEFAULT_DATA_DIR),
    repo_id: str = "draftvla/task12_tactile",
    *,
    labels_dir: str,
    exclude: list[str] | None = None,
    episodes: list[str] | None = None,
    episodes_file: str | None = None,
    max_episodes: int | None = None,
    fps: int = 10,
    push_to_hub: bool = False,
) -> None:
    """Convert the merged 20260916 DamageVLA batch (task1+2) to a local LeRobot dataset."""
    exclude = list(DEFAULT_EXCLUDE) if exclude is None else exclude
    episodes = list(episodes or [])
    if episodes_file:
        lines = pathlib.Path(episodes_file).read_text().splitlines()
        episodes += [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
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

    labels_root = pathlib.Path(labels_dir)
    print(f"Data root:   {root}")
    print(f"Labels:      sidecar {labels_root}")
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
            "contact_input": {
                "dtype": "float32",
                "shape": (contact.CONTACT_INPUT_DIM,),
                "names": (
                    [f"left_data_{i}" for i in range(25)]
                    + [f"right_data_{i}" for i in range(25)]
                    + ["gripper_width"]
                    + [f"ft_zeroed_{d}" for d in ("fx", "fy", "fz", "tx", "ty", "tz")]
                ),
            },
            # dims 0-5 are the commanded TCP twist; dim 6 is the gripper BUTTON (-1/0/+1), not a
            # position -- see gripper_button_action() for why.
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_button"],
            },
            "gt_safe_distribution": {"dtype": "float32", "shape": (2,), "names": ["gt_safe_distribution"]},
            "soft_prototype_target": {"dtype": "float32", "shape": (K_PROTOTYPES,), "names": ["soft_prototype_target"]},
            # LeRobot represents a scalar as shape (1,) -> datasets.Value (not a Sequence).
            "supervision_valid": {"dtype": "bool", "shape": (1,), "names": ["supervision_valid"]},
            "group_id": {"dtype": "int32", "shape": (1,), "names": ["group_id"]},
            "action_loss_weight": {"dtype": "float32", "shape": (1,), "names": ["action_loss_weight"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    valid_counts: collections.Counter = collections.Counter()
    button_counts: collections.Counter = collections.Counter()
    weight_counts: collections.Counter = collections.Counter()
    for episode_dir in selected:
        records = _load_records(episode_dir)
        sidecar = _load_label_sidecar(labels_root, episode_dir.name)
        condition = contact.episode_condition(records, episode_dir.name)
        zeroing = contact.tactile_zeroing(records, episode_dir.name)

        for r in records:
            tcp = r["tcp_pose"]
            state = np.asarray([*tcp["position_xyz"], *tcp["rotation_vector"], r["gripper_width"]], dtype=np.float32)
            contact_input = contact.contact_input_57(r, zeroing, episode_dir.name)
            safe, proto, valid, group_id = _frame_labels(sidecar[r["index"]])
            valid_counts[valid] += 1
            action, weight = frame_action_7(r, episode_dir.name)
            button_counts[float(action[6])] += 1
            weight_counts[weight] += 1
            dataset.add_frame(
                {
                    "image": np.asarray(Image.open(episode_dir / r["image_path"]).convert("RGB")),
                    "wrist_image": np.asarray(Image.open(episode_dir / r["wrist_image_path"]).convert("RGB")),
                    "state": state,
                    "contact_input": contact_input,
                    "actions": action,
                    "gt_safe_distribution": safe,
                    "soft_prototype_target": proto,
                    "supervision_valid": np.asarray([valid], dtype=bool),
                    "group_id": np.asarray([group_id], dtype=np.int32),
                    "action_loss_weight": np.asarray([weight], dtype=np.float32),
                    "task": r["prompt"],
                }
            )
        dataset.save_episode()

        total_frames += len(records)
        print(
            f"Saved {episode_dir.name}  condition={condition:<13} frames={len(records):>5} "
            f"zero_window={zeroing.window_size}"
        )

    print("\n" + "=" * 100)
    print(f"Episodes converted: {len(selected)}   skipped (denylist): {len(skipped)}   frames: {total_frames}")
    print(f"supervision_valid:  True={valid_counts[True]}  False={valid_counts[False]}")
    print(f"force input:        {FORCE_INPUT_SIGNAL}")
    print(
        f"gripper button:     close={button_counts[GRIPPER_CLOSE]}  hold={button_counts[GRIPPER_HOLD]}  "
        f"open={button_counts[GRIPPER_OPEN]}   (62-episode train split expects 558 / 34272 / 719)"
    )
    print(
        f"action weight:      {ACTION_WEIGHT_FULL}={weight_counts[ACTION_WEIGHT_FULL]}  "
        f"{ACTION_WEIGHT_IDLE}={weight_counts[ACTION_WEIGHT_IDLE]}  "
        f"({weight_counts[ACTION_WEIGHT_IDLE] / max(total_frames, 1):.0%} down-weighted; train split expects 39%)"
    )
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
        "labels_dir": str(labels_root.resolve()),
        "fps": fps,
        "force_input_signal": FORCE_INPUT_SIGNAL,
        "contact_input_dim": contact.CONTACT_INPUT_DIM,
        "k_prototypes": K_PROTOTYPES,
        # v2 action contract: dim 6 is the teleop button, not a position. A server built from this
        # dataset must be decoded with the matching client rule (|v| <= 0.5 -> send nothing).
        "gripper_action_encoding": "button_v2",
        "gripper_button_values": {"close": GRIPPER_CLOSE, "hold": GRIPPER_HOLD, "open": GRIPPER_OPEN},
        "gripper_button_counts": {
            "close": button_counts[GRIPPER_CLOSE],
            "hold": button_counts[GRIPPER_HOLD],
            "open": button_counts[GRIPPER_OPEN],
        },
        "action_loss_weight": {
            "full": ACTION_WEIGHT_FULL,
            "idle": ACTION_WEIGHT_IDLE,
            "idle_stages": list(IDLE_STAGES),
            "idle_frames": weight_counts[ACTION_WEIGHT_IDLE],
        },
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
        labels_dir=args.labels_dir,
        exclude=args.exclude,
        episodes=args.episodes,
        episodes_file=args.episodes_file,
        max_episodes=args.max_episodes,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
    )
