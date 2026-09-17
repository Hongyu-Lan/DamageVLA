"""Validate the merged DamageVLA task1+2 dataset (+ label sidecars) before conversion or training.

2026-09-16 contract: checks the RAW per-frame structure the converter consumes (images, stages,
tactile `_data` voltages, pre-zeroed flange wrench) and, when --labels-dir is given, the sidecar
labels written by build_safe_group_prototypes.py (gt_safe_distribution[2], soft target[K=6]).
The retired in-jsonl 12-D wrench labels are deliberately NOT validated -- nothing reads them now.
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import draftvla_contact as contact  # noqa: E402

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "DamageVLA_training_post_process_20260916"
K_PROTOTYPES = 6
REQUIRED = {
    "index",
    "prompt",
    "image_path",
    "wrist_image_path",
    "tcp_pose",
    "gripper_width",
    "tactile_voltage_signals",
    "force_torque_zeroed",
    "action_7",
    "stage",
}
STAGES = {"prepare", "grasp", "lift", "translate", "place", "reset"}


def main(
    data_dir: pathlib.Path,
    labels_dir: pathlib.Path | None,
    expected_episodes: int = 83,
    expected_frames: int = 47_591,
) -> None:
    expected_excluded = set(contact.DEFAULT_EXCLUDE)
    episode_dirs = sorted(path.parent for path in data_dir.glob("pi0_train_*/observations.jsonl"))
    errors: list[str] = []
    if len(episode_dirs) != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, found {len(episode_dirs)}")

    total_frames = 0
    excluded_frames = 0
    valid_frames = 0
    conditions: collections.Counter = collections.Counter()
    stage_counts: collections.Counter = collections.Counter()

    for episode_dir in episode_dirs:
        records = contact.load_records(episode_dir)
        condition = contact.episode_condition(records, episode_dir.name)
        try:
            zeroing = contact.tactile_zeroing(records, episode_dir.name)
        except ValueError as exc:
            errors.append(str(exc))
            zeroing = None

        sidecar = None
        if labels_dir is not None:
            sidecar_path = labels_dir / episode_dir.name / "labels.jsonl"
            if not sidecar_path.is_file():
                errors.append(f"{episode_dir.name}: missing sidecar {sidecar_path}")
            else:
                with sidecar_path.open() as f:
                    sidecar = {row["index"]: row for row in (json.loads(ln) for ln in f if ln.strip())}

        seen_indices: set[int] = set()
        for row in records:
            prefix = f"{episode_dir.name}[{row.get('index', '?')}]"
            seen_indices.add(row["index"])
            missing = REQUIRED - row.keys()
            if missing:
                errors.append(f"{prefix}: missing keys {sorted(missing)}")
                continue
            if row["stage"] not in STAGES:
                errors.append(f"{prefix}: invalid stage {row['stage']!r}")
            stage_counts[row["stage"]] += 1

            for key in ("image_path", "wrist_image_path"):
                if not (episode_dir / row[key]).is_file():
                    errors.append(f"{prefix}: missing {key} {row[key]!r}")
            if row.get("image_shape") != [224, 224, 3] or row.get("wrist_image_shape") != [224, 224, 3]:
                errors.append(f"{prefix}: image metadata is not 224x224x3")

            if zeroing is not None:
                try:
                    contact.contact_input_57(row, zeroing, episode_dir.name)
                except ValueError as exc:
                    errors.append(str(exc))

            if sidecar is not None:
                srow = sidecar.get(row["index"])
                if srow is None:
                    errors.append(f"{prefix}: no sidecar row")
                elif srow["prototype_supervision_valid"]:
                    valid_frames += 1
                    safe = srow.get("gt_safe_distribution") or []
                    target = srow.get("soft_prototype_target") or []
                    if len(safe) != 2 or len(target) != K_PROTOTYPES:
                        errors.append(f"{prefix}: valid supervision must have safe[2] and target[{K_PROTOTYPES}]")
                    elif safe[1] < contact.GRIP_SIGMA_FLOOR - 1e-12:
                        errors.append(f"{prefix}: sigma {safe[1]} below floor {contact.GRIP_SIGMA_FLOOR}")
                    elif abs(sum(target) - 1.0) > 1e-5:
                        errors.append(f"{prefix}: soft prototype probabilities do not sum to one")
                    elif (row["stage"] not in contact.CONTACT_STAGES) or srow["group_id"] != contact.GROUP_INDEX[
                        (contact.TASK, condition, contact.STAGE_TO_GROUP[row["stage"]])
                    ]:
                        errors.append(f"{prefix}: group_id {srow['group_id']} inconsistent with stage/condition")

            if len(errors) >= 50:
                raise ValueError("Dataset validation failed (first 50 errors):\n  " + "\n  ".join(errors))

        conditions[condition] += 1
        total_frames += len(records)
        if episode_dir.name in expected_excluded:
            excluded_frames += len(records)
        if seen_indices != set(range(len(seen_indices))):
            errors.append(f"{episode_dir.name}: frame indices are not contiguous from zero")

    if total_frames != expected_frames:
        errors.append(f"expected {expected_frames} frames, found {total_frames}")
    unknown_excluded = expected_excluded - {d.name for d in episode_dirs}
    if unknown_excluded:
        errors.append(f"denylisted episode(s) not on disk: {sorted(unknown_excluded)}")
    if errors:
        raise ValueError("Dataset validation failed:\n  " + "\n  ".join(errors))

    kept_frames = total_frames - excluded_frames
    print("DamageVLA task1+2 dataset validation: PASS")
    print(f"  source:                  {data_dir.resolve()}")
    print(f"  labels:                  {labels_dir.resolve() if labels_dir else '(not checked)'}")
    print(f"  episodes / frames:       {len(episode_dirs)} / {total_frames}")
    print(f"  denylisted:              {len(expected_excluded)} episodes / {excluded_frames} frames")
    print(f"  conversion keeps:        {len(episode_dirs) - len(expected_excluded)} episodes / {kept_frames} frames")
    if labels_dir is not None:
        print(f"  supervised frames:       {valid_frames} (over ALL episodes incl. denylist)")
    print(f"  contact input:            {contact.CONTACT_INPUT_SIGNAL} [{contact.CONTACT_INPUT_DIM}]")
    print(f"  episodes by condition:   {dict(sorted(conditions.items()))}")
    print(f"  frames by stage:         {dict(stage_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--labels-dir", default=None, help="Sidecar dir from build_safe_group_prototypes.py")
    parser.add_argument("--expected-episodes", type=int, default=83)
    parser.add_argument("--expected-frames", type=int, default=47_591)
    args = parser.parse_args()
    main(
        pathlib.Path(args.data_dir),
        pathlib.Path(args.labels_dir) if args.labels_dir else None,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
    )
