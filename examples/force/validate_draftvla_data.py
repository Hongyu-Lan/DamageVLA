"""Validate the uploaded DamageVLA JSONL/image dataset before conversion or training."""

import argparse
import collections
import json
import math
import pathlib

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "DamageVLA_training_post_process_20260821"
EXPECTED_EXCLUDED = {
    "pi0_train_20260821_152043",
    "pi0_train_20260821_152329",
    "pi0_train_20260821_155925",
    "pi0_train_20260821_161238",
}
REQUIRED = {
    "index",
    "prompt",
    "image_path",
    "wrist_image_path",
    "tcp_pose",
    "gripper_width",
    "tactile_estimated_wrenches",
    "action_7",
    "stage",
    "prototype_supervision_valid",
}
STAGES = {"prepare", "grasp", "lift", "translate", "place", "reset"}


def main(
    data_dir: pathlib.Path,
    expected_episodes: int = 26,
    expected_frames: int = 20_344,
    expected_excluded: set[str] | None = None,
) -> None:
    expected_excluded = EXPECTED_EXCLUDED if expected_excluded is None else expected_excluded
    episode_dirs = sorted(path.parent for path in data_dir.glob("pi0_train_*/observations.jsonl"))
    errors: list[str] = []
    if len(episode_dirs) != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, found {len(episode_dirs)}")

    names = {path.name for path in episode_dirs}
    if missing := expected_excluded - names:
        errors.append(f"expected failed episodes are missing: {sorted(missing)}")

    total_frames = 0
    valid_frames = 0
    excluded_frames = 0
    excluded_valid_frames = 0
    fruits: collections.Counter[str] = collections.Counter()
    stage_counts: collections.Counter[str] = collections.Counter()

    for episode_dir in episode_dirs:
        seen_indices: set[int] = set()
        with (episode_dir / "observations.jsonl").open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                total_frames += 1
                row = json.loads(line)
                prefix = f"{episode_dir.name}:{line_number}"

                if missing := REQUIRED - row.keys():
                    errors.append(f"{prefix}: missing fields {sorted(missing)}")
                    continue
                index = row["index"]
                if index in seen_indices:
                    errors.append(f"{prefix}: duplicate frame index {index}")
                seen_indices.add(index)

                if len(row["action_7"]) != 7:
                    errors.append(f"{prefix}: expected action[7]")
                tactile = row["tactile_estimated_wrenches"]
                if not isinstance(tactile, dict):
                    errors.append(f"{prefix}: tactile_estimated_wrenches must be a mapping")
                else:
                    for side in ("left_estimated", "right_estimated"):
                        values = tactile.get(side)
                        if not isinstance(values, list) or len(values) != 6:
                            errors.append(f"{prefix}: expected tactile {side}[6]")
                        elif any(not isinstance(value, int | float) or not math.isfinite(value) for value in values):
                            errors.append(f"{prefix}: tactile {side} contains a non-finite/non-numeric value")
                if row["stage"] not in STAGES:
                    errors.append(f"{prefix}: invalid stage {row['stage']!r}")
                stage_counts[row["stage"]] += 1

                for key in ("image_path", "wrist_image_path"):
                    image_path = episode_dir / row[key]
                    if not image_path.is_file():
                        errors.append(f"{prefix}: missing {key} {row[key]!r}")
                if row.get("image_shape") != [224, 224, 3] or row.get("wrist_image_shape") != [224, 224, 3]:
                    errors.append(f"{prefix}: image metadata is not 224x224x3")

                is_valid = bool(row["prototype_supervision_valid"])
                if is_valid:
                    valid_frames += 1
                    safe = row.get("gt_safe_distribution") or []
                    target = row.get("soft_prototype_target") or []
                    if len(safe) != 12 or len(target) != 4:
                        errors.append(f"{prefix}: valid supervision must have safe[12] and target[4]")
                    elif abs(sum(target) - 1.0) > 1e-5:
                        errors.append(f"{prefix}: soft prototype probabilities do not sum to one")
                    if fruit := row.get("group_fruit"):
                        fruits[fruit] += 1
                if episode_dir.name in expected_excluded:
                    excluded_frames += 1
                    excluded_valid_frames += int(is_valid)

                if len(errors) >= 50:
                    raise ValueError("Dataset validation failed (first 50 errors):\n  " + "\n  ".join(errors))

        if seen_indices and seen_indices != set(range(len(seen_indices))):
            errors.append(f"{episode_dir.name}: frame indices are not contiguous from zero")

    if total_frames != expected_frames:
        errors.append(f"expected {expected_frames} frames, found {total_frames}")
    if errors:
        raise ValueError("Dataset validation failed:\n  " + "\n  ".join(errors))

    kept_frames = total_frames - excluded_frames
    kept_valid = valid_frames - excluded_valid_frames
    print("DamageVLA dataset validation: PASS")
    print(f"  source:                  {data_dir.resolve()}")
    print(f"  episodes / frames:       {len(episode_dirs)} / {total_frames}")
    print(f"  excluded failed data:    {len(expected_excluded)} episodes / {excluded_frames} frames")
    print(f"  conversion keeps:        {len(episode_dirs) - len(expected_excluded)} episodes / {kept_frames} frames")
    print(f"  supervised kept frames:  {kept_valid} ({kept_valid / kept_frames:.1%})")
    print("  force input signal:       [mean(left, right), signed half-difference(left, right)] [12]")
    print(f"  valid frames by fruit:   {dict(sorted(fruits.items()))}")
    print(f"  frames by stage:         {dict(stage_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--expected-episodes", type=int, default=26)
    parser.add_argument("--expected-frames", type=int, default=20_344)
    parser.add_argument(
        "--exclude",
        action="append",
        dest="expected_excluded",
        help="Expected failed episode; repeat for each episode. Defaults to the 20260821 denylist.",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
        expected_excluded=set(args.expected_excluded) if args.expected_excluded is not None else None,
    )
