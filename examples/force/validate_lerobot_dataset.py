"""Validate that the local DraftVLA tactile conversion completed rather than stopping mid-run."""

import argparse
import json
import os
import pathlib

EXPECTED_FORCE_INPUT = "tactile_data_zeroed_50_plus_gripper_width_plus_force_torque_zeroed"


def main(
    repo_id: str = "draftvla/task12_tactile_train",
    expected_episodes: int = 22,
    expected_frames: int = 16_904,
    expected_force_input: str = EXPECTED_FORCE_INPUT,
) -> None:
    default_hf_home = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    lerobot_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", str(default_hf_home / "lerobot")))
    root = lerobot_home / repo_id
    info_path = root / "meta" / "info.json"
    marker_path = root / ".draftvla_conversion_complete.json"
    if not info_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"incomplete conversion: expected both {info_path} and {marker_path}")

    info = json.loads(info_path.read_text())
    marker = json.loads(marker_path.read_text())
    errors = []
    for source, name in ((info, "info.json"), (marker, "completion marker")):
        if source.get("total_episodes", source.get("episodes")) != expected_episodes:
            errors.append(f"{name}: expected {expected_episodes} episodes, got {source}")
        if source.get("total_frames", source.get("frames")) != expected_frames:
            errors.append(f"{name}: expected {expected_frames} frames, got {source}")
    if marker.get("force_input_signal") != expected_force_input:
        errors.append(
            f"completion marker: expected force_input_signal={expected_force_input!r}, "
            f"got {marker.get('force_input_signal')!r}"
        )
    contact_feature = info.get("features", {}).get("contact_input", {})
    if contact_feature.get("shape") != [57]:
        errors.append(f"info.json: expected contact_input shape [57], got {contact_feature.get('shape')}")
    safe_feature = info.get("features", {}).get("gt_safe_distribution", {})
    if safe_feature.get("shape") != [2]:
        errors.append(f"info.json: expected gt_safe_distribution shape [2] (mu, sigma), got {safe_feature}")
    proto_feature = info.get("features", {}).get("soft_prototype_target", {})
    if proto_feature.get("shape") != [6]:
        errors.append(f"info.json: expected soft_prototype_target shape [6] (K=6), got {proto_feature}")

    episode_rows = root / "meta" / "episodes.jsonl"
    if not episode_rows.is_file() or sum(1 for line in episode_rows.open() if line.strip()) != expected_episodes:
        errors.append(f"{episode_rows}: expected {expected_episodes} non-empty rows")
    parquet_files = list(root.glob("data/chunk-*/episode_*.parquet"))
    if len(parquet_files) != expected_episodes:
        errors.append(f"expected {expected_episodes} episode parquet files, found {len(parquet_files)}")
    if errors:
        raise ValueError("LeRobot validation failed:\n  " + "\n  ".join(errors))

    print(f"LeRobot dataset validation: PASS ({repo_id}, {expected_episodes} episodes, {expected_frames} frames)")
    print(f"  force input: {expected_force_input}")
    print(f"  path: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="draftvla/fruits_tactile")
    parser.add_argument("--expected-episodes", type=int, default=22)
    parser.add_argument("--expected-frames", type=int, default=16_904)
    parser.add_argument("--expected-force-input", default=EXPECTED_FORCE_INPUT)
    main(**vars(parser.parse_args()))
