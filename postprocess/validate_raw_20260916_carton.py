#!/usr/bin/env python3
"""Read-only structural, sensor, stage, and image audit for the raw 0916 carton batch."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from postprocess_20260916_carton import (
    STAGES,
    compute_ordered_stages,
    finite_vector,
    fruit_from_prompt,
    load_jsonl,
    physical_condition,
    summarize_segments,
    validate_stage_order,
)


def describe(values: list[list[float]]) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "p01": np.percentile(array, 1, axis=0).tolist(),
        "median": np.median(array, axis=0).tolist(),
        "p99": np.percentile(array, 99, axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "zero_fraction": np.mean(array == 0.0, axis=0).tolist(),
    }


def rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def source_ages(record: dict) -> tuple[dict[str, float], list[str]]:
    stamp = float(record["stamp_sec"])
    stamps = record["source_stamps"]
    values = {
        "image": stamps.get("image"),
        "wrist_image": stamps.get("wrist_image"),
        "joint_states": stamps.get("joint_states"),
        "force": stamps.get("force"),
        "cmd_speed_l": stamps.get("cmd_speed_l"),
        "gripper_command_label": stamps.get("gripper_command_label"),
    }
    for source, value in stamps.get("tactile_voltage_signals", {}).items():
        values[f"tactile_voltage.{source}"] = value
    for source, value in stamps.get("tactile_estimated_wrenches", {}).items():
        values[f"estimated_wrench.{source}"] = value
    nulls = [key for key, value in values.items() if value is None]
    return ({key: abs(stamp - float(value)) for key, value in values.items() if value is not None}, nulls)


def check_images(episode: Path, records: list[dict], errors: list[str]) -> None:
    for camera, field in (("rgb", "image_path"), ("wrist", "wrist_image_path")):
        paths = sorted((episode / camera).glob("*.jpg"))
        expected = [Path(record[field]).name for record in records]
        if [path.name for path in paths] != expected:
            errors.append(f"{episode.name}/{camera}: names differ from JSONL")
            continue
        for path in paths:
            try:
                with Image.open(path) as image:
                    if image.size != (1920, 1080) or image.mode != "RGB":
                        errors.append(f"{path}: size={image.size} mode={image.mode}")
                    image.verify()
            except Exception as exc:
                errors.append(f"{path}: unreadable: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    episodes = sorted(path for path in args.root.glob("pi0_train_*") if path.is_dir())
    errors, warnings, details = [], [], []
    fruits, conditions, stage_counts = Counter(), Counter(), Counter()
    threshold_diffs = []
    force, tactile, wrench = [], defaultdict(list), defaultdict(list)
    max_age, over_02, null_stamps = defaultdict(float), Counter(), Counter()
    post_fields = {
        "stage", "force_torque_stage_mean", "force_torque_chunk_mean",
        "tactile_estimated_wrench_stage_mean", "group_id", "force_torque_zeroed",
    }
    for episode in episodes:
        path = episode / "observations.jsonl"
        try:
            records = load_jsonl(path)
            if not records:
                raise ValueError(f"{path}: empty")
            if [record.get("index") for record in records] != list(range(len(records))):
                raise ValueError(f"{path}: non-contiguous indices")
            stamps = [float(record["stamp_sec"]) for record in records]
            if any(right <= left for left, right in zip(stamps, stamps[1:])):
                raise ValueError(f"{path}: timestamps not strictly increasing")
            prompts = {record.get("prompt") for record in records}
            if len(prompts) != 1 or not isinstance(next(iter(prompts)), str):
                raise ValueError(f"{path}: prompt missing or changes")
            fruit = fruit_from_prompt(str(records[0]["prompt"]), path)
            fruits[fruit] += 1
            object_category, load_condition, object_id = physical_condition(records[0], path)
            conditions[f"{object_category}/{load_condition}"] += 1
            if any(
                physical_condition(record, path)
                != (object_category, load_condition, object_id)
                for record in records
            ):
                raise ValueError(f"{path}: episode_context changes within episode")
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            if metadata["parameters"]["prompt"] != records[0]["prompt"]:
                raise ValueError(f"{episode.name}: metadata prompt mismatch")
            if metadata.get("episode_context") != records[0].get("episode_context"):
                raise ValueError(f"{episode.name}: metadata episode_context mismatch")
            present = post_fields.intersection(records[0])
            if present:
                raise ValueError(f"{path}: already has postprocess fields {sorted(present)}")
            maximum_jump = np.zeros(6)
            previous = None
            for index, record in enumerate(records):
                prefix = f"{path}:{index}"
                if record.get("image_shape") != [1080, 1920, 3]:
                    raise ValueError(f"{prefix}: invalid image_shape")
                if record.get("wrist_image_shape") != [1080, 1920, 3]:
                    raise ValueError(f"{prefix}: invalid wrist_image_shape")
                finite_vector(record.get("aloha_state_14"), 14, prefix + ".aloha_state_14")
                finite_vector(record.get("action_7"), 7, prefix + ".action_7")
                finite_vector(record.get("cmd_speed_l"), 6, prefix + ".cmd_speed_l")
                value = finite_vector(record.get("force_torque"), 6, prefix + ".force_torque")
                force.append(value)
                if previous is not None:
                    maximum_jump = np.maximum(maximum_jump, np.abs(np.asarray(value) - previous))
                previous = np.asarray(value)
                tv = record.get("tactile_voltage_signals")
                if not isinstance(tv, dict) or set(tv) != {"left_raw", "left_data", "right_raw", "right_data"}:
                    raise ValueError(f"{prefix}: invalid tactile voltage sources")
                for source, values in tv.items():
                    tactile[source].append(finite_vector(values, 25, prefix + "." + source))
                tw = record.get("tactile_estimated_wrenches")
                if not isinstance(tw, dict) or set(tw) != {"left_estimated", "right_estimated"}:
                    raise ValueError(f"{prefix}: invalid tactile wrench sources")
                for source, values in tw.items():
                    wrench[source].append(finite_vector(values, 6, prefix + "." + source))
                ages, nulls = source_ages(record)
                for source, age in ages.items():
                    max_age[source] = max(max_age[source], age)
                    if age > 0.2:
                        over_02[source] += 1
                null_stamps.update(nulls)
                if "gripper_command_label" in nulls and record.get("gripper_action_source") != "current_gripper_width_fallback":
                    raise ValueError(f"{prefix}: invalid null gripper stamp fallback")
            stages, events = compute_ordered_stages(records, path)
            validate_stage_order(stages, path)
            segments = summarize_segments(stages)
            if [item[0] for item in segments] != list(STAGES) or events["release_frame"] is None:
                raise ValueError(f"{path}: incomplete stage sequence/release")
            stage_counts.update(stages)
            sensitive, _ = compute_ordered_stages(records, path, close_delta_m=0.00025)
            if sensitive.index("grasp") != stages.index("grasp"):
                threshold_diffs.append({
                    "episode": episode.name,
                    "sensitive_close": sensitive.index("grasp"),
                    "robust_close": stages.index("grasp"),
                })
            details.append({
                "episode": episode.name,
                "fruit": fruit,
                "load_condition": load_condition,
                "object_id": object_id,
                "frames": len(records),
                "segments": segments,
                "release_frame": events["release_frame"],
                "maximum_force_jump": maximum_jump.tolist(),
            })
            if args.check_images:
                check_images(episode, records, errors)
        except Exception as exc:
            errors.append(str(exc))
    if over_02:
        warnings.append("some source timestamps differ from observation by more than 0.2 s")
    report = rounded({
        "root": str(args.root.resolve()),
        "episodes": len(episodes),
        "frames": sum(item["frames"] for item in details),
        "fruit_episode_counts": dict(fruits),
        "physical_condition_episode_counts": dict(conditions),
        "stage_counts_close_delta_0.005": {stage: stage_counts[stage] for stage in STAGES},
        "stage_threshold_differences": threshold_diffs,
        "maximum_source_age_sec": dict(max_age),
        "source_age_over_0.2_frame_counts": dict(over_02),
        "null_source_stamp_counts": dict(null_stamps),
        "force_torque": describe(force) if force else {},
        "tactile_voltage_signals": {key: describe(value) for key, value in sorted(tactile.items())},
        "tactile_estimated_wrenches": {key: describe(value) for key, value in sorted(wrench.items())},
        "episodes_detail": details,
        "warnings": warnings,
        "errors": errors,
    })
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
