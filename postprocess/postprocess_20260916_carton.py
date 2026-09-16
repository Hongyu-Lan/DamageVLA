#!/usr/bin/env python3
"""Non-image post-processing for the 2026-09-16 carton DamageVLA batch.

This follows the accepted 2026-09-05 pipeline: ordered macro stages, flange
and tactile stage/future-chunk statistics, group-level safe distributions and
soft prototypes, and per-episode initial-static flange F/T zeroing. Carton
load condition is part of the group key so empty and full cartons retain
distinct physical supervision. Camera images and image metadata are
deliberately outside this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np


STAGES = ("prepare", "grasp", "lift", "translate", "place", "reset")
SUPERVISED_STAGES = ("grasp", "lift", "translate", "place")
LAYOUT = ["fx", "fy", "fz", "tx", "ty", "tz"]
DISTRIBUTION_LAYOUT = [f"mu_{item}" for item in LAYOUT] + [
    f"sigma_{item}" for item in LAYOUT
]
TASK = "pick_place"
SIGNAL = "tactile_estimated_wrench_two_finger_mean"
ZEROING_METHOD = "per_episode_initial_static_open_median"
ZEROING_FALLBACK_METHOD = "per_episode_initial_open_adaptive_low_motion_median"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def backup_once(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)


def finite_vector(value: object, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label}: expected list[{size}]")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: non-numeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label}: non-finite value")
    return result


def fruit_from_prompt(prompt: str, path: Path) -> str:
    match = re.search(r"\bgrasp the (.+?) from the table\b", prompt.lower())
    if not match:
        raise ValueError(f"{path}: cannot parse fruit from {prompt!r}")
    fruit = match.group(1).strip()
    if not fruit:
        raise ValueError(f"{path}: empty fruit name")
    return fruit


def physical_condition(record: dict, path: Path) -> tuple[str, str, str]:
    """Return (object category, load condition, object instance id)."""
    object_category = fruit_from_prompt(str(record.get("prompt", "")), path)
    context = record.get("episode_context")
    if not isinstance(context, dict):
        raise ValueError(f"{path}: missing episode_context")
    load_condition = context.get("load_condition")
    object_id = context.get("object_id")
    if not isinstance(load_condition, str) or not load_condition:
        raise ValueError(f"{path}: invalid episode_context.load_condition")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError(f"{path}: invalid episode_context.object_id")
    return object_category, load_condition, object_id


def first_sustained(
    length: int,
    start: int,
    predicate: Callable[[int], bool],
    minimum_frames: int,
) -> int | None:
    run_start = None
    for index in range(max(0, start), length):
        if predicate(index):
            run_start = index if run_start is None else run_start
            if index - run_start + 1 >= minimum_frames:
                return run_start
        else:
            run_start = None
    return None


def compute_ordered_stages(
    records: list[dict],
    path: Path,
    close_delta_m: float = 0.005,
    release_delta_m: float = 0.001,
    command_threshold: float = 0.0001,
    minimum_command_frames: int = 2,
) -> tuple[list[str], dict[str, int | None]]:
    if not records:
        return [], {key: None for key in (
            "close_start", "lift_start", "translate_start", "place_start",
            "release_frame", "reset_start",
        )}
    widths = [float(record["gripper_width"]) for record in records]
    targets = [
        float(record.get("gripper_action_target", record["gripper_width"]))
        for record in records
    ]
    commands = [
        finite_vector(record.get("cmd_speed_l"), 6, f"{path}:{index}.cmd_speed_l")
        for index, record in enumerate(records)
    ]
    count = len(records)
    initial_width = widths[0]
    threshold = targets[0] - close_delta_m

    def closed(index: int) -> bool:
        return targets[index] < threshold

    lift_start = first_sustained(
        count,
        0,
        lambda index: closed(index) and commands[index][2] > command_threshold,
        minimum_command_frames,
    )
    if lift_start is not None:
        last_open = max(
            (index for index in range(lift_start) if not closed(index)), default=-1
        )
        close_start = last_open + 1
    else:
        close_start = first_sustained(count, 0, closed, minimum_command_frames)

    translate_start = None
    if lift_start is not None:
        translate_start = first_sustained(
            count,
            lift_start,
            lambda index: (
                commands[index][2] <= command_threshold
                and max(abs(commands[index][axis]) for axis in (0, 1, 3, 4, 5))
                > command_threshold
            ),
            minimum_command_frames,
        )
    motion_start = translate_start if translate_start is not None else (
        lift_start if lift_start is not None else count
    )
    place_start = first_sustained(
        count,
        motion_start,
        lambda index: commands[index][2] < -command_threshold,
        minimum_command_frames,
    )
    release_frame = reset_start = None
    if place_start is not None:
        open_threshold = initial_width - release_delta_m
        release_frame = next(
            (index for index in range(place_start, count) if widths[index] >= open_threshold),
            None,
        )
        if release_frame is not None:
            reset_start = release_frame + 1
        else:
            reset_start = first_sustained(
                count,
                place_start + 1,
                lambda index: commands[index][2] > command_threshold,
                minimum_command_frames,
            )

    stages = ["prepare"] * count
    for start, name in (
        (close_start, "grasp"),
        (lift_start, "lift"),
        (translate_start, "translate"),
        (place_start, "place"),
        (reset_start, "reset"),
    ):
        if start is not None and start < count:
            stages[start:] = [name] * (count - start)
    events = {
        "close_start": close_start,
        "lift_start": lift_start,
        "translate_start": translate_start,
        "place_start": place_start,
        "release_frame": release_frame,
        "reset_start": reset_start,
    }
    return stages, events


def summarize_segments(stages: list[str]) -> list[tuple[str, int, int, int]]:
    if not stages:
        return []
    result = []
    start = 0
    for index in range(1, len(stages) + 1):
        if index == len(stages) or stages[index] != stages[start]:
            result.append((stages[start], start, index - 1, index - start))
            start = index
    return result


def validate_stage_order(stages: list[str], path: Path) -> None:
    names = [item[0] for item in summarize_segments(stages)]
    if names != list(STAGES[: len(names)]):
        raise ValueError(f"{path}: invalid stage sequence {names}")


def mean_var(values: np.ndarray) -> tuple[list[float], list[float]]:
    if len(values) < 1:
        raise ValueError("empty statistics window")
    return values.mean(axis=0).tolist(), values.var(axis=0).tolist()


def add_statistics(records: list[dict], path: Path) -> None:
    stages = [record["stage"] for record in records]
    segments = summarize_segments(stages)
    flange = np.asarray(
        [finite_vector(record.get("force_torque"), 6, f"{path}:{i}.force_torque")
         for i, record in enumerate(records)],
        dtype=np.float64,
    )
    sources = sorted(records[0]["tactile_estimated_wrenches"])
    tactile = {
        source: np.asarray([
            finite_vector(
                record["tactile_estimated_wrenches"].get(source),
                6,
                f"{path}:{i}.tactile_estimated_wrenches.{source}",
            )
            for i, record in enumerate(records)
        ], dtype=np.float64)
        for source in sources
    }
    for stage, start, end, size in segments:
        force_mean, force_var = mean_var(flange[start : end + 1])
        tactile_stats = {
            source: mean_var(values[start : end + 1]) for source, values in tactile.items()
        }
        for record in records[start : end + 1]:
            record.update({
                "force_torque_stage_mean": force_mean,
                "force_torque_stage_var": force_var,
                "force_torque_stage_size": size,
                "force_torque_stage_name": stage,
                "force_torque_stage_start_index": start,
                "force_torque_stage_end_index": end,
                "force_torque_stage_layout": LAYOUT,
                "tactile_estimated_wrench_stage_mean": {
                    source: stats[0] for source, stats in tactile_stats.items()
                },
                "tactile_estimated_wrench_stage_var": {
                    source: stats[1] for source, stats in tactile_stats.items()
                },
                "tactile_estimated_wrench_stage_size": size,
                "tactile_estimated_wrench_stage_name": stage,
                "tactile_estimated_wrench_stage_start_index": start,
                "tactile_estimated_wrench_stage_end_index": end,
                "tactile_estimated_wrench_stage_layout": LAYOUT,
            })
    for index, record in enumerate(records):
        end = min(len(records), index + 50)
        force_mean, force_var = mean_var(flange[index:end])
        tactile_stats = {
            source: mean_var(values[index:end]) for source, values in tactile.items()
        }
        size = end - index
        record.update({
            "force_torque_chunk_mean": force_mean,
            "force_torque_chunk_var": [0.0] * 6 if size == 1 else force_var,
            "force_torque_chunk_size": size,
            "force_torque_chunk_target_size": 50,
            "force_torque_chunk_layout": LAYOUT,
            "tactile_estimated_wrench_chunk_mean": {
                source: stats[0] for source, stats in tactile_stats.items()
            },
            "tactile_estimated_wrench_chunk_var": {
                source: ([0.0] * 6 if size == 1 else stats[1])
                for source, stats in tactile_stats.items()
            },
            "tactile_estimated_wrench_chunk_size": size,
            "tactile_estimated_wrench_chunk_target_size": 50,
            "tactile_estimated_wrench_chunk_layout": LAYOUT,
        })


def finger_mean_wrench(record: dict, path: Path, index: int) -> list[float]:
    values = record.get("tactile_estimated_wrenches")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{path}:{index}: missing tactile estimated wrenches")
    array = np.asarray([
        finite_vector(value, 6, f"{path}:{index}.tactile.{source}")
        for source, value in sorted(values.items())
    ])
    return array.mean(axis=0).tolist()


def kmeans(
    points: np.ndarray, k: int = 4, seed: int = 0, restarts: int = 50
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) < k:
        raise ValueError(f"cannot fit K={k} to {len(points)} groups")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = points[rng.choice(len(points), size=k, replace=False)].copy()
        for _ in range(200):
            distances = ((points[:, None] - centers[None]) ** 2).sum(axis=-1)
            labels = distances.argmin(axis=1)
            updated = centers.copy()
            for cluster in range(k):
                members = points[labels == cluster]
                updated[cluster] = (
                    members.mean(axis=0) if len(members)
                    else points[distances.min(axis=1).argmax()]
                )
            if np.allclose(updated, centers):
                break
            centers = updated
        distances = ((points[:, None] - centers[None]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        inertia = float(distances.min(axis=1).sum())
        if best is None or inertia < best[2]:
            best = centers.copy(), labels.copy(), inertia
    assert best is not None
    return best


def build_prototypes(
    paths: list[Path], excluded: set[str], output: Path, tau: float
) -> dict:
    pooled: dict[tuple[str, str, str, str], list[list[float]]] = {}
    contributors: dict[tuple[str, str, str, str], set[str]] = {}
    object_ids: dict[tuple[str, str, str, str], set[str]] = {}
    used = []
    for path in paths:
        if path.parent.name in excluded:
            continue
        records = load_jsonl(path)
        object_category, load_condition, object_id = physical_condition(records[0], path)
        used.append(path.parent.name)
        for index, record in enumerate(records):
            stage = record.get("stage")
            if stage not in SUPERVISED_STAGES:
                continue
            key = (TASK, object_category, load_condition, stage)
            pooled.setdefault(key, []).append(finger_mean_wrench(record, path, index))
            contributors.setdefault(key, set()).add(path.parent.name)
            object_ids.setdefault(key, set()).add(object_id)
    keys = sorted(pooled, key=lambda key: (key[1], key[2], STAGES.index(key[3])))
    mu = np.asarray([np.mean(pooled[key], axis=0) for key in keys])
    sigma = np.asarray([np.std(pooled[key], axis=0) for key in keys])
    safe = np.concatenate([mu, sigma], axis=1)
    raw_desc = np.concatenate([mu, np.log(np.clip(sigma, 1e-4, None))], axis=1)
    medians = np.median(raw_desc, axis=0)
    iqrs = np.percentile(raw_desc, 75, axis=0) - np.percentile(raw_desc, 25, axis=0)
    descriptors = (raw_desc - medians) / (iqrs + 1e-6)
    centers, labels, inertia = kmeans(descriptors)
    dist_sq = ((descriptors[:, None] - centers[None]) ** 2).mean(axis=-1)
    logits = -dist_sq / tau
    logits -= logits.max(axis=1, keepdims=True)
    targets = np.exp(logits)
    targets /= targets.sum(axis=1, keepdims=True)
    groups = [{
        "group_id": row,
        "task": key[0],
        "fruit": key[1],
        "object_category": key[1],
        "load_condition": key[2],
        "stage": key[3],
        "frame_count": len(pooled[key]),
        "cluster": int(labels[row]),
        "contributing_episodes": sorted(contributors[key]),
        "contributing_object_ids": sorted(object_ids[key]),
    } for row, key in enumerate(keys)]
    metadata = {
        "signal": SIGNAL,
        "signal_layout": LAYOUT,
        "distribution_layout": DISTRIBUTION_LAYOUT,
        "stages": list(SUPERVISED_STAGES),
        "task": TASK,
        "group_definition": ["task", "object_category", "load_condition", "stage"],
        "n_group": len(keys),
        "k": 4,
        "tau_q": tau,
        "sigma_min": 1e-4,
        "epsilon": 1e-6,
        "seed": 0,
        "restarts": 50,
        "kmeans_inertia": inertia,
        "sigma_estimator": "population_std_ddof0",
        "episodes_used": sorted(used),
        "episodes_excluded": sorted(excluded),
        "groups": groups,
    }
    tables = {
        "safe_distribution_all": safe.tolist(),
        "descriptors": descriptors.tolist(),
        "prototype_centers": centers.tolist(),
        "soft_prototype_targets_all": targets.tolist(),
        "normalizer_medians": medians.tolist(),
        "normalizer_iqrs": iqrs.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "group_metadata.json", metadata)
    atomic_write_json(output / "prototype_tables.json", tables)
    np.save(output / "safe_distribution_all.npy", safe)
    np.save(output / "descriptors.npy", descriptors)
    np.save(output / "prototype_centers.npy", centers)
    np.save(output / "soft_prototype_targets_all.npy", targets)
    np.savez(
        output / "normalizer_stats.npz",
        medians=medians,
        iqrs=iqrs,
        epsilon=1e-6,
        sigma_min=1e-4,
    )
    return {"metadata": metadata, "tables": tables}


def add_prototype_labels(records: list[dict], path: Path, bundle: dict) -> tuple[int, int]:
    metadata, tables = bundle["metadata"], bundle["tables"]
    group_index = {
        (
            group["task"], group["object_category"],
            group["load_condition"], group["stage"],
        ): group["group_id"]
        for group in metadata["groups"]
    }
    object_category, load_condition, object_id = physical_condition(records[0], path)
    valid = 0
    for record in records:
        stage = record["stage"]
        record["gt_safe_distribution_layout"] = DISTRIBUTION_LAYOUT
        record["gt_safe_distribution_signal"] = SIGNAL
        if stage in SUPERVISED_STAGES:
            group_id = group_index[(TASK, object_category, load_condition, stage)]
            record.update({
                "group_id": group_id,
                "group_task": TASK,
                "group_fruit": object_category,
                "group_object_category": object_category,
                "group_load_condition": load_condition,
                "group_object_id": object_id,
                "group_stage": stage,
                "gt_safe_distribution": tables["safe_distribution_all"][group_id],
                "soft_prototype_target": tables["soft_prototype_targets_all"][group_id],
                "prototype_supervision_valid": True,
            })
            valid += 1
        else:
            record.update({
                "group_id": -1,
                "group_task": None,
                "group_fruit": None,
                "group_object_category": None,
                "group_load_condition": None,
                "group_object_id": None,
                "group_stage": None,
                "gt_safe_distribution": None,
                "soft_prototype_target": None,
                "prototype_supervision_valid": False,
            })
    return valid, len(records) - valid


def add_zeroed_force(
    records: list[dict], path: Path
) -> tuple[list[float], int, float, str]:
    initial_target = float(records[0]["gripper_action_target"])
    initial_width = float(records[0]["gripper_width"])
    commands = [
        finite_vector(record.get("cmd_speed_l"), 6, f"{path}:{index}.cmd_speed_l")
        for index, record in enumerate(records)
    ]
    size = 0
    command_threshold = 0.0001
    for candidate in (0.0001, 0.0005, 0.001, 0.002, 0.005):
        candidate_size = 0
        for record, command in zip(records, commands):
            if not (
                max(abs(value) for value in command) <= candidate
                and abs(float(record["gripper_action_target"]) - initial_target) <= 0.001
                and abs(float(record["gripper_width"]) - initial_width) <= 0.001
            ):
                break
            candidate_size += 1
        if candidate_size >= 2:
            size = candidate_size
            command_threshold = candidate
            break
    if size < 2:
        raise ValueError(f"{path}: initial static/open window has {size} frame(s)")
    raw = np.asarray([record["force_torque"] for record in records], dtype=np.float64)
    offset = [statistics.median(raw[:size, dim].tolist()) for dim in range(6)]
    zeroed = raw - np.asarray(offset)
    method = (
        ZEROING_METHOD
        if command_threshold == 0.0001
        else ZEROING_FALLBACK_METHOD
    )
    common = {
        "force_torque_zero_offset": offset,
        "force_torque_zero_offset_layout": LAYOUT,
        "force_torque_zeroing_method": method,
        "force_torque_zeroing_start_index": 0,
        "force_torque_zeroing_end_index": size - 1,
        "force_torque_zeroing_size": size,
        "force_torque_zeroing_command_threshold": command_threshold,
        "force_torque_zeroing_gripper_tolerance_m": 0.001,
    }
    for record, value in zip(records, zeroed):
        record["force_torque_zeroed"] = value.tolist()
        record["force_torque_zeroed_layout"] = LAYOUT
        record.update(common)
    for stage, start, end, segment_size in summarize_segments(
        [record["stage"] for record in records]
    ):
        mean, var = mean_var(zeroed[start : end + 1])
        for record in records[start : end + 1]:
            record.update({
                "force_torque_zeroed_stage_mean": mean,
                "force_torque_zeroed_stage_var": var,
                "force_torque_zeroed_stage_size": segment_size,
                "force_torque_zeroed_stage_name": stage,
                "force_torque_zeroed_stage_start_index": start,
                "force_torque_zeroed_stage_end_index": end,
                "force_torque_zeroed_stage_layout": LAYOUT,
            })
    for index, record in enumerate(records):
        chunk = zeroed[index : index + 50]
        mean, var = mean_var(chunk)
        record.update({
            "force_torque_zeroed_chunk_mean": mean,
            "force_torque_zeroed_chunk_var": [0.0] * 6 if len(chunk) == 1 else var,
            "force_torque_zeroed_chunk_size": len(chunk),
            "force_torque_zeroed_chunk_target_size": 50,
            "force_torque_zeroed_chunk_layout": LAYOUT,
        })
    return offset, size, command_threshold, method


def zeroing_metadata(
    offset: list[float], size: int, command_threshold: float, method: str
) -> dict:
    return {
        "raw_field": "force_torque",
        "recommended_training_field": "force_torque_zeroed",
        "layout": LAYOUT,
        "method": method,
        "zero_offset": offset,
        "calibration_start_index": 0,
        "calibration_end_index": size - 1,
        "calibration_size": size,
        "stationary_command_threshold": command_threshold,
        "open_gripper_tolerance_m": 0.001,
        "stage_statistics_prefix": "force_torque_zeroed_stage_",
        "chunk_statistics_prefix": "force_torque_zeroed_chunk_",
        "chunk_target_size": 50,
        "raw_values_preserved": True,
        "note": (
            "Removes the per-episode initial open, static or low-motion sensor offset. "
            "The smallest configured command threshold producing at least two leading "
            "frames is used. It does not compensate tool gravity under changing "
            "orientation or dynamics."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    paths = sorted(args.root.glob("pi0_train_*/observations.jsonl"))
    if not paths:
        raise SystemExit(f"no episodes under {args.root}")
    excluded = set(args.exclude)
    unknown = excluded - {path.parent.name for path in paths}
    if unknown:
        raise SystemExit(f"unknown excluded episodes: {sorted(unknown)}")

    stage_counts = Counter()
    cached = {}
    for path in paths:
        records = load_jsonl(path)
        stages, events = compute_ordered_stages(records, path)
        validate_stage_order(stages, path)
        names = [item[0] for item in summarize_segments(stages)]
        if names != list(STAGES) or events["release_frame"] is None:
            raise ValueError(f"{path}: incomplete stages/release: {names}, {events}")
        for record, stage in zip(records, stages):
            record["stage"] = stage
        stage_counts.update(stages)
        if args.write:
            backup_once(path, ".before_stage_20260916_carton")
            atomic_write_jsonl(path, records)
        cached[path] = records

    if not args.write:
        print("DRY-RUN stage:", dict(stage_counts))
        return 0

    for path in paths:
        records = cached[path]
        backup_once(path, ".before_ft_labels")
        add_statistics(records, path)
        atomic_write_jsonl(path, records)

    bundles = {}
    for name, tau in (("prototype_metadata", 0.1), ("prototype_metadata_tau05", 0.5),
                      ("prototype_metadata_tau10", 1.0)):
        bundles[tau] = build_prototypes(paths, excluded, args.root / name, tau)
    primary = bundles[0.1]
    valid = masked = 0
    for path in paths:
        records = cached[path]
        backup_once(path, ".before_proto_labels")
        v, m = add_prototype_labels(records, path, primary)
        valid += v
        masked += m
        atomic_write_jsonl(path, records)

    calibration_sizes = []
    adaptive_thresholds = {}
    zeroed = []
    for path in paths:
        records = cached[path]
        metadata_path = path.parent / "metadata.json"
        metadata = load_json(metadata_path)
        backup_once(path, ".before_force_zeroing")
        backup_once(metadata_path, ".before_force_zeroing")
        offset, size, command_threshold, method = add_zeroed_force(records, path)
        calibration_sizes.append(size)
        if command_threshold != 0.0001:
            adaptive_thresholds[path.parent.name] = command_threshold
        zeroed.extend(record["force_torque_zeroed"] for record in records)
        postprocessing = metadata.setdefault("postprocessing", {})
        if not isinstance(postprocessing, dict):
            raise ValueError(f"{metadata_path}: postprocessing is not an object")
        postprocessing["force_torque_zeroing"] = zeroing_metadata(
            offset, size, command_threshold, method
        )
        atomic_write_jsonl(path, records)
        atomic_write_json(metadata_path, metadata)

    zeroed_array = np.asarray(zeroed)
    print(
        f"WROTE episodes={len(paths)} frames={sum(stage_counts.values())} "
        f"stages={dict(stage_counts)} valid={valid} masked={masked} "
        f"groups={primary['metadata']['n_group']} excluded={sorted(excluded)}"
    )
    print(
        "zeroing calibration min/median/max="
        f"{min(calibration_sizes)}/{statistics.median(calibration_sizes)}/{max(calibration_sizes)} "
        f"adaptive={adaptive_thresholds} global_mean={zeroed_array.mean(axis=0).tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
