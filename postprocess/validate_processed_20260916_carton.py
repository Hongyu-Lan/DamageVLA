#!/usr/bin/env python3
"""Independently recompute and validate the 0916 carton post-processing."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from apply_image_crop_20260916_carton import DEFAULT_RGB, DEFAULT_WRIST, LAYOUT_NOTE, SIZE

from postprocess_20260916_carton import (
    DISTRIBUTION_LAYOUT,
    LAYOUT,
    SIGNAL,
    STAGES,
    SUPERVISED_STAGES,
    TASK,
    ZEROING_FALLBACK_METHOD,
    ZEROING_METHOD,
    compute_ordered_stages,
    fruit_from_prompt,
    load_jsonl,
    physical_condition,
    summarize_segments,
)


def close(actual: object, expected: object, label: str) -> None:
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{label}: invalid shape or non-finite")
    if not np.allclose(a, e, rtol=1e-9, atol=5e-10):
        raise ValueError(f"{label}: max difference {float(np.max(np.abs(a-e)))}")


def check_stat_metadata(
    record: dict, prefix: str, size: int, start: int | None = None,
    end: int | None = None, stage: str | None = None,
) -> None:
    if record[prefix + "size"] != size or record[prefix + "layout"] != LAYOUT:
        raise ValueError(f"invalid {prefix} size/layout")
    if start is not None and (
        record[prefix + "start_index"] != start
        or record[prefix + "end_index"] != end
        or record[prefix + "name"] != stage
    ):
        raise ValueError(f"invalid {prefix} segment metadata")


def check_episode_stats(records: list[dict], path: Path) -> tuple[int, list[list[float]]]:
    flange = np.asarray([record["force_torque"] for record in records], dtype=np.float64)
    tactile = {
        source: np.asarray(
            [record["tactile_estimated_wrenches"][source] for record in records],
            dtype=np.float64,
        )
        for source in ("left_estimated", "right_estimated")
    }
    segments = summarize_segments([record["stage"] for record in records])
    for stage, start, end, size in segments:
        force_values = flange[start : end + 1]
        tactile_values = {source: values[start : end + 1] for source, values in tactile.items()}
        for index in range(start, end + 1):
            record = records[index]
            close(record["force_torque_stage_mean"], force_values.mean(0), f"{path}:{index} force stage mean")
            close(record["force_torque_stage_var"], force_values.var(0), f"{path}:{index} force stage var")
            check_stat_metadata(record, "force_torque_stage_", size, start, end, stage)
            for source, values in tactile_values.items():
                close(record["tactile_estimated_wrench_stage_mean"][source], values.mean(0), f"{path}:{index} tactile stage mean")
                close(record["tactile_estimated_wrench_stage_var"][source], values.var(0), f"{path}:{index} tactile stage var")
            check_stat_metadata(record, "tactile_estimated_wrench_stage_", size, start, end, stage)
    for index, record in enumerate(records):
        force_values = flange[index : index + 50]
        size = len(force_values)
        close(record["force_torque_chunk_mean"], force_values.mean(0), f"{path}:{index} force chunk mean")
        close(record["force_torque_chunk_var"], force_values.var(0), f"{path}:{index} force chunk var")
        check_stat_metadata(record, "force_torque_chunk_", size)
        if record["force_torque_chunk_target_size"] != 50:
            raise ValueError(f"{path}:{index}: force chunk target")
        for source, values in tactile.items():
            chunk = values[index : index + 50]
            close(record["tactile_estimated_wrench_chunk_mean"][source], chunk.mean(0), f"{path}:{index} tactile chunk mean")
            close(record["tactile_estimated_wrench_chunk_var"][source], chunk.var(0), f"{path}:{index} tactile chunk var")
        check_stat_metadata(record, "tactile_estimated_wrench_chunk_", size)
        if record["tactile_estimated_wrench_chunk_target_size"] != 50:
            raise ValueError(f"{path}:{index}: tactile chunk target")

    thresholds = (0.0001, 0.0005, 0.001, 0.002, 0.005)
    initial_target = float(records[0]["gripper_action_target"])
    initial_width = float(records[0]["gripper_width"])
    chosen = None
    calibration_size = 0
    for threshold in thresholds:
        size = 0
        for record in records:
            if not (
                max(abs(float(value)) for value in record["cmd_speed_l"]) <= threshold
                and abs(float(record["gripper_action_target"]) - initial_target) <= 0.001
                and abs(float(record["gripper_width"]) - initial_width) <= 0.001
            ):
                break
            size += 1
        if size >= 2:
            chosen, calibration_size = threshold, size
            break
    if chosen is None:
        raise ValueError(f"{path}: no valid zeroing threshold")
    offset = np.asarray([
        statistics.median(flange[:calibration_size, dim].tolist()) for dim in range(6)
    ])
    zeroed = flange - offset
    method = ZEROING_METHOD if chosen == 0.0001 else ZEROING_FALLBACK_METHOD
    for index, record in enumerate(records):
        close(record["force_torque_zero_offset"], offset, f"{path}:{index} offset")
        close(record["force_torque_zeroed"], zeroed[index], f"{path}:{index} zeroed")
        if (
            record["force_torque_zeroing_method"] != method
            or record["force_torque_zeroing_start_index"] != 0
            or record["force_torque_zeroing_end_index"] != calibration_size - 1
            or record["force_torque_zeroing_size"] != calibration_size
            or record["force_torque_zeroing_command_threshold"] != chosen
            or record["force_torque_zeroing_gripper_tolerance_m"] != 0.001
        ):
            raise ValueError(f"{path}:{index}: zeroing metadata")
    for stage, start, end, size in segments:
        values = zeroed[start : end + 1]
        for index in range(start, end + 1):
            record = records[index]
            close(record["force_torque_zeroed_stage_mean"], values.mean(0), f"{path}:{index} zero stage mean")
            close(record["force_torque_zeroed_stage_var"], values.var(0), f"{path}:{index} zero stage var")
            check_stat_metadata(record, "force_torque_zeroed_stage_", size, start, end, stage)
    for index, record in enumerate(records):
        values = zeroed[index : index + 50]
        close(record["force_torque_zeroed_chunk_mean"], values.mean(0), f"{path}:{index} zero chunk mean")
        close(record["force_torque_zeroed_chunk_var"], values.var(0), f"{path}:{index} zero chunk var")
        check_stat_metadata(record, "force_torque_zeroed_chunk_", len(values))
        if record["force_torque_zeroed_chunk_target_size"] != 50:
            raise ValueError(f"{path}:{index}: zero chunk target")
    return calibration_size, zeroed.tolist()


def load_bundle(directory: Path) -> tuple[dict, dict]:
    metadata = json.loads((directory / "group_metadata.json").read_text(encoding="utf-8"))
    tables = json.loads((directory / "prototype_tables.json").read_text(encoding="utf-8"))
    for filename, key in (
        ("safe_distribution_all.npy", "safe_distribution_all"),
        ("descriptors.npy", "descriptors"),
        ("prototype_centers.npy", "prototype_centers"),
        ("soft_prototype_targets_all.npy", "soft_prototype_targets_all"),
    ):
        close(np.load(directory / filename), tables[key], f"{directory}/{filename}")
    safe = np.asarray(tables["safe_distribution_all"])
    descriptors = np.asarray(tables["descriptors"])
    centers = np.asarray(tables["prototype_centers"])
    targets = np.asarray(tables["soft_prototype_targets_all"])
    raw = np.concatenate([safe[:, :6], np.log(np.clip(safe[:, 6:], 1e-4, None))], axis=1)
    medians = np.median(raw, axis=0)
    iqrs = np.percentile(raw, 75, axis=0) - np.percentile(raw, 25, axis=0)
    close(descriptors, (raw - medians) / (iqrs + 1e-6), f"{directory}: descriptors")
    dist_sq = ((descriptors[:, None] - centers[None]) ** 2).mean(-1)
    logits = -dist_sq / float(metadata["tau_q"])
    logits -= logits.max(1, keepdims=True)
    expected = np.exp(logits)
    expected /= expected.sum(1, keepdims=True)
    close(targets, expected, f"{directory}: targets")
    labels = dist_sq.argmin(1)
    if labels.tolist() != [group["cluster"] for group in metadata["groups"]]:
        raise ValueError(f"{directory}: nearest-center labels mismatch")
    for cluster in range(4):
        members = descriptors[labels == cluster]
        if len(members):
            close(centers[cluster], members.mean(0), f"{directory}: center {cluster}")
    return metadata, tables


def check_images(episode: Path, records: list[dict]) -> tuple[int, int]:
    cropped_checked = full_checked = 0
    for camera, field in (("rgb", "image_path"), ("wrist", "wrist_image_path")):
        paths = sorted((episode / camera).glob("*.jpg"))
        expected = [Path(record[field]).name for record in records]
        if [path.name for path in paths] != expected:
            raise ValueError(f"{episode}/{camera}: image name mismatch")
        full_paths = sorted((episode / f"{camera}_full").glob("*.jpg"))
        if [path.name for path in full_paths] != expected:
            raise ValueError(f"{episode}/{camera}_full: image name mismatch")
        if (episode / f"{camera}_{SIZE}").exists():
            raise ValueError(f"{episode}/{camera}_{SIZE}: temporary directory remains")
        for path in paths:
            with Image.open(path) as image:
                if image.size != (SIZE, SIZE) or image.mode != "RGB":
                    raise ValueError(f"{path}: invalid cropped size/mode")
                image.verify()
            cropped_checked += 1
        for path in full_paths:
            with Image.open(path) as image:
                if image.size != (1920, 1080) or image.mode != "RGB":
                    raise ValueError(f"{path}: invalid full-resolution size/mode")
                image.verify()
            full_checked += 1
    return cropped_checked, full_checked


def expected_after_image_crop(record: dict) -> dict:
    expected = copy.deepcopy(record)
    expected["image_shape"] = [SIZE, SIZE, 3]
    expected["image_source_shape"] = [1080, 1920, 3]
    expected["image_crop_box"] = DEFAULT_RGB
    expected["wrist_image_shape"] = [SIZE, SIZE, 3]
    expected["wrist_image_source_shape"] = [1080, 1920, 3]
    expected["wrist_image_crop_box"] = DEFAULT_WRIST
    sim = expected.get("pi0_aloha_sim_observation")
    if isinstance(sim, dict):
        sim["image_layout"] = LAYOUT_NOTE
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("pi0_train_*/observations.jsonl"))
    errors = []
    stage_counts, fruits, conditions = Counter(), Counter(), Counter()
    valid = masked = 0
    calibration_sizes, all_zeroed = [], []
    fallback_thresholds = {}
    bundles = {}
    cropped_images_checked = full_images_checked = 0
    try:
        for name in ("prototype_metadata", "prototype_metadata_tau05", "prototype_metadata_tau10"):
            bundles[name] = load_bundle(args.root / name)
        primary_meta, primary_tables = bundles["prototype_metadata"]
        if primary_meta["episodes_excluded"]:
            raise ValueError("prototype metadata unexpectedly excludes successful episodes")
        if primary_meta["n_group"] != 8 or primary_meta["k"] != 4:
            raise ValueError("unexpected group/prototype count")
        for name in ("prototype_metadata_tau05", "prototype_metadata_tau10"):
            metadata, tables = bundles[name]
            close(tables["safe_distribution_all"], primary_tables["safe_distribution_all"], name + " safe")
            close(tables["descriptors"], primary_tables["descriptors"], name + " descriptors")
        group_index = {
            (
                g["task"], g["object_category"],
                g["load_condition"], g["stage"],
            ): g["group_id"]
            for g in primary_meta["groups"]
        }
    except Exception as exc:
        errors.append(str(exc))
        primary_meta = primary_tables = group_index = None

    for path in paths:
        try:
            records = load_jsonl(path)
            raw = load_jsonl(path.with_name(path.name + ".before_stage_20260916_carton"))
            if len(records) != len(raw):
                raise ValueError(f"{path}: record count changed")
            for index, (before, record) in enumerate(zip(raw, records)):
                for key, value in before.items():
                    if key in ("image_shape", "wrist_image_shape"):
                        continue
                    if key == "pi0_aloha_sim_observation":
                        before_sim = copy.deepcopy(value)
                        after_sim = copy.deepcopy(record.get(key))
                        if isinstance(before_sim, dict):
                            before_sim.pop("image_layout", None)
                        if isinstance(after_sim, dict):
                            after_sim.pop("image_layout", None)
                        if before_sim != after_sim:
                            raise ValueError(f"{path}:{index}: original simulation field changed")
                    elif record.get(key) != value:
                        raise ValueError(f"{path}:{index}: original field {key} changed")
            for suffix in (
                ".before_ft_labels",
                ".before_proto_labels",
                ".before_force_zeroing",
            ):
                backup = load_jsonl(path.with_name(path.name + suffix))
                if len(backup) != len(records):
                    raise ValueError(f"{path}{suffix}: record count")
            if args.check_images:
                before_crop_path = path.with_name(path.name + ".before_224_switch")
                before_crop = load_jsonl(before_crop_path)
                if len(before_crop) != len(records):
                    raise ValueError(f"{before_crop_path}: record count")
                for index, (before, record) in enumerate(zip(before_crop, records)):
                    if expected_after_image_crop(before) != record:
                        raise ValueError(f"{path}:{index}: unexpected change during image crop")
            stages, events = compute_ordered_stages(records, path)
            if stages != [record["stage"] for record in records] or events["release_frame"] is None:
                raise ValueError(f"{path}: stage mismatch")
            stage_counts.update(stages)
            fruit = fruit_from_prompt(records[0]["prompt"], path)
            fruits[fruit] += 1
            object_category, load_condition, object_id = physical_condition(records[0], path)
            conditions[f"{object_category}/{load_condition}"] += 1
            size, zeroed = check_episode_stats(records, path)
            calibration_sizes.append(size)
            all_zeroed.extend(zeroed)
            threshold = records[0]["force_torque_zeroing_command_threshold"]
            if threshold != 0.0001:
                fallback_thresholds[path.parent.name] = threshold
            metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
            zero_meta = metadata["postprocessing"]["force_torque_zeroing"]
            if (
                zero_meta["recommended_training_field"] != "force_torque_zeroed"
                or zero_meta["calibration_size"] != size
                or zero_meta["stationary_command_threshold"] != threshold
                or not zero_meta["raw_values_preserved"]
            ):
                raise ValueError(f"{path.parent}/metadata.json: zeroing metadata")
            if not (path.parent / "metadata.json.before_force_zeroing").is_file():
                raise ValueError(f"{path.parent}: missing metadata backup")
            if group_index is not None:
                for index, record in enumerate(records):
                    if record["stage"] in SUPERVISED_STAGES:
                        group_id = group_index[
                            (TASK, object_category, load_condition, record["stage"])
                        ]
                        if (
                            record["group_id"] != group_id
                            or record["group_task"] != TASK
                            or record["group_fruit"] != object_category
                            or record["group_object_category"] != object_category
                            or record["group_load_condition"] != load_condition
                            or record["group_object_id"] != object_id
                            or record["group_stage"] != record["stage"]
                            or record["gt_safe_distribution_layout"] != DISTRIBUTION_LAYOUT
                            or record["gt_safe_distribution_signal"] != SIGNAL
                            or not record["prototype_supervision_valid"]
                        ):
                            raise ValueError(f"{path}:{index}: prototype metadata")
                        close(record["gt_safe_distribution"], primary_tables["safe_distribution_all"][group_id], f"{path}:{index} dist")
                        close(record["soft_prototype_target"], primary_tables["soft_prototype_targets_all"][group_id], f"{path}:{index} target")
                        valid += 1
                    else:
                        if not (
                            record["group_id"] == -1
                            and record["gt_safe_distribution"] is None
                            and record["soft_prototype_target"] is None
                            and not record["prototype_supervision_valid"]
                        ):
                            raise ValueError(f"{path}:{index}: unmasked stage")
                        masked += 1
            if args.check_images:
                cropped_count, full_count = check_images(path.parent, records)
                cropped_images_checked += cropped_count
                full_images_checked += full_count
        except Exception as exc:
            errors.append(str(exc))

    zeroed_array = np.asarray(all_zeroed)
    report = {
        "episodes": len(paths),
        "frames": int(sum(stage_counts.values())),
        "fruit_episode_counts": dict(fruits),
        "physical_condition_episode_counts": dict(conditions),
        "stage_counts": {stage: stage_counts[stage] for stage in STAGES},
        "prototype_valid_frames": valid,
        "prototype_masked_frames": masked,
        "prototype_groups": primary_meta["n_group"] if primary_meta else None,
        "prototype_k": primary_meta["k"] if primary_meta else None,
        "excluded_from_group_statistics": primary_meta["episodes_excluded"] if primary_meta else None,
        "jsonl_backup_files": sum(1 for _ in args.root.glob("pi0_train_*/observations.jsonl.before_*")),
        "metadata_backup_files": sum(1 for _ in args.root.glob("pi0_train_*/metadata.json.before_force_zeroing")),
        "force_torque_zeroing": {
            "field": "force_torque_zeroed",
            "raw_field_preserved": True,
            "calibration_size_min": min(calibration_sizes) if calibration_sizes else None,
            "calibration_size_median": statistics.median(calibration_sizes) if calibration_sizes else None,
            "calibration_size_max": max(calibration_sizes) if calibration_sizes else None,
            "adaptive_threshold_episodes": fallback_thresholds,
            "global_zeroed_mean": zeroed_array.mean(0).tolist() if len(zeroed_array) else None,
            "global_zeroed_std": zeroed_array.std(0).tolist() if len(zeroed_array) else None,
        },
        "image_processing": {
            "applied": args.check_images,
            "pending_manual_crop_approval": not args.check_images,
            "rgb_crop_box": DEFAULT_RGB,
            "wrist_crop_box": DEFAULT_WRIST,
            "output_shape": [SIZE, SIZE, 3],
            "originals_preserved_in": ["rgb_full", "wrist_full"],
            "cropped_image_files_checked": cropped_images_checked if args.check_images else None,
            "full_resolution_backup_image_files_checked": full_images_checked if args.check_images else None,
            "replay_or_conversion_must_not_recrop": True,
        },
        "errors": errors,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
