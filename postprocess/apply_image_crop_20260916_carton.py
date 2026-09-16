#!/usr/bin/env python3
"""Apply an approved square crop, preserve originals, and update image metadata.

Dry-run is the default.  Nothing is written unless --write is supplied.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from postprocess_20260916_carton import atomic_write_jsonl, backup_once, load_jsonl


SIZE = 224
SOURCE_SIZE = (1920, 1080)
DEFAULT_RGB = [500, 0, 1580, 1080]
DEFAULT_WRIST = [420, 0, 1500, 1080]
LAYOUT_NOTE = (
    "images pre-cropped (see image_crop_box fields) and resized to 224x224 "
    "before saving; replay/conversion must not re-crop and should not trust "
    "the original camera size"
)


def valid_box(name: str, box: list[int]) -> list[int]:
    left, top, right, bottom = box
    if not (0 <= left < right <= 1920 and 0 <= top < bottom <= 1080):
        raise ValueError(f"{name}: crop lies outside source: {box}")
    if right - left != bottom - top:
        raise ValueError(f"{name}: crop is not square: {box}")
    return box


def state(episode: Path) -> str:
    plain = all((episode / camera).is_dir() for camera in ("rgb", "wrist"))
    small = all((episode / f"{camera}_{SIZE}").is_dir() for camera in ("rgb", "wrist"))
    full = all((episode / f"{camera}_full").is_dir() for camera in ("rgb", "wrist"))
    if plain and not small and not full:
        return "pending"
    if plain and small and not full:
        return "generated"
    if plain and full and not small:
        return "switched"
    raise ValueError(f"{episode}: inconsistent image directory layout")


def crop_one(job: tuple[str, str, tuple[int, int, int, int]]) -> str | None:
    source, destination, box = job
    try:
        with Image.open(source) as image:
            if image.size != SOURCE_SIZE or image.mode != "RGB":
                return f"{source}: size={image.size} mode={image.mode}"
            image.crop(box).resize((SIZE, SIZE), Image.Resampling.LANCZOS).save(
                destination, quality=95
            )
        return None
    except Exception as exc:
        return f"{source}: {exc}"


def generate(episodes: list[Path], boxes: dict[str, list[int]], workers: int) -> None:
    jobs = []
    for episode in episodes:
        if state(episode) == "switched":
            continue
        for camera, box in boxes.items():
            source = episode / camera
            destination = episode / f"{camera}_{SIZE}"
            destination.mkdir(exist_ok=True)
            for path in sorted(source.glob("*.jpg")):
                jobs.append((str(path), str(destination / path.name), tuple(box)))
    with Pool(workers) as pool:
        errors = [error for error in pool.imap_unordered(crop_one, jobs, chunksize=64) if error]
    if errors:
        raise RuntimeError("crop failures:\n" + "\n".join(errors[:20]))
    print(f"generated={len(jobs)} cropped images")


def validate_generated(episode: Path, records: list[dict]) -> None:
    for camera, field in (("rgb", "image_path"), ("wrist", "wrist_image_path")):
        directory = episode / f"{camera}_{SIZE}"
        paths = sorted(directory.glob("*.jpg"))
        expected = [Path(record[field]).name for record in records]
        if [path.name for path in paths] != expected:
            raise ValueError(f"{directory}: names do not match JSONL")
        for path in paths:
            with Image.open(path) as image:
                if image.size != (SIZE, SIZE) or image.mode != "RGB":
                    raise ValueError(f"{path}: invalid cropped image")
                image.verify()


def update_records(records: list[dict], boxes: dict[str, list[int]]) -> None:
    for record in records:
        record["image_shape"] = [SIZE, SIZE, 3]
        record["image_source_shape"] = [1080, 1920, 3]
        record["image_crop_box"] = boxes["rgb"]
        record["wrist_image_shape"] = [SIZE, SIZE, 3]
        record["wrist_image_source_shape"] = [1080, 1920, 3]
        record["wrist_image_crop_box"] = boxes["wrist"]
        sim = record.get("pi0_aloha_sim_observation")
        if isinstance(sim, dict):
            sim["image_layout"] = LAYOUT_NOTE


def switch(episode: Path, boxes: dict[str, list[int]]) -> None:
    current_state = state(episode)
    path = episode / "observations.jsonl"
    records = load_jsonl(path)
    if current_state == "generated":
        validate_generated(episode, records)
        for camera in ("rgb", "wrist"):
            (episode / camera).rename(episode / f"{camera}_full")
            (episode / f"{camera}_{SIZE}").rename(episode / camera)
    elif current_state == "switched":
        done = all(
            record.get("image_shape") == [SIZE, SIZE, 3]
            and record.get("wrist_image_shape") == [SIZE, SIZE, 3]
            for record in records
        )
        if done:
            print(f"{episode.name}: already switched")
            return
    else:
        raise ValueError(f"{episode}: crops were not generated")
    backup_once(path, ".before_224_switch")
    update_records(records, boxes)
    atomic_write_jsonl(path, records)
    print(f"{episode.name}: originals -> *_full, 224x224 -> rgb/wrist")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--rgb-box", type=int, nargs=4, default=DEFAULT_RGB)
    parser.add_argument("--wrist-box", type=int, nargs=4, default=DEFAULT_WRIST)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--approved",
        action="store_true",
        help="required with --write after manual crop preview approval",
    )
    args = parser.parse_args()
    boxes = {
        "rgb": valid_box("rgb", args.rgb_box),
        "wrist": valid_box("wrist", args.wrist_box),
    }
    episodes = sorted(path for path in args.root.glob("pi0_train_*") if path.is_dir())
    counts = {}
    for episode in episodes:
        counts[state(episode)] = counts.get(state(episode), 0) + 1
    print(f"episodes={len(episodes)} states={counts} boxes={boxes}")
    if not args.write:
        print("DRY-RUN: no images or JSONL changed")
        return 0
    if not args.approved:
        raise SystemExit("refusing to crop: manual review must be recorded with --approved")
    generate(episodes, boxes, args.workers)
    for episode in episodes:
        switch(episode, boxes)
    print(f"WROTE episodes={len(episodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
