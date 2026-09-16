#!/usr/bin/env python3
"""Create read-only crop and outcome review sheets for the 2026-09-11 batch.

All crops are made in memory for inspection.  This script never changes source
images or observations.jsonl.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from postprocess_20260911 import (
    compute_ordered_stages,
    fruit_from_prompt,
    load_jsonl,
    summarize_segments,
)


SOURCE_SIZE = (1920, 1080)
OUTPUT_SIZE = 224
CANDIDATES = {
    "rgb": {
        "0905_accepted": (500, 0, 1580, 1080),
        "left": (420, 0, 1500, 1080),
        "right": (580, 0, 1660, 1080),
    },
    "wrist": {
        "0905_accepted": (420, 0, 1500, 1080),
        "left": (300, 0, 1380, 1080),
        "right": (540, 0, 1620, 1080),
    },
}


def label(image: Image.Image, text: str, height: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "black")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((5, 6), text, fill="white")
    return canvas


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", size, "black")
    cell.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return cell


def sheet(images: list[Image.Image], columns: int, size: tuple[int, int]) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    result = Image.new("RGB", (columns * size[0], rows * size[1]), "black")
    for index, image in enumerate(images):
        result.paste(fit(image, size), ((index % columns) * size[0], (index // columns) * size[1]))
    return result


def source_image(episode: Path, record: dict, camera: str) -> Image.Image:
    field = "image_path" if camera == "rgb" else "wrist_image_path"
    path = episode / camera / Path(record[field]).name
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if image.size != SOURCE_SIZE:
        raise ValueError(f"{path}: {image.size}, expected {SOURCE_SIZE}")
    return image


def episode_info(episode: Path) -> tuple[list[dict], str, list[tuple[str, int, int, int]]]:
    path = episode / "observations.jsonl"
    records = load_jsonl(path)
    fruit = fruit_from_prompt(str(records[0]["prompt"]), path)
    stages, _ = compute_ordered_stages(records, path)
    return records, fruit, summarize_segments(stages)


def representative_episodes(episodes: list[Path]) -> list[Path]:
    grouped: dict[str, list[Path]] = {}
    for episode in episodes:
        records = load_jsonl(episode / "observations.jsonl")
        fruit = fruit_from_prompt(str(records[0]["prompt"]), episode)
        grouped.setdefault(fruit, []).append(episode)
    selected = []
    for fruit in sorted(grouped):
        values = grouped[fruit]
        selected.extend(values[index] for index in sorted({0, len(values) // 2, len(values) - 1}))
    return selected


def crop_sheets(episodes: list[Path], output: Path) -> None:
    selected = representative_episodes(episodes)
    for camera, candidates in CANDIDATES.items():
        unique_candidates = dict(candidates)
        for name, box in unique_candidates.items():
            overlays, crops = [], []
            for episode in selected:
                records, fruit, segments = episode_info(episode)
                for stage, start, end, _ in segments:
                    index = (start + end) // 2
                    image = source_image(episode, records[index], camera)
                    overlay = image.copy()
                    ImageDraw.Draw(overlay).rectangle(box, outline="red", width=10)
                    tag = f"{fruit} {episode.name[-6:]} {stage} f={index}"
                    overlays.append(label(overlay, f"{tag} box={box}"))
                    cropped = image.crop(box).resize(
                        (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
                    )
                    crops.append(label(cropped, tag))
            sheet(overlays, 6, (480, 294)).save(
                output / f"{camera}_{name}_overlays.jpg", quality=94
            )
            sheet(crops, 6, (240, 264)).save(
                output / f"{camera}_{name}_224.jpg", quality=94
            )


def rgb_comparison(episodes: list[Path], output: Path) -> None:
    images = []
    candidates = CANDIDATES["rgb"]
    for episode in representative_episodes(episodes):
        records, fruit, segments = episode_info(episode)
        segment_index = {stage: (start + end) // 2 for stage, start, end, _ in segments}
        for stage in ("prepare", "grasp", "place"):
            index = segment_index[stage]
            original = source_image(episode, records[index], "rgb")
            for name, box in candidates.items():
                crop = original.crop(box).resize(
                    (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
                )
                images.append(label(crop, f"{fruit} {episode.name[-6:]} {stage} | {name}"))
    sheet(images, len(candidates) * 3, (240, 264)).save(
        output / "rgb_candidate_comparison_224.jpg", quality=94
    )


def wrist_comparison(episodes: list[Path], output: Path) -> None:
    images = []
    candidates = CANDIDATES["wrist"]
    for episode in representative_episodes(episodes):
        records, fruit, segments = episode_info(episode)
        segment_index = {stage: (start + end) // 2 for stage, start, end, _ in segments}
        for stage in ("prepare", "grasp", "place"):
            index = segment_index[stage]
            original = source_image(episode, records[index], "wrist")
            for name, box in candidates.items():
                crop = original.crop(box).resize(
                    (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
                )
                images.append(label(crop, f"{fruit} {episode.name[-6:]} {stage} | {name}"))
    sheet(images, len(candidates) * 3, (240, 264)).save(
        output / "wrist_candidate_comparison_224.jpg", quality=94
    )


def end_contact_sheet(episodes: list[Path], output: Path) -> None:
    images = []
    closeups = []
    for episode in episodes:
        records, fruit, _ = episode_info(episode)
        index = len(records) - 1
        image = source_image(episode, records[index], "rgb")
        images.append(label(image, f"{episode.name[-6:]} {fruit} end f={index}"))
        closeups.append(
            label(
                image.crop((720, 520, 1320, 1080)),
                f"{episode.name[-6:]} {fruit} end f={index}",
            )
        )
    sheet(images, 5, (384, 240)).save(output / "rgb_episode_end_contact.jpg", quality=94)
    sheet(closeups, 5, (360, 360)).save(
        output / "rgb_episode_end_box_closeup.jpg", quality=94
    )


def force_contact_review(episodes: list[Path], output: Path) -> None:
    flagged = []
    for episode in episodes:
        records = load_jsonl(episode / "observations.jsonl")
        peak = min(range(len(records)), key=lambda index: float(records[index]["force_torque"][2]))
        if float(records[peak]["force_torque"][2]) < -20.0:
            flagged.append((episode, records, peak))
    for camera in ("rgb", "wrist"):
        images = []
        for episode, records, peak in flagged:
            reset = next(index for index, record in enumerate(records) if record.get("stage") == "reset")
            indices = sorted({max(0, peak - 5), peak, min(len(records) - 1, peak + 5), reset - 1})
            for index in indices:
                fz = float(records[index]["force_torque"][2])
                image = source_image(episode, records[index], camera)
                images.append(label(image, f"{episode.name[-6:]} {camera} f={index} Fz={fz:.1f}N"))
        sheet(images, 4, (480, 294)).save(
            output / f"{camera}_high_force_contact_review.jpg", quality=94
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / "review_20260911"
    episodes = sorted(path for path in args.root.glob("pi0_train_*") if path.is_dir())
    output.mkdir(parents=True, exist_ok=True)
    crop_sheets(episodes, output)
    rgb_comparison(episodes, output)
    wrist_comparison(episodes, output)
    end_contact_sheet(episodes, output)
    force_contact_review(episodes, output)
    print(f"episodes={len(episodes)} output={output.resolve()}")
    for camera, candidates in CANDIDATES.items():
        for name, box in candidates.items():
            print(f"preview only: {camera}.{name}={box} -> 224x224")
    print("SOURCE IMAGES AND JSONL UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
