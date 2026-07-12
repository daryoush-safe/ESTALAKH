from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grid_extraction import GridNotFoundError, extract_grid


def load_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"warning: no label file for {image_path.name}, skipping")
            continue
        pairs.append((image_path, label_path))
    return pairs


def split_pairs(pairs, test_ratio: float, seed: int):
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_ratio)
    return shuffled[n_test:], shuffled[:n_test]


def convert(pairs, split: str, out_dir: Path) -> tuple[int, int]:
    images_written = 0
    images_failed = 0

    for image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"warning: cannot read {image_path.name}, skipping")
            images_failed += 1
            continue

        grid_labels = np.loadtxt(label_path, delimiter=",", dtype=int)

        try:
            result = extract_grid(image)
        except GridNotFoundError as error:
            print(f"warning: {image_path.name}: {error}, skipping")
            images_failed += 1
            continue

        for index, cell in enumerate(result.cells):
            row, col = divmod(index, 9)
            label = int(grid_labels[row, col])
            class_dir = out_dir / split / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            dest = class_dir / f"{image_path.stem}_r{row}c{col}.png"
            cv2.imwrite(str(dest), cell.image)
            images_written += 1

    return images_written, images_failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert generated Persian Sudoku images/labels into a "
        "per-class train/test digit dataset (same layout as "
        "data/persian_digit_finetune_dataset)."
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("out/persian_sudoku_dataset_font/images"),
        help="directory of generated Sudoku images",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("out/persian_sudoku_dataset_font/labels"),
        help="directory of matching label CSVs",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("out/persian_digit_finetune_dataset"),
        help="output dataset directory (train/<label>/*.png, test/<label>/*.png)",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="fraction of images held out for test")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed for the train/test split")
    parser.add_argument("--clean", action="store_true", help="delete --out before writing")
    args = parser.parse_args(argv)

    pairs = load_pairs(args.images_dir, args.labels_dir)
    if not pairs:
        print(f"error: no image/label pairs found in {args.images_dir}", file=sys.stderr)
        return 1

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    train_pairs, test_pairs = split_pairs(pairs, args.test_ratio, args.seed)
    print(f"{len(pairs)} source images -> {len(train_pairs)} train / {len(test_pairs)} test")

    total_written, total_failed = 0, 0
    for split, split_pairs_ in (("train", train_pairs), ("test", test_pairs)):
        written, failed = convert(split_pairs_, split, args.out)
        print(f" {split}: {written} cell crops written, {failed} source images failed extraction")
        total_written += written
        total_failed += failed

    print(f"\nDone! {total_written} cell crops written to {args.out.resolve()} ({total_failed} images skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
