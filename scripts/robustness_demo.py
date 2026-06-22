from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grid_extraction import GridNotFoundError, extract_grid, save_report

PUZZLE = [
    "53..7....",
    "6..195...",
    ".98....6.",
    "8...6...3",
    "4..8.3..1",
    "7...2...6",
    ".6....28.",
    "...419..5",
    "....8..79",
]
GRID_SIDE = 900
CANVAS = (1600, 1200)


def synthesize_puzzle():
    side = GRID_SIDE
    cell = side // 9
    image = np.full((side, side), 255, np.uint8)
    for index in range(10):
        thickness = 7 if index % 3 == 0 else 2
        position = min(index * cell, side - 1)
        cv2.line(image, (position, 0), (position, side), 0, thickness)
        cv2.line(image, (0, position), (side, position), 0, thickness)
    for row in range(9):
        for col in range(9):
            char = PUZZLE[row][col]
            if char == ".":
                continue
            scale, thickness = 2.2, 6
            (w, h), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            org = (col * cell + (cell - w) // 2, row * cell + (cell + h) // 2)
            cv2.putText(image, char, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness)
    return image


def place_on_background(puzzle, corners):
    width, height = CANVAS
    ramp = np.linspace(120, 180, width, dtype=np.uint8)
    background = np.tile(ramp, (height, 1))

    side = puzzle.shape[0]
    source = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped = cv2.warpPerspective(puzzle, matrix, (width, height))
    mask = cv2.warpPerspective(np.full_like(puzzle, 255), matrix, (width, height))
    return np.where(mask > 0, warped, background)


def baseline():
    corners = np.array([[320, 140], [1280, 170], [1250, 1090], [350, 1060]])
    return place_on_background(synthesize_puzzle(), corners)



def low_light(image):
    dark = (image.astype(np.float32) * 0.3).clip(0, 255)
    return dark.astype(np.uint8)


def gaussian_noise(image):
    noise = np.random.default_rng(0).normal(0, 25, image.shape)
    return (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def motion_blur(image):
    return cv2.GaussianBlur(image, (9, 9), 0)


def rotation(image):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 25, 0.8)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=150)


def extreme_angle(_):
    corners = np.array([[520, 220], [1080, 220], [1380, 1050], [220, 1050]])
    return place_on_background(synthesize_puzzle(), corners)


def shadow(image):
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gradient = ((xx / width + yy / height - 0.8) * 2.5).clip(0, 1)
    factor = 1.0 - 0.55 * gradient
    return (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


CONDITIONS = {
    "baseline": lambda img: img,
    "low_light": low_light,
    "noise": gaussian_noise,
    "blur": motion_blur,
    "rotation": rotation,
    "extreme_angle": extreme_angle,
    "shadow": shadow,
}


def expected_empty_mask():
    return np.array([[char == "." for char in row] for row in PUZZLE])


def main():
    out_root = Path("out/robustness")
    clean = baseline()
    truth = expected_empty_mask()

    print(f"{'condition':<15} {'grid':<6} {'empty-cell accuracy':<20}")
    print("-" * 45)
    failures = 0
    for name, degrade in CONDITIONS.items():
        image = degrade(clean.copy())
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "00_input.png"), image)

        try:
            result = extract_grid(image, keep_stages=True)
        except GridNotFoundError:
            print(f"{name:<15} {'NO':<6} {'-':<20}")
            failures += 1
            continue

        save_report(result, out_dir)
        correct = int((result.empty_mask == truth).sum())
        status = f"{correct}/81"
        if correct < 81:
            failures += 1
            wrong = np.argwhere(result.empty_mask != truth)
            status += "  wrong at " + ", ".join(f"r{r}c{c}" for r, c in wrong[:5])
        print(f"{name:<15} {'yes':<6} {status:<20}")

    print(f"\noutputs written to {out_root}/")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
