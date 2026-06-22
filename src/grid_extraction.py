from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WARP_SIZE = 450
CELL_SIZE = WARP_SIZE // 9
DIGIT_SIZE = 224
MAX_DETECT_DIM = 1024  # downsample target for grid detection (block size 11 is good for 500–1024px images)


MIN_DIGIT_AREA_RATIO = 0.015
MIN_DIGIT_HEIGHT_RATIO = 0.25
CELL_MARGIN_RATIO = 0.12
MIN_GRID_AREA_RATIO = 0.05 


class GridNotFoundError(RuntimeError):
    """Raised when no plausible Sudoku grid is found in the image."""


@dataclass
class Cell:
    image: np.ndarray
    is_empty: bool
    ink_ratio: float


@dataclass
class GridExtraction:
    corners: np.ndarray # 4x2 float32 (tl, tr, br, bl) in the source image
    matrix: np.ndarray # perspective: source -> warped
    inverse_matrix: np.ndarray  # perspective: warped -> source (Phase 4)
    warped: np.ndarray 
    cells: list[Cell] # row major
    stages: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def empty_mask(self):
        return np.array([c.is_empty for c in self.cells]).reshape(9, 9)


def extract_grid(image, keep_stages = False) -> GridExtraction:
    # if not gray, make gray
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = image
    else:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    stages = {}

    # some robustness stuff
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    denoised = cv2.medianBlur(normalized, 5)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    if keep_stages:
        stages["01_gray"] = gray
        stages["02_clahe"] = normalized
        stages["03_denoised"] = blurred
        stages["04_binary"] = binary

    detect_scale = min(1.0, MAX_DETECT_DIM / max(gray.shape[:2]))
    if detect_scale < 1.0:
        dh = int(gray.shape[0] * detect_scale)
        dw = int(gray.shape[1] * detect_scale)
        detect_gray = cv2.resize(gray, (dw, dh))
        detect_norm = clahe.apply(detect_gray)
        detect_blur = cv2.GaussianBlur(cv2.medianBlur(detect_norm, 5), (5, 5), 0)
        detect_bin = cv2.adaptiveThreshold(
            detect_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
    else:
        detect_blur = blurred
        detect_bin = binary

    corners = _locate_grid(detect_bin, detect_blur)
    if corners is None:
        raise GridNotFoundError("no Sudoku grid found")

    if detect_scale < 1.0:
        corners = corners / detect_scale

    corners = _order_corners(corners)

    # the first set of corners we find is often not very accurate,
    # so we get the cornners of the first pass warp and then find the cornors again
    corners = _refine_corners(normalized, corners)

    if keep_stages:
        outline = color.copy()
        cv2.polylines(outline, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        for point in corners.astype(int):
            cv2.circle(outline, tuple(point), 8, (0, 0, 255), -1)
        stages["05_grid_outline"] = outline

    matrix = cv2.getPerspectiveTransform(corners, _warp_destination())
    inverse_matrix = np.linalg.inv(matrix)

    # one problem we had was the hole for 6/8/9 getting filled in by the median blur,
    # so we use a more permissive threshold to get the grid lines,
    #  and a more strict one to get the digit mask for cell extraction
    warped = cv2.warpPerspective(normalized, matrix, (WARP_SIZE, WARP_SIZE))
    warped_blurred = cv2.medianBlur(warped, 3)

    # keeps grid lines thick enough for boundary detection
    warped_binary = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )
    # C=15 removes noise pixels
    # line geometry from warped_binary is applied to it
    warped_binary_clean = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15,
    )

    line_free, horizontal_lines, vertical_lines = _remove_grid_lines(warped_binary, warped_binary_clean)
    row_bounds = _grid_boundaries(horizontal_lines, axis=1)
    col_bounds = _grid_boundaries(vertical_lines, axis=0)

    if keep_stages:
        stages["06_warped"] = warped
        stages["07_warped_binary"] = warped_binary
        stages["08_lines_removed"] = line_free
        cell_grid = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        for bound in row_bounds:
            cv2.line(cell_grid, (0, int(bound)), (WARP_SIZE, int(bound)), (0, 255, 0), 1)
        for bound in col_bounds:
            cv2.line(cell_grid, (int(bound), 0), (int(bound), WARP_SIZE), (0, 255, 0), 1)
        stages["09_cell_grid"] = cell_grid

    cells = [
        _extract_cell(line_free, warped, row_bounds, col_bounds, row, col)
        for row in range(9)
        for col in range(9)
    ]

    return GridExtraction(
        corners=corners,
        matrix=matrix,
        inverse_matrix=inverse_matrix,
        warped=warped,
        cells=cells,
        stages=stages,
    )


def _warp_destination() -> np.ndarray:
    # get the corners of the warped grid in the order (tl, tr, br, bl)
    return np.array(
        [[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
        dtype=np.float32,
    )


def _refine_corners(normalized, corners) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(corners, _warp_destination())
    warped = cv2.warpPerspective(normalized, matrix, (WARP_SIZE, WARP_SIZE))

    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(warped, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )

    # isolate long horizontal and vertical lines (about 45 pixels long)
    length = WARP_SIZE // 10
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    
    # Combine them.
    grid_lines = cv2.bitwise_or(horizontal, vertical)
    grid_lines = cv2.dilate(grid_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(grid_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return corners

    largest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w < 0.5 * WARP_SIZE or h < 0.5 * WARP_SIZE:
        return corners

    if w > 0.95 * WARP_SIZE and h > 0.95 * WARP_SIZE and x < 0.05 * WARP_SIZE and y < 0.05 * WARP_SIZE:
        return corners

    perimeter = cv2.arcLength(largest_contour, True)
    
    for epsilon in (0.02, 0.05, 0.1):
        quad = cv2.approxPolyDP(largest_contour, epsilon * perimeter, True)
        if len(quad) == 4 and cv2.isContourConvex(quad):
            refined = cv2.perspectiveTransform(
                quad.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(matrix)
            ).reshape(4, 2)
            return _order_corners(refined)
            
    quad = np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h]
    ], dtype=np.float32)
    
    refined = cv2.perspectiveTransform(
        quad.reshape(-1, 1, 2), np.linalg.inv(matrix)
    ).reshape(4, 2)
    
    return _order_corners(refined)

def _locate_grid(binary, blurred) -> np.ndarray | None:
    min_area = MIN_GRID_AREA_RATIO * binary.size

    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.05, 0.1):
            quad = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                return quad.reshape(4, 2).astype(np.float32)

    corners = _corners_from_hough(blurred, min_area)
    if corners is not None:
        return corners
    
    return None


def _corners_from_hough(blurred, min_area) -> np.ndarray | None:
    print("herehereedfjkahfjhfhjdhfkakj")

    edges = cv2.Canny(blurred, 50, 150)
    threshold = max(80, min(blurred.shape) // 4)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    if lines is None:
        return None

    horizontal = []
    vertical = []

    for dist_to_tl, theta in lines[:, 0]:
        if dist_to_tl < 0:
            dist_to_tl, theta = -dist_to_tl, theta - np.pi

        if abs(theta) < np.pi / 4:
            vertical.append((dist_to_tl, theta))

        elif abs(theta - np.pi / 2) < np.pi / 4:
            horizontal.append((dist_to_tl, theta))


    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    top, bottom = min(horizontal), max(horizontal)
    left, right = min(vertical), max(vertical)
    points = []

    for pair in ((top, left), (top, right), (bottom, right), (bottom, left)):
        point = _intersect_lines(*pair)
        if point is None:
            return None
        points.append(point)
    corners = np.array(points, dtype=np.float32)

    if cv2.contourArea(corners) < min_area:
        return None
    if not cv2.isContourConvex(corners.astype(np.int32)):
        return None
    return corners


def _intersect_lines(line_a, line_b) -> tuple[float, float] | None:
    (rho_a, theta_a), (rho_b, theta_b) = line_a, line_b
    coefficients = np.array(
        [[np.cos(theta_a), np.sin(theta_a)], [np.cos(theta_b), np.sin(theta_b)]]
    )
    if abs(np.linalg.det(coefficients)) < 1e-8:
        return None
    x, y = np.linalg.solve(coefficients, np.array([rho_a, rho_b]))
    return float(x), float(y)


def _order_corners(points) -> np.ndarray:
    # 4 points as (tl, tr, br, bl) — invariant to rotations below 45 degrees
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[sums.argmin()],   # top-left: smallest x + y
            points[diffs.argmin()],  # top-right: smallest y - x
            points[sums.argmax()],   # bottom-right: largest x + y
            points[diffs.argmax()],  # bottom-left: largest y - x
        ],
        dtype=np.float32,
    )




def _remove_grid_lines(warped_binary, digit_binary=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CELL_SIZE, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, CELL_SIZE))
    horizontal = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    target = digit_binary if digit_binary is not None else warped_binary
    line_free = cv2.bitwise_and(target, cv2.bitwise_not(lines))
    return line_free, horizontal, vertical


def _grid_boundaries(lines_mask, axis) -> np.ndarray:
    profile = (lines_mask > 0).sum(axis=axis)
    search = CELL_SIZE // 3
    bounds = np.empty(10, dtype=int)

    for index in range(10):
        expected = min(index * CELL_SIZE, WARP_SIZE - 1)
        low = max(0, expected - search)
        window = profile[low : min(WARP_SIZE, expected + search + 1)]

        if window.max() >= 0.3 * WARP_SIZE:
            bounds[index] = low + int(window.argmax())
        else:
            bounds[index] = expected

    return np.maximum.accumulate(bounds)


def _extract_cell(line_free, warped, row_bounds, col_bounds, row, col,) -> Cell:

    y0, y1 = row_bounds[row], row_bounds[row + 1]
    x0, x1 = col_bounds[col], col_bounds[col + 1]
    y0, y1 = y0 + int((y1 - y0) * CELL_MARGIN_RATIO), y1 - int((y1 - y0) * CELL_MARGIN_RATIO)
    x0, x1 = x0 + int((x1 - x0) * CELL_MARGIN_RATIO), x1 - int((x1 - x0) * CELL_MARGIN_RATIO)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return Cell(np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8), True, 0.0)
    
    interior = line_free[y0:y1, x0:x1]
    ink_ratio = float(cv2.countNonZero(interior)) / interior.size

    digit_mask = _find_digit_mask(interior)
    if digit_mask is None:
        return Cell(np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8), True, ink_ratio)

    # crop the digit from the warped *grayscale*: a binary mask fills the
    # holes of 6/8/9 under noise, grayscale keeps them visible for Phase 2
    ys, xs = np.nonzero(digit_mask)
    pad = 3
    gy0, gy1 = max(0, y0 + ys.min() - pad), min(WARP_SIZE, y0 + ys.max() + 1 + pad)
    gx0, gx1 = max(0, x0 + xs.min() - pad), min(WARP_SIZE, x0 + xs.max() + 1 + pad)
    mask_full = np.zeros(line_free.shape, np.uint8)
    mask_full[y0:y1, x0:x1] = digit_mask
    digit = _normalize_digit(warped[gy0:gy1, gx0:gx1], mask_full[gy0:gy1, gx0:gx1])
    return Cell(digit, False, ink_ratio)


def _find_digit_mask(interior) -> np.ndarray | None:
    height, width = interior.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=8)

    anchor, anchor_area = None, 0
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        
        # size checks
        if area < MIN_DIGIT_AREA_RATIO * interior.size:
            continue
        if h < MIN_DIGIT_HEIGHT_RATIO * height:
            continue

        # aspect Ratio: Digits are generally taller than they are wide.
        if w > 1.2 * h:
            continue
            
        # fill Density: Wispy smudges/wrinkles have large bounding boxes but very few actual pixels. Digits are solid strokes.
        density = area / (w * h)
        if density < 0.15:
            continue

        # centroid check
        cx, cy = centroids[label]
        if not (0.15 * width < cx < 0.85 * width and 0.15 * height < cy < 0.85 * height):
            continue
            
        if area > anchor_area:
            anchor, anchor_area = label, area

    if anchor is None:
        return None

    ax, ay, aw, ah, _ = stats[anchor]
    grow = max(2, min(height, width) // 8)
    left, top = ax - grow, ay - grow
    right, bottom = ax + aw + grow, ay + ah + grow
    min_fragment = 0.3 * MIN_DIGIT_AREA_RATIO * interior.size

    mask = np.zeros_like(interior)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if label != anchor:
            if area < min_fragment:
                continue
            if x + w < left or y + h < top or x > right or y > bottom:
                continue
        mask[labels == label] = 255
        
    return mask


def _normalize_digit(gray_crop, mask_crop) -> np.ndarray:
    # 1. Closes the mask with a scale aware kernel (~4% of the smaller side)
    #    to bridge small gaps without thickening strokes or filling loops.
    # 2. Inverts the grayscale (white digit on black background).
    # 3. Masks out everything outside the digit.
    # 4. Normalizes pixel values to 0–255.
    # 5. Pads to a square canvas at 1.3× the digit size.
    # 6. Resize

    h, w = mask_crop.shape
    k = max(1, round(min(h, w) * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel)

    inverted = cv2.bitwise_not(gray_crop)
    digit = cv2.bitwise_and(inverted, inverted, mask=mask)
    digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)
    height, width = digit.shape
    side = int(max(height, width) * 1.3)
    canvas = np.zeros((side, side), np.uint8)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = digit
    return cv2.resize(canvas, (DIGIT_SIZE, DIGIT_SIZE), interpolation=cv2.INTER_AREA)







def cell_montage(cells) -> np.ndarray:
    pad, tile = 3, DIGIT_SIZE
    step = tile + 2 * pad
    canvas = np.full((9 * step, 9 * step, 3), 40, np.uint8)
    for index, cell in enumerate(cells):
        row, col = divmod(index, 9)
        y0, x0 = row * step, col * step
        frame_color = (40, 40, 40) if cell.is_empty else (0, 160, 0)
        cv2.rectangle(canvas, (x0, y0), (x0 + step - 1, y0 + step - 1), frame_color, pad)
        patch = cv2.cvtColor(cell.image, cv2.COLOR_GRAY2BGR)
        canvas[y0 + pad : y0 + pad + tile, x0 + pad : x0 + pad + tile] = patch
    return canvas


def empty_mask_text(result) -> str:
    #  '#' = digit , '.' = empty
    return "\n".join(
        " ".join("." if empty else "#" for empty in row)
        for row in result.empty_mask
    )


def save_report(result, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, stage in result.stages.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), stage)
    cv2.imwrite(str(out_dir / "10_cells_montage.png"), cell_montage(result.cells))
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    for index, cell in enumerate(result.cells):
        if not cell.is_empty:
            row, col = divmod(index, 9)
            cv2.imwrite(str(cells_dir / f"cell_r{row}c{col}.png"), cell.image)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 — Sudoku grid extraction")
    parser.add_argument("image", type=Path, help="path to a Sudoku photo")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="directory for stage images and cell crops")
    args = parser.parse_args(argv)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"error: cannot read image {args.image}", file=sys.stderr)
        return 1

    try:
        result = extract_grid(image, keep_stages=True)
    except GridNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    save_report(result, args.out)
    filled = sum(not cell.is_empty for cell in result.cells)
    print(f"grid found; {filled} filled cells, {81 - filled} empty")
    print(empty_mask_text(result))
    print(f"stages + cells written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
