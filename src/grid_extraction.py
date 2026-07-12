from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WARP_SIZE = 450
CELL_SIZE = WARP_SIZE // 9

WARP_MARGIN = CELL_SIZE // 2
GRID_SPAN = WARP_SIZE - 1 - 2 * WARP_MARGIN
DIGIT_SIZE = 224
MAX_DETECT_DIM = 1024  # downsample target for grid detection (block size 11 is good for 500–1024px images)


MIN_DIGIT_AREA_RATIO = 0.015
MIN_DIGIT_HEIGHT_RATIO = 0.25

CELL_MARGIN_RATIO = 0.12
MIN_GRID_AREA_RATIO = 0.05 


ROTATION_CODE = {
    0: None,
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}


class GridNotFoundError(RuntimeError):
    """Raised when no plausible Sudoku grid is found in the image."""


@dataclass
class Cell:
    image: np.ndarray
    is_empty: bool


@dataclass
class GridExtraction:
    corners: np.ndarray # 4x2 float32 (tl, tr, br, bl) in the source image
    matrix: np.ndarray # perspective: source -> warped
    inverse_matrix: np.ndarray  # perspective: warped -> source (Phase 4)
    warped: np.ndarray
    cells: list[Cell] # row major
    cell_quads: np.ndarray | None = None
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

    # if polarity is reversed, invert to dark digits on light background
    if _is_light_on_dark(gray):
        gray = cv2.bitwise_not(gray)

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

    warped = cv2.warpPerspective(normalized, matrix, (WARP_SIZE, WARP_SIZE))
    cells, cell_quads, warp_stages = _process_warp(warped, keep_stages=keep_stages)
    stages.update(warp_stages)

    return GridExtraction(
        corners=corners,
        matrix=matrix,
        inverse_matrix=inverse_matrix,
        warped=warped,
        cells=cells,
        cell_quads=cell_quads,
        stages=stages,
    )


def _is_light_on_dark(gray) -> bool:
    h, w = gray.shape[:2]
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    center = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(center, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_fraction = float(cv2.countNonZero(mask)) / mask.size
    return bright_fraction < 0.5


def _warp_destination() -> np.ndarray:
    # corners of the warped grid in the order (tl, tr, br, bl), inset by WARP_MARGIN
    low, high = WARP_MARGIN, WARP_SIZE - 1 - WARP_MARGIN
    return np.array(
        [[low, low], [high, low], [high, high], [low, high]],
        dtype=np.float32,
    )


def _process_warp(warped, keep_stages=False):
    warped_blurred = cv2.medianBlur(warped, 3)

    # one problem we had was the hole for 6/8/9 getting filled in by the median
    # blur, so we use a more permissive threshold to keep grid lines thick enough
    # for boundary detection, and a stricter one for the digit mask.
    warped_binary = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10,
    )
    # C=15 removes noise pixels; line geometry from warped_binary is applied to it
    warped_binary_clean = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15,
    )

    line_free, horizontal_lines, vertical_lines = _remove_grid_lines(warped_binary, warped_binary_clean)
    row_bounds = _grid_boundaries(horizontal_lines, axis=1)
    col_bounds = _grid_boundaries(vertical_lines, axis=0)

    # local cell for situations like page fold
    quads, quad_stages = _detect_cell_quads(
        warped_binary, row_bounds, col_bounds,
        visual_base=warped if keep_stages else None,
    )

    if quads is not None:
        cells = [_extract_cell_quad(line_free, warped, quad) for quad in quads]
    else:
        cells = [
            _extract_cell(line_free, warped, row_bounds, col_bounds, row, col)
            for row in range(9)
            for col in range(9)
        ]

    stages = {}
    if keep_stages:
        stages["06_warped"] = warped
        stages["07_warped_binary"] = warped_binary
        stages["08_lines_removed"] = line_free
        stages.update(quad_stages)
        cell_grid = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        if quads is not None:
            polys = [quad.astype(np.int32) for quad in quads]
            cv2.polylines(cell_grid, polys, True, (0, 255, 0), 1)
        else:
            for bound in row_bounds:
                cv2.line(cell_grid, (0, int(bound)), (WARP_SIZE, int(bound)), (0, 255, 0), 1)
            for bound in col_bounds:
                cv2.line(cell_grid, (int(bound), 0), (int(bound), WARP_SIZE), (0, 255, 0), 1)
        stages["09_cell_grid"] = cell_grid

    return cells, quads, stages




def _rotation_homography(label, size=WARP_SIZE):
    w = size - 1
    label %= 4
    if label == 0:
        return np.eye(3, dtype=np.float64)
    if label == 1: 
        return np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64)
    if label == 2:
        return np.array([[-1, 0, w], [0, -1, w], [0, 0, 1]], dtype=np.float64)
    return np.array([[0, -1, w], [1, 0, 0], [0, 0, 1]], dtype=np.float64)


def rotate_extraction(extraction: GridExtraction, label: int, keep_stages: bool = False) -> GridExtraction:
    label %= 4
    if label == 0:
        return extraction

    new_warped = cv2.rotate(extraction.warped, ROTATION_CODE[label])
    rot = _rotation_homography(label)
    new_matrix = rot @ extraction.matrix  # source -> old warp -> upright warp
    new_inverse = np.linalg.inv(new_matrix)

    cells, cell_quads, warp_stages = _process_warp(new_warped, keep_stages=keep_stages)

    new_stages = {}
    if keep_stages:
        new_stages = {
            k: v for k, v in extraction.stages.items()
            if not k.startswith(("06", "07", "08", "09"))
        }
        new_stages.update(warp_stages)

    return GridExtraction(
        corners=extraction.corners,
        matrix=new_matrix,
        inverse_matrix=new_inverse,
        warped=new_warped,
        cells=cells,
        cell_quads=cell_quads,
        stages=new_stages,
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

    # pad when the grid is close to the edge of the image, so that the contour can be closed
    pad = max(8, int(0.02 * max(binary.shape[:2])))
    binary = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    blurred = cv2.copyMakeBorder(blurred, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.05, 0.1):
            quad = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                return quad.reshape(4, 2).astype(np.float32) - pad

        # A grid whose outer border is faint can not to reduce to a clean quad, even though
        # this contour clearly is the grid. Its min-area rect still spans the
        # full grid extent, so use that (square-ish and grid-sized)
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw > 0 and rh > 0 and rw * rh >= min_area and 0.5 < rw / rh < 2.0:
            return cv2.boxPoints(rect).astype(np.float32) - pad

    corners = _corners_from_hough(blurred, min_area)
    if corners is not None:
        return corners - pad

    return None


def _corners_from_hough(blurred, min_area) -> np.ndarray | None:
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
        expected = int(round(WARP_MARGIN + index * GRID_SPAN / 9))
        low = max(0, expected - search)
        window = profile[low : min(WARP_SIZE, expected + search + 1)]

        if window.max() >= 0.3 * WARP_SIZE:
            bounds[index] = low + int(window.argmax())
        else:
            bounds[index] = expected

    return np.maximum.accumulate(bounds)


def _line_mask(binary) -> np.ndarray:
    thicken = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    run = cv2.getStructuringElement(cv2.MORPH_RECT, (CELL_SIZE // 2, 1))
    opened = cv2.morphologyEx(cv2.dilate(binary, thicken), cv2.MORPH_OPEN, run)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    keep = stats[:, cv2.CC_STAT_WIDTH] >= CELL_SIZE
    keep[0] = False  # background
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _trace_curve(mask, expected):
    half = CELL_SIZE // 2 - 4
    lo = max(0, int(expected) - half)
    hi = min(WARP_SIZE, int(expected) + half + 1)
    band = mask[lo:hi] > 0
    counts = band.sum(axis=0)
    sampled = counts > 0
    if sampled.sum() < 0.35 * WARP_SIZE:
        return None
    rows = np.arange(lo, hi, dtype=np.float64)[:, None]
    centers = (band * rows).sum(axis=0)[sampled] / counts[sampled]
    fit = np.polyfit(np.nonzero(sampled)[0], centers, 2)
    curve = np.polyval(fit, np.arange(WARP_SIZE))
    return np.clip(curve, lo, hi - 1)


def _trace_lines(mask, bounds) -> tuple[np.ndarray, int]:
    curves, traced = [], 0
    for expected in bounds:
        curve = _trace_curve(mask, expected)
        traced += curve is not None
        curves.append(curve if curve is not None else np.full(WARP_SIZE, float(expected)))
    # neighbouring lines must never cross
    return np.maximum.accumulate(np.stack(curves), axis=0), traced


def _detect_cell_quads(warped_binary, row_bounds, col_bounds, visual_base=None):
    h_mask = _line_mask(warped_binary)
    v_mask = _line_mask(warped_binary.T).T

    h_curves, h_traced = _trace_lines(h_mask, row_bounds)
    v_curves, v_traced = _trace_lines(v_mask.T, col_bounds)

    stages = {}
    if visual_base is not None:
        stages["08a_line_masks"] = _draw_line_masks(h_mask, v_mask)

    if h_traced + v_traced < 8:
        return None, stages  # the uniform split is just as good

    # node (i, j) = where horizontal curve i crosses vertical curve j
    nodes = np.empty((10, 10, 2), dtype=np.float32)
    for i in range(10):
        for j in range(10):
            y = float(row_bounds[i])
            for _ in range(2):
                x = v_curves[j][int(y)]
                y = h_curves[i][int(x)]
            nodes[i, j] = (x, y)

    if visual_base is not None:
        stages["08b_line_curves"] = _draw_line_curves(visual_base, h_curves, v_curves, nodes)

    tl, tr = nodes[:-1, :-1], nodes[:-1, 1:]
    bl, br = nodes[1:, :-1], nodes[1:, 1:]
    quads = np.stack([tl, tr, br, bl], axis=2).reshape(81, 4, 2)
    return quads, stages


def _draw_line_masks(h_mask, v_mask) -> np.ndarray:
    image = np.zeros((*h_mask.shape, 3), np.uint8)
    image[h_mask > 0] = (0, 255, 0)
    image[v_mask > 0] = (0, 0, 255)
    image[(h_mask > 0) & (v_mask > 0)] = (0, 255, 255)
    return image


def _draw_line_curves(base, h_curves, v_curves, nodes) -> np.ndarray:
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    span = np.arange(WARP_SIZE)
    for curve in h_curves:
        cv2.polylines(image, [np.stack([span, curve], axis=1).astype(np.int32)], False, (0, 255, 0), 1)
    for curve in v_curves:
        cv2.polylines(image, [np.stack([curve, span], axis=1).astype(np.int32)], False, (0, 0, 255), 1)
    for point in nodes.reshape(-1, 2).astype(int):
        cv2.circle(image, tuple(point), 3, (255, 0, 255), -1)
    return image


def _extract_cell_quad(line_free, warped, quad) -> Cell:
    destination = np.array(
        [[0, 0], [CELL_SIZE - 1, 0], [CELL_SIZE - 1, CELL_SIZE - 1], [0, CELL_SIZE - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(quad, destination)
    line_free_patch = cv2.warpPerspective(
        line_free, homography, (CELL_SIZE, CELL_SIZE), flags=cv2.INTER_NEAREST
    )
    gray_patch = cv2.warpPerspective(warped, homography, (CELL_SIZE, CELL_SIZE))
    return _cell_from_patch(line_free_patch, gray_patch)


def _extract_cell(line_free, warped, row_bounds, col_bounds, row, col,) -> Cell:
    y0, y1 = row_bounds[row], row_bounds[row + 1]
    x0, x1 = col_bounds[col], col_bounds[col + 1]
    return _cell_from_patch(line_free[y0:y1, x0:x1], warped[y0:y1, x0:x1])


def _cell_from_patch(line_free_patch, gray_patch) -> Cell:
    height, width = line_free_patch.shape[:2]
    y0, y1 = int(height * CELL_MARGIN_RATIO), height - int(height * CELL_MARGIN_RATIO)
    x0, x1 = int(width * CELL_MARGIN_RATIO), width - int(width * CELL_MARGIN_RATIO)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return Cell(np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8), True)

    interior = line_free_patch[y0:y1, x0:x1]

    digit_mask = _find_digit_mask(interior)
    if digit_mask is None:
        return Cell(np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8), True)

    # crop the digit from the warped *grayscale*: a binary mask fills the
    # holes of 6/8/9 under noise, grayscale keeps them visible for Phase 2
    ys, xs = np.nonzero(digit_mask)
    pad = 3
    gy0, gy1 = max(0, y0 + ys.min() - pad), min(height, y0 + ys.max() + 1 + pad)
    gx0, gx1 = max(0, x0 + xs.min() - pad), min(width, x0 + xs.max() + 1 + pad)

    mask_full = np.zeros((height, width), np.uint8)
    mask_full[y0:y1, x0:x1] = digit_mask
    digit = _normalize_digit(gray_patch[gy0:gy1, gx0:gx1], mask_full[gy0:gy1, gx0:gx1])
    return Cell(digit, False)


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
