import cv2
import numpy as np

from src.grid_extraction import GRID_SPAN, WARP_MARGIN, WARP_SIZE, GridExtraction


def _draw_centered_digit(canvas, center, cell_height, text, color, font, thickness):
    # ~60% of the cell height
    scale = cv2.getFontScaleFromHeight(font, int(cell_height * 0.6), thickness)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(canvas, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _cell_geometry(extraction: GridExtraction, row, col):
    if extraction.cell_quads is not None:
        quad = extraction.cell_quads[row * 9 + col]
        center = quad.mean(axis=0)
        cell_height = (quad[3][1] + quad[2][1] - quad[0][1] - quad[1][1]) / 2
        return center, max(8.0, cell_height)
    cell = GRID_SPAN / 9
    center = (WARP_MARGIN + (col + 0.5) * cell, WARP_MARGIN + (row + 0.5) * cell)
    return center, cell


def render_solution( image, extraction: GridExtraction, solved_grid, given_mask, color=(0, 180, 0), thickness= 3):
    
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas = np.zeros((WARP_SIZE, WARP_SIZE, 3), np.uint8)
    for row in range(9):
        for col in range(9):
            if given_mask[row][col]:
                continue
            value = solved_grid[row][col]
            if value == 0:
                continue
            center, cell_height = _cell_geometry(extraction, row, col)
            _draw_centered_digit(canvas, center, cell_height, str(value), color, font, thickness)

    warped_back = cv2.warpPerspective(canvas, extraction.inverse_matrix, (width, height))

    overlay_mask = warped_back.any(axis=2)
    out = image.copy()
    out[overlay_mask] = warped_back[overlay_mask]
    return out
