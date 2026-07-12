import base64
import io
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from pathlib import Path

from src.grid_extraction import GridExtraction, GridNotFoundError, cell_montage, extract_grid, rotate_extraction
from src.orientation.infer import DEFAULT_WARP_CHECKPOINT, _DEGREES, resolve_orientation
from src.orientation.model import load_model as load_orientation_model
from src.overlay import render_solution
from src.recognition.infer import predict_cell, predict_cell_proba
from src.recognition.model import load_model
from src.solver import solve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sudoku.api")

ml: dict[str, torch.nn.Module] = {}

ENGLISH_RECOGNITION_KEY = "english_recognition"
PERSIAN_RECOGNITION_KEY = "persian_recognition"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ml[ENGLISH_RECOGNITION_KEY], ml[PERSIAN_RECOGNITION_KEY] = load_model(device=device)
    # ml["orientation_warp"] = load_orientation_model(str(DEFAULT_WARP_CHECKPOINT), device=device)
    yield
    ml.clear()


app = FastAPI(title="Sudoku Solver", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_INDEX = Path(__file__).resolve().parents[2] / "ui" / "index.html"


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(_UI_INDEX, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/cell")
async def predict(file: UploadFile = File(...)):
    english_model = ml.get(ENGLISH_RECOGNITION_KEY)
    persian_model = ml.get(PERSIAN_RECOGNITION_KEY)
    if english_model is None or persian_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    predicted_class = predict_cell(english_model, image)

    return {"predicted_class": predicted_class}



def _save_extraction(extraction, grid, out_dir: Path) -> None:
    # debug only
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "warped.png"), extraction.warped)
    cv2.imwrite(str(out_dir / "cells_montage.png"), cell_montage(extraction.cells))
    for index, cell in enumerate(extraction.cells):
        row, col = divmod(index, 9)
        cv2.imwrite(str(cells_dir / f"r{row}c{col}_pred{grid[row][col]}.png"), cell.image)
    logger.info("saved extraction to %s", out_dir)


def predict_grid(persian_model, extraction: GridExtraction) -> tuple[list[list[int]], list[list[float]]]:
    grid = [[0] * 9 for _ in range(9)]
    confidences = [[0.0] * 9 for _ in range(9)]
    for index, cell in enumerate(extraction.cells):
        row, col = divmod(index, 9)
        cell_pil = Image.fromarray(cell.image).convert("RGB")
        value, conf = predict_cell_proba(persian_model, cell_pil)
        grid[row][col] = value
        confidences[row][col] = conf
    return grid, confidences


def _grid_score(grid: list[list[int]], confidences: list[list[float]]) -> tuple[int, float]:
    values = [confidences[r][c] for r in range(9) for c in range(9) if grid[r][c] != 0]
    avg_confidence = sum(values) / len(values) if values else 0.0
    return len(values), avg_confidence


def _encode_image(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode stage image")
    return base64.b64encode(buf).decode("ascii")


@app.post("/solve")
async def solve_sudoku(file: UploadFile = File(...), debug: bool = False, stages: bool = False):
    persian_model = ml.get(PERSIAN_RECOGNITION_KEY)
    english_model = ml.get(ENGLISH_RECOGNITION_KEY)
    if persian_model is None or english_model is None:
        raise HTTPException(status_code=503, detail="Recognition model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    try:
        extraction = extract_grid(bgr_image, keep_stages=stages)
    except GridNotFoundError:
        raise HTTPException(status_code=422, detail="No Sudoku grid found in the image")


    english_orientation_label = 0
    persian_orientation_label = 0
    english_orientation_label = resolve_orientation(extraction, recognition_model=english_model)
    persian_orientation_label = resolve_orientation(extraction, recognition_model=persian_model)
    
    english_extraction = rotate_extraction(extraction, english_orientation_label, keep_stages=stages)
    persian_extraction = rotate_extraction(extraction, persian_orientation_label, keep_stages=stages)

    english_grid, english_confidences = predict_grid(english_model, english_extraction)
    persian_grid, persian_confidences = predict_grid(persian_model, persian_extraction)
    
    if _grid_score(english_grid, english_confidences) >= _grid_score(persian_grid, persian_confidences):
        grid, confidences = english_grid, english_confidences
        orientation_label = english_orientation_label
        language = "english"
    else:
        grid, confidences = persian_grid, persian_confidences
        orientation_label = persian_orientation_label
        language = "persian"

    source = file.filename or "upload"
    if debug:
        _save_extraction(extraction, grid, Path("out/solve") / Path(source).stem)

    solution = solve(grid)

    given_mask = [[grid[r][c] != 0 for c in range(9)] for r in range(9)]
    response = {
        "grid": grid,
        "confidences": confidences,
        "given_mask": given_mask,
        "solution": solution,
        "language": language,
        "orientation": {"label": orientation_label, "degrees": _DEGREES[orientation_label]},
    }

    if stages:
        response["stages"] = {name: _encode_image(img) for name, img in extraction.stages.items()}

    if solution is None:
        response["error"] = "Could not solve the recognized grid — likely a misread digit."
        return response

    overlay_image = render_solution(bgr_image, extraction, solution, given_mask, language=language)
    response["overlay_image"] = _encode_image(overlay_image)
    return response



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)