import io
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from pathlib import Path

from src.grid_extraction import GridNotFoundError, cell_montage, extract_grid
from src.orientation.infer import DEFAULT_CHECKPOINT, _DEGREES, predict_orientation
from src.orientation.model import load_model as load_orientation_model
from src.recognition.infer import predict_cell
from src.recognition.model import load_model
from src.solver import print_grid, solve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sudoku.api")

ml: dict[str, torch.nn.Module] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    dv = "cuda" if torch.cuda.is_available() else "cpu"
    ml["recognition"] = load_model()
    ml["orientation"] = load_orientation_model(str(DEFAULT_CHECKPOINT), device=dv)
    yield
    ml.clear()


app = FastAPI(title="Image Classification Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/cell")
async def predict(file: UploadFile = File(...)):
    model = ml.get("recognition")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    predicted_class = predict_cell(model, image)

    return {"predicted_class": predicted_class}


def _read_upload(file: UploadFile, key: str) -> tuple[torch.nn.Module, Image.Image]:
    model = ml.get(key)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    return model


@app.post("/predict/orientation")
async def predict_orientation_endpoint(file: UploadFile = File(...)):
    model = ml.get("orientation")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")
    
    label = predict_orientation(model, image)

    return {"label": label, "degrees": _DEGREES[label]}


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


def predict_grid(
    cell_model, bgr_image, save_dir: Path | None = None
) -> list[list[int]]:
    extraction = extract_grid(bgr_image)

    grid = [[0] * 9 for _ in range(9)]
    for index, cell in enumerate(extraction.cells):
        row, col = divmod(index, 9)
        cell_pil = Image.fromarray(cell.image).convert("RGB")
        grid[row][col] = predict_cell(cell_model, cell_pil)

    if save_dir is not None:
        _save_extraction(extraction, grid, save_dir)
    return grid


@app.post("/solve")
async def solve_sudoku(file: UploadFile = File(...), debug: bool = False):
    # orientation_model = ml.get("orientation")
    cell_model = ml.get("recognition")
    if cell_model is None:
        raise HTTPException(status_code=503, detail="Recognition model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    # degree_index = predict_orientation(orientation_model, image)
    # degrees = _DEGREES[degree_index]

    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    source = file.filename or "upload"
    save_dir = Path("out/solve") / Path(source).stem if debug else None
    try:
        grid = predict_grid(cell_model, bgr_image, save_dir=save_dir)
    except GridNotFoundError:
        raise HTTPException(status_code=422, detail="No Sudoku grid found in the image")

    solution = solve(grid)
    if solution is None:
        raise HTTPException(status_code=422, detail="Could not solve the predicted grid")

    return {"grid": grid, "solution": solution}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)