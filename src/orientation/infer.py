import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

DEFAULT_CHECKPOINT = ROOT / "src" / "models" / "orientation_best.pth"

DEFAULT_WARP_CHECKPOINT = ROOT / "src" / "models" / "orientation_warp.pth"


_CORRECTION_CODE = {
    0: None,
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}

_DEGREES = {0: 0, 1: 90, 2: 180, 3: 270}


def _get_model(checkpoint, device):
    from src.orientation.model import load_model
    return load_model(str(checkpoint), device=device)


class OrientationCorrector:
    # lazy loading
    def __init__(self, checkpoint = DEFAULT_CHECKPOINT, device = ""):
        if not device:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.model = _get_model(checkpoint, device)
        self._transform = _build_transform()

    def predict_label(self, img_bgr) :
        from PIL import Image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            label = int(self.model(tensor).argmax(1).item())
        return label

    def correct(self, img_bgr):
        label = self.predict_label(img_bgr)
        code = _CORRECTION_CODE[label]
        corrected = cv2.rotate(img_bgr, code) if code is not None else img_bgr.copy()
        return corrected, label


def _build_transform():
    from torchvision import transforms
    from src.orientation.dataset import IMG_SIZE, MEAN, STD
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def predict_warp_orientation(model, warped, device=None):
    from PIL import Image

    if device is None:
        device = next(model.parameters()).device
    if warped.ndim == 2:
        pil = Image.fromarray(warped).convert("RGB")
    else:
        pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    tensor = _build_transform()(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        label = int(probs.argmax().item())
    return label, probs.cpu().tolist()


class WarpOrientationClassifier:
    def __init__(self, checkpoint=DEFAULT_WARP_CHECKPOINT, device=""):
        if not device:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.model = _get_model(checkpoint, device)

    def predict(self, warped):
        return predict_warp_orientation(self.model, warped, self.device)


def _count_conflicts(grid):
    def dup(vals):
        vals = [v for v in vals if v != 0]
        return len(vals) - len(set(vals))

    conflicts = 0
    for i in range(9):
        conflicts += dup(grid[i])
        conflicts += dup([grid[r][i] for r in range(9)])
    for bi in range(0, 9, 3):
        for bj in range(0, 9, 3):
            conflicts += dup([grid[r][c] for r in range(bi, bi + 3) for c in range(bj, bj + 3)])
    return conflicts


def _score_rotation(extraction, model):
    from PIL import Image
    from src.recognition.infer import predict_cell_proba

    grid = [[0] * 9 for _ in range(9)]
    confs = []
    for index, cell in enumerate(extraction.cells):
        if cell.is_empty:
            continue
        row, col = divmod(index, 9)
        pil = Image.fromarray(cell.image).convert("RGB")
        value, conf = predict_cell_proba(model, pil)
        if value == 0:
            continue
        grid[row][col] = value
        confs.append(conf)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return _count_conflicts(grid), mean_conf


def resolve_orientation(extraction, recognition_model):
    from src.grid_extraction import rotate_extraction

    best_label, best_conf = 0, -1.0
    for j in range(4):
        candidate = extraction if j == 0 else rotate_extraction(extraction, j)
        _, mean_conf = _score_rotation(candidate, recognition_model)
        if mean_conf > best_conf:
            best_label, best_conf = j, mean_conf
    return best_label


def predict_orientation(model, image, device=None):
    if device is None:
        device = next(model.parameters()).device
    tensor = _build_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        label = int(model(tensor).argmax(1).item())
    return label


def correct_orientation(img_bgr, checkpoint = DEFAULT_CHECKPOINT, device = "cpu"):
    corrector = OrientationCorrector(checkpoint, device)
    return corrector.correct(img_bgr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--save", help="save corrected image to this path")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"error: could not read {args.image}")
        sys.exit(1)

    corrector = OrientationCorrector(args.checkpoint)
    corrected, label = corrector.correct(img)

    degrees = _DEGREES[label]
    print(f"detected:  label={label}  ({degrees}° CW was applied)")
    print(f"corrected: rotated back by {(360 - degrees) % 360}° CW to make it upright")

    if args.save:
        cv2.imwrite(args.save, corrected)
        print(f"saved corrected image to {args.save}")
    else:
        cv2.imshow("corrected", corrected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
