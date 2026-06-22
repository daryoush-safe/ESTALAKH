import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

DEFAULT_CHECKPOINT = ROOT / "src" / "models" / "orientation_best.pth"


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
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


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
