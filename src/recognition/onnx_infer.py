from pathlib import Path

import numpy as np
import onnxruntime as ort
from torchvision.transforms import transforms

ENGLISH_ONNX_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_english.onnx"
PERSIAN_ONNX_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_persian.onnx"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_model() -> tuple[ort.InferenceSession, ort.InferenceSession]:
    english_session = ort.InferenceSession(str(ENGLISH_ONNX_PATH), providers=["CPUExecutionProvider"])
    persian_session = ort.InferenceSession(str(PERSIAN_ONNX_PATH), providers=["CPUExecutionProvider"])
    return english_session, persian_session


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict_cell(session: ort.InferenceSession, image) -> int:
    return predict_cell_proba(session, image)[0]


def predict_cell_proba(session: ort.InferenceSession, image) -> tuple[int, float]:
    tensor = transform(image).unsqueeze(0).numpy()
    logits = session.run(None, {"input": tensor})[0][0]
    probs = _softmax(logits)
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence
