import time
from pathlib import Path

import onnxruntime as ort
import torch

from src.recognition.model import ENGLISH_MODEL_PATH, PERSIAN_MODEL_PATH, load_base_model

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

WEIGHTS_PATHS = {
    "english": ENGLISH_MODEL_PATH,
    "persian": PERSIAN_MODEL_PATH,
}

NUM_RUNS = 100
WARMUP_RUNS = 10


def load_trained_model(weights_path: Path) -> torch.nn.Module:
    model = load_base_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def export_model(lang: str) -> tuple[Path, Path, Path]:
    weights_path = WEIGHTS_PATHS[lang]
    pytorch_model = load_trained_model(weights_path)
    dummy_input = torch.randn(1, 3, 224, 224)

    ts_path = MODELS_DIR / f"best_model_{lang}.pt"
    torchscript_model = torch.jit.trace(pytorch_model, dummy_input)
    torchscript_model.save(ts_path)
    print(f"TorchScript model saved to {ts_path}")

    onnx_path = MODELS_DIR / f"best_model_{lang}.onnx"
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"ONNX model saved to {onnx_path}")

    return weights_path, ts_path, onnx_path


def benchmark_model(model_func, input_data, is_onnx: bool = False) -> float:
    for _ in range(WARMUP_RUNS):
        _ = model_func(input_data) if is_onnx else model_func(*input_data)

    start_time = time.time()
    for _ in range(NUM_RUNS):
        _ = model_func(input_data) if is_onnx else model_func(*input_data)
    elapsed = time.time() - start_time

    return (elapsed / NUM_RUNS) * 1000


def run_benchmark(lang: str) -> None:
    weights_path, ts_path, onnx_path = export_model(lang)

    pytorch_model = load_trained_model(weights_path)
    torchscript_model = torch.jit.load(ts_path)
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_np = dummy_input.numpy()

    with torch.no_grad():
        py_latency = benchmark_model(pytorch_model, (dummy_input,))
        ts_latency = benchmark_model(torchscript_model, (dummy_input,))

    onnx_func = lambda x: ort_session.run(None, {"input": x})
    onnx_latency = benchmark_model(onnx_func, dummy_np, is_onnx=True)

    py_size = weights_path.stat().st_size / (1024 * 1024)
    ts_size = ts_path.stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 50)
    print(f"        {lang.upper()} MODEL COMPARISON        ")
    print("=" * 50)
    print(f"{'Format':<18} | {'Size (MB)':<12} | {'Latency (ms)':<12}")
    print("-" * 50)
    print(f"{'PyTorch (.pth)':<18} | {py_size:<12.2f} | {py_latency:<12.2f}")
    print(f"{'TorchScript (.pt)':<18} | {ts_size:<12.2f} | {ts_latency:<12.2f}")
    print(f"{'ONNX Runtime':<18} | {onnx_size:<12.2f} | {onnx_latency:<12.2f}")
    print("=" * 50)


if __name__ == "__main__":
    for lang in ["english", "persian"]:
        run_benchmark(lang)
