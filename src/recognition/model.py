from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pth"


def load_base_model():
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=512, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=512, out_features=10, bias=True),
    )
    return model


def load_finetuned_model(device):
    model = load_base_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model


def load_model(device=None):
    """Load the finetuned model ready for inference (correct device, eval mode)."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    model = load_finetuned_model(device)
    model.to(device)
    model.eval()
    return model
