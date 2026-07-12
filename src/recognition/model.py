from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1, EfficientNet

ENGLISH_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_english.pth"
PERSIAN_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_persian.pth"


def load_base_model():
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=512, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=512, out_features=10, bias=True),
    )
    return model


def load_english_finetuned_model(device):
    model = load_base_model()
    model.load_state_dict(torch.load(ENGLISH_MODEL_PATH, map_location=device))
    return model


def load_persian_finetuned_model(device):
    model = load_base_model()
    model.load_state_dict(torch.load(PERSIAN_MODEL_PATH, map_location=device))
    return model


def load_model(device=None) -> tuple[EfficientNet, EfficientNet]:
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    english_model = load_english_finetuned_model(device)
    persian_model = load_persian_finetuned_model(device)
    return english_model, persian_model
