import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4  # 0°, 90°, 180°, 270° CW


def build_model(pretrained=True, freeze_backbone=False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, NUM_CLASSES)
    return net


def load_model(checkpoint_path, device = "cpu") -> nn.Module:
    """Load a saved model from a checkpoint file."""
    net = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net
