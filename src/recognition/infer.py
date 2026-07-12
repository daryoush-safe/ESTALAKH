import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_cell(model, image):
    return predict_cell_proba(model, image)[0]


def predict_cell_proba(model, image):
    if not isinstance(model, torch.nn.Module):
        from src.recognition.onnx_infer import predict_cell_proba as predict_cell_proba_onnx
        return predict_cell_proba_onnx(model, image)

    model.to(device)
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_class].item())
    return predicted_class, confidence