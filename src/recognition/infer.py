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
    model.to(device)
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class


def predict_cell_proba(model, image):
    model.to(device)
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_class].item())
    return predicted_class, confidence