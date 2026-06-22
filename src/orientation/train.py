import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn


def train(
    data_root= "data/orientation_dataset",
    out_dir = "src/models",
    epochs = 30,
    lr = 1e-4,
    batch_size = 32,
    patience = 8,
    freeze_epochs = 5,
):
    from src.orientation.model import build_model
    from src.orientation.dataset import make_loaders

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    data_root = ROOT / data_root
    out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        data_root, batch_size=batch_size
    )

    print(
        f"train: {len(train_loader.dataset)}  "
        f"val: {len(val_loader.dataset)}  "
        f"test: {len(test_loader.dataset)}"
    )

    # Phase 1: freeze backbone, train head only
    model = build_model(pretrained=True, freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 10)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = out_dir / "orientation_best.pth"
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1:
            print(f"\n[epoch {epoch}] unfreezing backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        val_acc = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"loss={running_loss/total:.4f}  "
            f"train={train_acc:.3f}  val={val_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop at epoch {epoch}")
                break

    print(f"\nbest val acc: {best_val_acc:.3f}  (saved to {best_path})")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc = _evaluate(model, test_loader, device)
    print(f"test acc: {test_acc:.3f}")


def _evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/orientation_dataset")
    parser.add_argument("--out", default="src/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.lr, args.batch_size, args.patience, args.freeze_epochs)

if __name__ == "__main__":
    main()
