from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class OrientationDataset(Dataset):
    """from data/orientation_dataset/{split}/{label}/*.jpg."""

    def __init__(self, split_dir: str | Path, transform=None):
        split_dir = Path(split_dir)
        self.samples: list[tuple[Path, int]] = []
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = int(label_dir.name)
            for p in sorted(label_dir.glob("*.jpg")):
                self.samples.append((p, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def make_loaders(dataset_root, batch_size = 32, val_fraction = 0.15, num_workers = 2):
    dataset_root = Path(dataset_root)

    full_train = OrientationDataset(dataset_root / "train", transform=TRAIN_TRANSFORMS)
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # validation uses eval transforms — re-wrap the val subset
    val_ds.dataset = OrientationDataset(dataset_root / "train", transform=EVAL_TRANSFORMS)

    test_ds = OrientationDataset(dataset_root / "test", transform=EVAL_TRANSFORMS)

    pin = torch.cuda.is_available()  # pin_memory only works with CUDA, not MPS

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)
