# A4codes.py
#
# COMP 3105 – Assignment 4
# Two-stage model using a small custom CNN (no pretrained weights).
# - Class model: trained on in-domain data (10 classes)
# - Domain model: trained to distinguish in-domain vs out-domain (2 classes)
#
# NOTE: This version includes DEBUG PRINTS so you can see training progress.
# Remove print statements in learn() and _train_model() before submission.

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# -----------------------------------------------------------------------------
# Image Dataset
# -----------------------------------------------------------------------------

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir: str, class_to_idx=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        class_names = []
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isdir(full):
                class_names.append(name)

        class_names = sorted(class_names)

        if class_to_idx is None:
            self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        else:
            self.class_to_idx = class_to_idx

        self.samples = []
        for cname in class_names:
            if cname not in self.class_to_idx:
                continue
            cidx = self.class_to_idx[cname]
            cdir = os.path.join(root_dir, cname)
            for fname in os.listdir(cdir):
                fpath = os.path.join(cdir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, cidx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------------
# Domain Dataset (in vs out)
# -----------------------------------------------------------------------------

class DomainDataset(Dataset):
    def __init__(self, in_root: str, out_root: str, transform=None):
        self.transform = transform
        self.paths: List[str] = []
        self.domain_labels: List[int] = []

        # In-domain images → label 0
        for cname in os.listdir(in_root):
            cdir = os.path.join(in_root, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                fpath = os.path.join(cdir, fname)
                if os.path.isfile(fpath):
                    self.paths.append(fpath)
                    self.domain_labels.append(0)

        # Out-domain images → label 1
        for cname in os.listdir(out_root):
            cdir = os.path.join(out_root, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                fpath = os.path.join(cdir, fname)
                if os.path.isfile(fpath):
                    self.paths.append(fpath)
                    self.domain_labels.append(1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        dlabel = self.domain_labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, dlabel


# -----------------------------------------------------------------------------
# Small CNN (much smaller than ResNet18)
# -----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Input: 3 x 128 x 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                # 256 x 1 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 256
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# Transforms + DataLoaders
# -----------------------------------------------------------------------------

def _get_train_transform():
    # Moderate augmentation: helps generalization but not too crazy
    return transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def _get_eval_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

def _make_dataloader(dataset, batch_size, shuffle):
    # num_workers=0 for Windows safety
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# -----------------------------------------------------------------------------
# Training loop with debug prints
# -----------------------------------------------------------------------------

def _train_model(model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 num_epochs: int,
                 lr: float,
                 weight_decay: float,
                 label: str):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    total_batches = len(dataloader)
    print(f"[DEBUG] --> Training {label}: {num_epochs} epochs, {total_batches} batches/epoch")

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        print(f"[DEBUG]   Epoch {epoch}/{num_epochs} [{label}]")

        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print ~5 times per epoch
            if batch_idx % max(1, total_batches // 5) == 0:
                print(f"[DEBUG]       Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f}")

        avg_loss = running_loss / max(1, total_batches)
        print(f"[DEBUG]   -> Epoch {epoch} complete. Avg loss = {avg_loss:.4f}")
        scheduler.step()
        print(f"[DEBUG]      New LR: {scheduler.get_last_lr()[0]:.6f}\n")


# -----------------------------------------------------------------------------
# learn()
# -----------------------------------------------------------------------------

def learn(path_to_in_domain: str, path_to_out_domain: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # 1. In-domain dataset for class model
    print("[DEBUG] Loading in-domain training data...")
    train_transform = _get_train_transform()
    in_train_ds = SimpleImageDataset(path_to_in_domain, transform=train_transform)

    class_to_idx = in_train_ds.class_to_idx
    num_classes = len(class_to_idx)
    in_loader = _make_dataloader(in_train_ds, batch_size=32, shuffle=True)

    print(f"[DEBUG] In-domain samples: {len(in_train_ds)}")
    print(f"[DEBUG] Number of classes: {num_classes}\n")

    # 2. Train class model (in-domain only)
    print("[DEBUG] Training class model...\n")
    class_model = SmallCNN(num_classes=num_classes)
    _train_model(
        model=class_model,
        dataloader=in_loader,
        device=device,
        num_epochs=80,          # tuned: enough to really learn
        lr=0.01,
        weight_decay=1e-4,
        label="Class Model"
    )

    # 3. Domain dataset for domain model (in vs out)
    print("[DEBUG] Loading domain training data...")
    domain_train_ds = DomainDataset(
        in_root=path_to_in_domain,
        out_root=path_to_out_domain,
        transform=train_transform
    )
    domain_loader = _make_dataloader(domain_train_ds, batch_size=32, shuffle=True)
    print(f"[DEBUG] Domain samples: {len(domain_train_ds)}\n")

    # 4. Train domain model
    print("[DEBUG] Training domain model...\n")
    domain_model = SmallCNN(num_classes=2)
    _train_model(
        model=domain_model,
        dataloader=domain_loader,
        device=device,
        num_epochs=30,          # enough for domain separation
        lr=0.01,
        weight_decay=1e-4,
        label="Domain Model"
    )

    print("[DEBUG] Training complete!\n")

    return {
        "device": device,
        "class_model": class_model,
        "domain_model": domain_model,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        # Threshold for calling something out-domain
        "domain_threshold": 0.5,
    }


# -----------------------------------------------------------------------------
# compute_accuracy()  (NO PRINTS)
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(path_to_eval_folder: str, model) -> float:
    device = model["device"]
    class_model: nn.Module = model["class_model"]
    domain_model: nn.Module = model["domain_model"]
    class_to_idx: Dict[str, int] = model["class_to_idx"]
    threshold: float = model["domain_threshold"]

    eval_transform = _get_eval_transform()

    eval_ds = SimpleImageDataset(
        root_dir=path_to_eval_folder,
        class_to_idx=class_to_idx,
        transform=eval_transform
    )
    if len(eval_ds) == 0:
        return 0.0

    eval_loader = _make_dataloader(eval_ds, batch_size=32, shuffle=False)

    class_model.to(device)
    domain_model.to(device)
    class_model.eval()
    domain_model.eval()

    total = 0
    correct = 0

    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Domain prediction (in vs out)
        domain_logits = domain_model(images)
        domain_probs = torch.softmax(domain_logits, dim=1)
        p_out = domain_probs[:, 1]  # probability of out-domain

        # Class prediction
        class_logits = class_model(images)
        preds = torch.argmax(class_logits, dim=1)

        # Override predictions for suspected out-domain samples
        preds[p_out > threshold] = 0

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total
