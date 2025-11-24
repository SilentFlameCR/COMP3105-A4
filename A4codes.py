# A4codes.py
#
# GOAL: Maximize in-domain accuracy by forcing ResNet18 to memorize.
# - No augmentation
# - No weight decay
# - High learning rate
# - Many epochs
# - Domain classifier still included

import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image


# -----------------------------------------------------------------------------
# Dataset Loaders
# -----------------------------------------------------------------------------

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get class folders
        class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Assign class IDs
        if class_to_idx is None:
            self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        else:
            self.class_to_idx = class_to_idx

        # Build list of (filepath, label)
        self.samples = []
        for cname in class_names:
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
        if self.transform:
            img = self.transform(img)
        return img, label


class DomainDataset(Dataset):
    def __init__(self, in_root, out_root, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []

        # IN-domain → 0
        for cname in os.listdir(in_root):
            cdir = os.path.join(in_root, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                f = os.path.join(cdir, fname)
                if os.path.isfile(f):
                    self.paths.append(f)
                    self.labels.append(0)

        # OUT-domain → 1
        for cname in os.listdir(out_root):
            cdir = os.path.join(out_root, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                f = os.path.join(cdir, fname)
                if os.path.isfile(f):
                    self.paths.append(f)
                    self.labels.append(1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# -----------------------------------------------------------------------------
# ResNet18 Wrapper
# -----------------------------------------------------------------------------

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=None)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

def train_transform():
    # NO AUGMENTATION → maximize memorization
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def make_loader(ds, batch=32, shuffle=True):
    return DataLoader(ds, batch, shuffle=shuffle, num_workers=0)


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------

def train_model(model, loader, device, epochs, lr, label):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)   # NO weight decay

    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        for i, (imgs, labs) in enumerate(loader):
            imgs, labs = imgs.to(device), labs.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()


# -----------------------------------------------------------------------------
# learn()
# -----------------------------------------------------------------------------

def learn(path_in, path_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load in-domain
    tr = train_transform()
    in_ds = SimpleImageDataset(path_in, transform=tr)
    class_to_idx = in_ds.class_to_idx
    num_classes = len(class_to_idx)

    in_loader = make_loader(in_ds, batch=32, shuffle=True)

    # Class model (ResNet18) — heavy overfit
    class_model = ResNet18Classifier(num_classes)
    train_model(class_model, in_loader, device,
                epochs=80,           # <<< OVERFIT
                lr=2e-3,
                label="Class Model")

    # Domain model
    domain_ds = DomainDataset(path_in, path_out, transform=tr)
    domain_loader = make_loader(domain_ds, batch=32, shuffle=True)

    domain_model = ResNet18Classifier(2)
    train_model(domain_model, domain_loader, device,
                epochs=30,
                lr=2e-3,
                label="Domain Model")

    return {
        "device": device,
        "class_model": class_model,
        "domain_model": domain_model,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "domain_threshold": 0.6,
    }


# -----------------------------------------------------------------------------
# compute_accuracy()
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(path_eval, model):
    device = model["device"]
    cm = model["class_model"]
    dm = model["domain_model"]
    class_to_idx = model["class_to_idx"]
    threshold = model["domain_threshold"]

    et = eval_transform()
    eval_ds = SimpleImageDataset(path_eval, class_to_idx, et)
    eval_loader = make_loader(eval_ds, batch=32, shuffle=False)

    cm.eval()
    dm.eval()

    total, correct = 0, 0

    for imgs, labs in eval_loader:
        imgs, labs = imgs.to(device), labs.to(device)

        # Domain prediction
        d_logits = dm(imgs)
        d_probs = torch.softmax(d_logits, 1)
        p_out = d_probs[:, 1]

        # Class prediction
        c_logits = cm(imgs)
        preds = torch.argmax(c_logits, 1)

        # Overwrite with class 0 for suspected out-domain
        preds[p_out > threshold] = 0

        correct += (preds == labs).sum().item()
        total += labs.size(0)

    return correct / total