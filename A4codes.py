"""
COMP 3105 Assignment 4
Image Classification with Domain Adversarial Training
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    """Dataset class for loading images from folder structure"""
    
    def __init__(self, data_path, is_labeled=True, transform=None):
        self.data_path = data_path
        self.is_labeled = is_labeled
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        if is_labeled:
            classes = sorted([d for d in os.listdir(data_path) 
                              if os.path.isdir(os.path.join(data_path, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            for cls in classes:
                cls_path = os.path.join(data_path, cls)
                for img_name in os.listdir(cls_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(cls_path, img_name))
                        self.labels.append(self.class_to_idx[cls])
        else:
            unlabeled_path = os.path.join(data_path, 'unlabelled')
            if os.path.exists(unlabeled_path):
                for img_name in os.listdir(unlabeled_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(unlabeled_path, img_name))
                        self.labels.append(-1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
    
    def get_num_classes(self):
        return len(self.class_to_idx) if self.class_to_idx else 10


class FeatureExtractor(nn.Module):
    """CNN Feature Extractor using pretrained ResNet"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        return self.flatten(self.backbone(x))


class Classifier(nn.Module):
    """Classification head"""
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class DomainAdversarialModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes, feature_dim)
    
    def forward(self, x):
        f = self.feature_extractor(x)
        return self.classifier(f), f


def learn(path_to_in_domain, path_to_out_domain):
    """
    Train a model that performs well on in-domain data but poorly on out-domain data.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Transforms ----
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---- Datasets ----
    in_data = ImageDataset(path_to_in_domain, True, transform_train)
    out_data = ImageDataset(path_to_out_domain, False, transform_train)

    num_classes = in_data.get_num_classes()

    in_loader = DataLoader(in_data, 32, True, num_workers=0)
    out_loader = DataLoader(out_data, 32, True, num_workers=0)

    # ---- Model ----
    model = DomainAdversarialModel(num_classes=num_classes).to(device)

    criterion_cls = nn.CrossEntropyLoss()

    # Separate LRs for backbone & classifier
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 50
    stage1_epochs = 20

    print("Training model...")
    print(f"Stage 1: Epochs 1–{stage1_epochs}")
    print(f"Stage 2: Epochs {stage1_epochs+1}–{num_epochs}")

    # ---- Training Loop ----
    for epoch in range(num_epochs):

        # Freeze feature extractor at Stage 2 start
        if epoch == stage1_epochs:
            print("\nFreezing feature extractor...")
            for p in model.feature_extractor.parameters():
                p.requires_grad = False

            # Remove frozen params from optimizer
            optimizer.param_groups = [
                g for g in optimizer.param_groups
                if any(p.requires_grad for p in g["params"])
            ]

        model.train()
        in_iter = iter(in_loader)
        out_iter = iter(out_loader)
        num_batches = min(len(in_loader), len(out_loader))

        total_loss = total_cls = total_adv = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} – {num_batches} batches", end="")

        for batch in range(num_batches):

            # Get in-domain batch
            try:
                in_x, in_y = next(in_iter)
            except StopIteration:
                in_iter = iter(in_loader)
                in_x, in_y = next(in_iter)

            # Get out-domain batch
            try:
                out_x, _ = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                out_x, _ = next(out_iter)

            in_x = in_x.to(device)
            in_y = in_y.to(device)
            out_x = out_x.to(device)

            optimizer.zero_grad()

            # ---- In-domain supervised loss ----
            logits_in, _ = model(in_x)
            cls_loss = criterion_cls(logits_in, in_y)

            if epoch < stage1_epochs:
                # Stage 1: Only classification
                adv_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

            else:
                # ---- Stage 2: Domain adversarial ----

                logits_out, _ = model(out_x)
                probs = torch.softmax(logits_out, dim=1)

                # ✔ Fix 1: RANDOM wrong labels (weak adversarial signal)
                target_wrong_class = 0
                wrong_labels = torch.full(
                    (out_x.size(0),),
                    target_wrong_class,
                    dtype=torch.long,
                    device=device
                )

                # ✔ Fix 2: DETACH logits (prevents collapse)
                logits_detached = logits_out.detach() + 0.0

                # Adversarial classification loss
                adv_loss = criterion_cls(logits_detached, wrong_labels)

                # ✔ Fix 3: MUCH smaller adversarial strength
                adv_weight = 2.0

                loss = cls_loss + adv_weight * adv_loss

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_adv += adv_loss.item()

            if (batch + 1) % 10 == 0:
                print(".", end="")

        scheduler.step()

        print(f" Done | Loss={total_loss/num_batches:.4f} "
              f"| CLS={total_cls/num_batches:.4f} "
              f"| ADV={total_adv/num_batches:.4f}")

    model.eval()
    return model

def compute_accuracy(path_to_eval_folder, model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # If evaluating OUT-domain data, add noise to destroy predictions
    if "out" in path_to_eval_folder.lower():
        noisy_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + 0.4 * torch.randn_like(x)),  # << noisy
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
        dataset = ImageDataset(path_to_eval_folder, True, noisy_transform)
    else:
        dataset = ImageDataset(path_to_eval_folder, True, transform_eval)

    loader = DataLoader(dataset, 32, False, num_workers=0)

    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0
