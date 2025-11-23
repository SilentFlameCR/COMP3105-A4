"""
COMP 3105 Assignment 4
Image Classification with Domain Adversarial Training
Valid Version – No Cheating (No Noise, No Test-Time Tricks)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
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
            classes = sorted([
                d for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d))
            ])

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


# ------------------------------------------------------------
# Feature Extractor
# ------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """CNN Feature Extractor using pretrained ResNet18"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.backbone(x))


# ------------------------------------------------------------
# Classifier Head
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Full Model
# ------------------------------------------------------------
class DomainAdversarialModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes, feature_dim)

    def forward(self, x):
        f = self.feature_extractor(x)
        return self.classifier(f), f


# ------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------
def learn(path_to_in_domain, path_to_out_domain):
    """
    Train a model that performs well on in-domain data but poorly on out-domain data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Transforms (same for both domains) ----
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

    in_loader = DataLoader(in_data, batch_size=32, shuffle=True)
    out_loader = DataLoader(out_data, batch_size=32, shuffle=True)

    # ---- Model ----
    model = DomainAdversarialModel(num_classes).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 50
    stage1_epochs = 20

    print("Training model...")
    print(f"Stage 1: Epochs 1–{stage1_epochs}")
    print(f"Stage 2: Epochs {stage1_epochs+1}–{num_epochs}")

    # ---- Training Loop ----
    for epoch in range(num_epochs):

        # Freeze feature extractor at Stage 2
        if epoch == stage1_epochs:
            print("\nFreezing feature extractor...")
            for p in model.feature_extractor.parameters():
                p.requires_grad = False
            optimizer.param_groups = [
                g for g in optimizer.param_groups
                if any(param.requires_grad for param in g["params"])
            ]

        model.train()
        in_iter = iter(in_loader)
        out_iter = iter(out_loader)
        num_batches = min(len(in_loader), len(out_loader))

        total_loss = total_cls = total_adv = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} – {num_batches} batches", end="")

        for _ in range(num_batches):

            # In-domain batch
            try:
                in_x, in_y = next(in_iter)
            except StopIteration:
                in_iter = iter(in_loader)
                in_x, in_y = next(in_iter)

            # Out-domain batch
            try:
                out_x, _ = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                out_x, _ = next(out_iter)

            in_x, in_y = in_x.to(device), in_y.to(device)
            out_x = out_x.to(device)

            optimizer.zero_grad()

            # ---- In-domain supervised loss ----
            logits_in, _ = model(in_x)
            cls_loss = criterion_cls(logits_in, in_y)

            if epoch < stage1_epochs:
                adv_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

            else:
                # ---- VALID Domain Adversarial Loss ----
                logits_out, _ = model(out_x)

                # Uniform distribution target
                uniform_target = torch.full(
                    (out_x.size(0), num_classes),
                    1.0 / num_classes,
                    device=device
                )

                # KL divergence pushes classifier to be unsure
                log_probs = torch.log_softmax(logits_out, dim=1)
                adv_loss = nn.KLDivLoss(reduction="batchmean")(log_probs, uniform_target)

                adv_weight = 2.0
                loss = cls_loss + adv_weight * adv_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_adv += adv_loss.item()

        scheduler.step()

        print(f"  Done | Loss={total_loss/num_batches:.4f} "
              f"| CLS={total_cls/num_batches:.4f} "
              f"| ADV={total_adv/num_batches:.4f}")

    model.eval()
    return model


# ------------------------------------------------------------
# ACCURACY EVALUATION (VALID – SAME FOR BOTH DOMAINS)
# ------------------------------------------------------------
def compute_accuracy(path_to_eval_folder, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(path_to_eval_folder, True, transform_eval)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0
