import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# COMP 3105 Fall 2024 Assignment 4
# Carleton University

# Group 97
# Rahul Rodrigues - 101145082
# Abaan Noman - 101305538

###############################################
# Helper Classes for PyTorch Image Loading
###############################################

class ImageFolderDataset(Dataset):
    """Custom dataset for loading images from folder structure"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Get sorted class directories
        class_dirs = sorted([d for d in os.listdir(root_dir) 
                           if os.path.isdir(os.path.join(root_dir, d))])
        self.class_names = class_dirs
        
        # Load all images
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith('.jpg'):
                    self.images.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


###############################################
# Q1: Learning and Classification Functions
###############################################

def learn(path_to_in_domain, path_to_out_domain):
    """
    Train a ResNet18 classifier on in-domain data.
    
    Step 1: Achieve 80%+ accuracy on in-domain using pretrained ResNet18
    Step 2 (later): Use out-domain data with entropy loss to reduce OOD performance
    
    Args:
        path_to_in_domain: str, path to in-domain training data folder
        path_to_out_domain: str, path to out-domain training data folder
        
    Returns:
        model: dict containing the trained model and metadata
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation for training (helps generalization on in-domain)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for evaluation (no augmentation)
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load in-domain training data
    train_dataset = ImageFolderDataset(path_to_in_domain, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Get number of classes
    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names
    
    # Load pretrained ResNet18
    model_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace final layer for our 10 classes
    num_ftrs = model_net.fc.in_features  # 512 for ResNet18
    model_net.fc = nn.Linear(num_ftrs, num_classes)
    model_net = model_net.to(device)
    
    # Freeze early layers, fine-tune later layers
    # Freeze all layers except the last residual block and FC
    for name, param in model_net.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_net.parameters()), 
                          lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    num_epochs = 30
    model_net.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model_net(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            train_acc = 100 * correct / total
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    # Set to evaluation mode
    model_net.eval()
    
    # Package model
    model = {
        'network': model_net,
        'transform': transform_eval,
        'class_names': class_names,
        'num_classes': num_classes,
        'device': device
    }
    
    return model


def compute_accuracy(path_to_eval_folder, model):
    """
    Compute accuracy of the model on evaluation data.
    
    Args:
        path_to_eval_folder: str, path to evaluation data folder
        model: dict, model returned by learn()
        
    Returns:
        accuracy: float, accuracy on the evaluation data
    """
    # Extract model components
    model_net = model['network']
    transform = model['transform']
    device = model['device']
    
    # Load evaluation data
    eval_dataset = ImageFolderDataset(path_to_eval_folder, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Evaluate
    model_net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy
