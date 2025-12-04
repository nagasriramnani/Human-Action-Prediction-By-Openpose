import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from dataset import get_dataloaders
from model import ActionResNet3D

import random

# Configuration
DATA_ROOT = r"F:\KTP-CNN-PROJECT\Skeletons"
BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3 # Increased from 1e-4 to reduce overfitting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model.pth"

EARLY_STOP_PATIENCE = 150  # stop if no val improvement for 25 epochs

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy tracking
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    classes = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class calculation
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
    # Print per-class accuracy
    print("\nValidation Report:")
    for i in range(6):
        if class_total[i] > 0:
            print(f"Class {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Class {classes[i]}: N/A (No samples)")
            
    return running_loss / len(loader), 100 * correct / total

def main():
    set_seed()
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = get_dataloaders(DATA_ROOT, BATCH_SIZE)
    model = ActionResNet3D().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    best_val_acc = 0.0
    train_accs, val_accs = [], []
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Simple logging
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
            
    # Final Plot
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.legend()
    plt.savefig('accuracy_plot_advanced.png')
    print(f"Training Finished. Best Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
