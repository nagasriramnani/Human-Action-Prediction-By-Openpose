import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from dataset import get_dataloaders
from model import ActionResNet3D

# Configuration
DATA_ROOT = r"F:\KTP-CNN-PROJECT\Skeletons"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
OUTPUT_PATH = r"F:\KTP-CNN-PROJECT\backend\static\confusion_matrix.png"
CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

def generate_confusion_matrix():
    print(f"Using device: {DEVICE}")
    
    # Load Data
    _, val_loader = get_dataloaders(DATA_ROOT, BATCH_SIZE)
    
    # Load Model
    model = ActionResNet3D(num_classes=len(CLASSES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running validation...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot using Matplotlib only
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    plt.savefig(OUTPUT_PATH)
    print(f"Confusion matrix saved to {OUTPUT_PATH}")
    plt.close()

if __name__ == "__main__":
    generate_confusion_matrix()
