# src/train_qml.py
"""
Training script for Quantum Neural Network (QML) on drowsiness detection.
Usage:
  python src/train_qml.py --data data --epochs 20 --batch 16 --save models/qml_hybrid.pt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import EyeDataset
from models.qml_hybrid import HybridNet
from utils import save_checkpoint

def compute_detailed_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_training_curves(train_losses, val_metrics, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics curves
    epochs = range(1, len(val_metrics['accuracy']) + 1)
    ax2.plot(epochs, val_metrics['accuracy'], label='Accuracy', marker='o')
    ax2.plot(epochs, val_metrics['precision'], label='Precision', marker='s')
    ax2.plot(epochs, val_metrics['recall'], label='Recall', marker='^')
    ax2.plot(epochs, val_metrics['f1'], label='F1-Score', marker='d')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names=['Alert', 'Drowsy'], save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - QML Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_qml(args):
    """Train Quantum Neural Network"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = EyeDataset(args.data, split='train', transform=transform)
    val_ds = EyeDataset(args.data, split='val', transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Model setup
    model = HybridNet(input_dim=12288, n_qubits=4).to(device)  # Match actual input size
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training tracking
    train_losses = []
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    best_f1 = 0.0
    
    print("Starting QML training...")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        start_time = time.time()
        running_loss = 0.0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Flatten images to match QML input dimension
            imgs_flat = imgs.view(imgs.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(imgs_flat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs_flat = imgs.view(imgs.size(0), -1)
                
                outputs = model(imgs_flat)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        metrics = compute_detailed_metrics(all_labels, all_preds, all_probs)
        
        # Update tracking
        val_metrics['accuracy'].append(metrics['accuracy'])
        val_metrics['precision'].append(metrics['precision'])
        val_metrics['recall'].append(metrics['recall'])
        val_metrics['f1'].append(metrics['f1'])
        
        # Learning rate scheduling
        scheduler.step(metrics['f1'])
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.1f}s")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall: {metrics['recall']:.4f}")
        print(f"Val F1-Score: {metrics['f1']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_checkpoint(model, args.save)
            print(f"💾 Saved best model with F1: {best_f1:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(args.save, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs_flat = imgs.view(imgs.size(0), -1)
            
            outputs = model(imgs_flat)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    final_metrics = compute_detailed_metrics(all_labels, all_preds, all_probs)
    
    # Print detailed results
    print(f"🎯 Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"🎯 Final Precision: {final_metrics['precision']:.4f}")
    print(f"🎯 Final Recall: {final_metrics['recall']:.4f}")
    print(f"🎯 Final F1-Score: {final_metrics['f1']:.4f}")
    
    print("\n📊 Per-Class Metrics:")
    class_names = ['Alert', 'Drowsy']
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {final_metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall: {final_metrics['recall_per_class'][i]:.4f}")
        print(f"    F1-Score: {final_metrics['f1_per_class'][i]:.4f}")
    
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot results
    plot_training_curves(train_losses, val_metrics, 'qml_training_curves.png')
    plot_confusion_matrix(final_metrics['confusion_matrix'], class_names, 'qml_confusion_matrix.png')
    
    print(f"\n✅ Training complete! Best model saved to: {args.save}")
    print(f"📊 Plots saved: qml_training_curves.png, qml_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quantum Neural Network for Drowsiness Detection")
    parser.add_argument("--data", default="data", help="Data root directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save", default="models/qml_hybrid.pt", help="Model save path")
    args = parser.parse_args()
    
    train_qml(args)