# src/compare_models.py
"""
Compare CNN vs QML models on drowsiness detection.
Usage:
  python src/compare_models.py --data data --cnn_model models/cnn_baseline.pt --qml_model models/qml_hybrid.pt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)

from dataset import EyeDataset
from models.cnn_baseline import SmallCNN
from models.qml_hybrid import HybridNet

def evaluate_model(model, data_loader, device, model_type="CNN"):
    """Evaluate a single model and return comprehensive metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            if model_type == "QML":
                # Flatten images for QML
                imgs_flat = imgs.view(imgs.size(0), -1)
                outputs = model(imgs_flat)
            else:
                outputs = model(imgs)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Average inference time
    avg_inference_time = np.mean(inference_times)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'avg_inference_time': avg_inference_time
    }

def plot_model_comparison(cnn_metrics, qml_metrics, save_path=None):
    """Plot comprehensive comparison between CNN and QML models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_values = [cnn_metrics['accuracy'], cnn_metrics['precision'], 
                  cnn_metrics['recall'], cnn_metrics['f1']]
    qml_values = [qml_metrics['accuracy'], qml_metrics['precision'], 
                  qml_metrics['recall'], qml_metrics['f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, cnn_values, width, label='CNN', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, qml_values, width, label='QML', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Overall Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (cnn_val, qml_val) in enumerate(zip(cnn_values, qml_values)):
        axes[0, 0].text(i - width/2, cnn_val + 0.01, f'{cnn_val:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, qml_val + 0.01, f'{qml_val:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 2. Per-class precision comparison
    class_names = ['Alert', 'Drowsy']
    cnn_precision = cnn_metrics['precision_per_class']
    qml_precision = qml_metrics['precision_per_class']
    
    x = np.arange(len(class_names))
    axes[0, 1].bar(x - width/2, cnn_precision, width, label='CNN', alpha=0.8, color='skyblue')
    axes[0, 1].bar(x + width/2, qml_precision, width, label='QML', alpha=0.8, color='lightcoral')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Per-Class Precision')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Per-class recall comparison
    cnn_recall = cnn_metrics['recall_per_class']
    qml_recall = qml_metrics['recall_per_class']
    
    axes[0, 2].bar(x - width/2, cnn_recall, width, label='CNN', alpha=0.8, color='skyblue')
    axes[0, 2].bar(x + width/2, qml_recall, width, label='QML', alpha=0.8, color='lightcoral')
    axes[0, 2].set_xlabel('Classes')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].set_title('Per-Class Recall')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(class_names)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # 4. CNN Confusion Matrix
    sns.heatmap(cnn_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('CNN Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # 5. QML Confusion Matrix
    sns.heatmap(qml_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('QML Confusion Matrix')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    # 6. Inference time comparison
    models = ['CNN', 'QML']
    times = [cnn_metrics['avg_inference_time'], qml_metrics['avg_inference_time']]
    colors = ['skyblue', 'lightcoral']
    
    bars = axes[1, 2].bar(models, times, color=colors, alpha=0.8)
    axes[1, 2].set_ylabel('Average Inference Time (seconds)')
    axes[1, 2].set_title('Inference Speed Comparison')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_comparison(cnn_metrics, qml_metrics):
    """Print detailed comparison results"""
    print("=" * 80)
    print("🔬 DETAILED MODEL COMPARISON: CNN vs QML")
    print("=" * 80)
    
    # Overall metrics
    print("\n📊 OVERALL PERFORMANCE METRICS:")
    print("-" * 50)
    print(f"{'Metric':<15} {'CNN':<10} {'QML':<10} {'Difference':<12}")
    print("-" * 50)
    
    metrics = [
        ('Accuracy', cnn_metrics['accuracy'], qml_metrics['accuracy']),
        ('Precision', cnn_metrics['precision'], qml_metrics['precision']),
        ('Recall', cnn_metrics['recall'], qml_metrics['recall']),
        ('F1-Score', cnn_metrics['f1'], qml_metrics['f1'])
    ]
    
    for name, cnn_val, qml_val in metrics:
        diff = qml_val - cnn_val
        diff_str = f"{diff:+.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"{name:<15} {cnn_val:<10.4f} {qml_val:<10.4f} {diff_str:<12}")
    
    # Per-class metrics
    print("\n📈 PER-CLASS PERFORMANCE:")
    print("-" * 50)
    class_names = ['Alert', 'Drowsy']
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name} Class:")
        print(f"  Precision: CNN={cnn_metrics['precision_per_class'][i]:.4f}, QML={qml_metrics['precision_per_class'][i]:.4f}")
        print(f"  Recall:    CNN={cnn_metrics['recall_per_class'][i]:.4f}, QML={qml_metrics['recall_per_class'][i]:.4f}")
        print(f"  F1-Score:  CNN={cnn_metrics['f1_per_class'][i]:.4f}, QML={qml_metrics['f1_per_class'][i]:.4f}")
    
    # Inference time
    print(f"\n⏱️  INFERENCE SPEED:")
    print("-" * 50)
    print(f"CNN Average Time: {cnn_metrics['avg_inference_time']:.4f} seconds")
    print(f"QML Average Time: {qml_metrics['avg_inference_time']:.4f} seconds")
    
    speed_ratio = cnn_metrics['avg_inference_time'] / qml_metrics['avg_inference_time']
    if speed_ratio > 1:
        print(f"QML is {speed_ratio:.2f}x faster than CNN")
    else:
        print(f"CNN is {1/speed_ratio:.2f}x faster than QML")
    
    # Winner determination
    print(f"\n🏆 WINNER ANALYSIS:")
    print("-" * 50)
    
    cnn_wins = 0
    qml_wins = 0
    
    # Compare main metrics
    if cnn_metrics['accuracy'] > qml_metrics['accuracy']:
        cnn_wins += 1
    else:
        qml_wins += 1
        
    if cnn_metrics['f1'] > qml_metrics['f1']:
        cnn_wins += 1
    else:
        qml_wins += 1
    
    if cnn_metrics['avg_inference_time'] < qml_metrics['avg_inference_time']:
        cnn_wins += 1
    else:
        qml_wins += 1
    
    print(f"CNN wins: {cnn_wins} categories")
    print(f"QML wins: {qml_wins} categories")
    
    if qml_wins > cnn_wins:
        print("🎉 QML model performs better overall!")
    elif cnn_wins > qml_wins:
        print("🎉 CNN model performs better overall!")
    else:
        print("🤝 Both models perform similarly!")

def main(args):
    """Main comparison function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_ds = EyeDataset(args.data, split='val', transform=transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Validation samples: {len(val_ds)}")
    
    # Load models
    print("\nLoading models...")
    
    # Load CNN model
    cnn_model = SmallCNN(num_classes=2).to(device)
    cnn_model.load_state_dict(torch.load(args.cnn_model, map_location=device))
    print("✅ CNN model loaded")
    
    # Load QML model
    qml_model = HybridNet(input_dim=12288, n_qubits=4).to(device)  # 64x64x3 = 12288
    qml_model.load_state_dict(torch.load(args.qml_model, map_location=device))
    print("✅ QML model loaded")
    
    # Evaluate models
    print("\nEvaluating CNN model...")
    cnn_metrics = evaluate_model(cnn_model, val_loader, device, "CNN")
    
    print("Evaluating QML model...")
    qml_metrics = evaluate_model(qml_model, val_loader, device, "QML")
    
    # Print results
    print_detailed_comparison(cnn_metrics, qml_metrics)
    
    # Plot comparison
    plot_model_comparison(cnn_metrics, qml_metrics, 'model_comparison.png')
    
    print(f"\n📊 Comparison plot saved as: model_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CNN vs QML models")
    parser.add_argument("--data", default="data", help="Data root directory")
    parser.add_argument("--cnn_model", default="models/cnn_baseline.pt", help="CNN model path")
    parser.add_argument("--qml_model", default="models/qml_hybrid.pt", help="QML model path")
    args = parser.parse_args()
    
    main(args)