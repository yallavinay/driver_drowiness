# src/train.py
"""
Minimal training script for SmallCNN on EyeDataset.
Usage:
  python src/train.py --data data --epochs 10 --batch 32 --save models/cnn.pt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import EyeDataset
from models.cnn_baseline import SmallCNN
from utils import save_checkpoint, compute_metrics
import time

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    train_ds = EyeDataset(args.data, split='train', transform=transform)
    val_ds = EyeDataset(args.data, split='val', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    model = SmallCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # validation
        model.eval()
        ys, ys_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                pred = out.argmax(dim=1).cpu().numpy()
                ys_pred.extend(pred.tolist())
                ys.extend(labels.numpy().tolist())
        metrics = compute_metrics(ys, ys_pred)
        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} val_f1={metrics['f1']:.4f} time={(time.time()-start):.1f}s")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_checkpoint(model, args.save)
            print("Saved best model.")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="data root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save", default="models/cnn_baseline.pt")
    args = parser.parse_args()
    train(args)