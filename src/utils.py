# src/utils.py
import torch
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)