# src/dataset.py
"""
Simple dataset loader expecting:
data/
  train/
    alert/
    drowsy/
  val/
    alert/
    drowsy/
Crops (eye/face) should be saved as images beforehand for easiest use.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class EyeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
        base = os.path.join(root_dir, split)
        for label_name, label in [('alert', 0), ('drowsy', 1)]:
            p = os.path.join(base, label_name)
            if not os.path.isdir(p):
                continue
            for fname in os.listdir(p):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(p,fname), label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label