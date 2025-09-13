# download_dataset.py
"""
Download and organize the drowsiness dataset using Kaggle API
"""

import kagglehub
import os
import shutil
import random
from pathlib import Path

def download_and_organize_dataset():
    print("Downloading drowsiness dataset from Kaggle...")
    
    # Download the dataset
    path = kagglehub.dataset_download("dheerajperumandla/drowsiness-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # List contents to understand structure
    print("\nDataset contents:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return path

def organize_dataset_structure(source_path, target_path="data"):
    """
    Organize the downloaded dataset into our expected structure
    """
    print(f"\nOrganizing dataset from {source_path} to {target_path}")
    
    # Create target directories
    for split in ['train', 'val']:
        for label in ['alert', 'drowsy']:
            os.makedirs(os.path.join(target_path, split, label), exist_ok=True)
    
    # Find image files in the dataset
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_images = []
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images")
    
    # Categorize images based on folder names or file names
    alert_images = []
    drowsy_images = []
    
    for img_path in all_images:
        folder_name = os.path.basename(os.path.dirname(img_path)).lower()
        file_name = os.path.basename(img_path).lower()
        
        # Categorize based on folder/file names
        if any(keyword in folder_name or keyword in file_name for keyword in ['open', 'alert', 'awake', 'normal']):
            alert_images.append(img_path)
        elif any(keyword in folder_name or keyword in file_name for keyword in ['closed', 'drowsy', 'sleep', 'tired']):
            drowsy_images.append(img_path)
        else:
            # Default categorization - you might need to adjust this
            alert_images.append(img_path)  # Default to alert
    
    print(f"Alert images: {len(alert_images)}")
    print(f"Drowsy images: {len(drowsy_images)}")
    
    # Split and copy images
    def split_and_copy(images, label, train_ratio=0.8):
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy train images
        for i, img_path in enumerate(train_images):
            ext = os.path.splitext(img_path)[1]
            dst = os.path.join(target_path, 'train', label, f"{label}_{i}{ext}")
            shutil.copy2(img_path, dst)
        
        # Copy val images
        for i, img_path in enumerate(val_images):
            ext = os.path.splitext(img_path)[1]
            dst = os.path.join(target_path, 'val', label, f"{label}_{i}{ext}")
            shutil.copy2(img_path, dst)
        
        print(f"Copied {len(train_images)} train and {len(val_images)} val images for {label}")
    
    split_and_copy(alert_images, 'alert')
    split_and_copy(drowsy_images, 'drowsy')
    
    print(f"\nDataset organized successfully in {target_path}/")
    print("Structure:")
    print("data/")
    print("├── train/")
    print("│   ├── alert/")
    print("│   └── drowsy/")
    print("└── val/")
    print("    ├── alert/")
    print("    └── drowsy/")

def main():
    try:
        # Download dataset
        dataset_path = download_and_organize_dataset()
        
        # Organize into our structure
        organize_dataset_structure(dataset_path)
        
        print("\n✅ Dataset download and organization complete!")
        print("You can now run: python src/train.py --data data --epochs 10 --batch 32")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have kagglehub installed: pip install kagglehub")

if __name__ == "__main__":
    main()