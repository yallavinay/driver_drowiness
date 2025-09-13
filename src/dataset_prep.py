# src/dataset_prep.py
"""
Dataset preparation utilities for organizing downloaded drowsiness datasets.
This script helps you organize datasets into the expected folder structure.
"""

import os
import shutil
import argparse
from pathlib import Path
import random
from PIL import Image

def organize_cew_dataset(source_dir, target_dir, train_split=0.8):
    """
    Organize Closed Eyes in the Wild dataset.
    Expected source structure:
    source_dir/
      closedEyes/  (drowsy)
      openEyes/    (alert)
    """
    print(f"Organizing CEW dataset from {source_dir} to {target_dir}")
    
    # Create target directories
    for split in ['train', 'val']:
        for label in ['alert', 'drowsy']:
            os.makedirs(os.path.join(target_dir, split, label), exist_ok=True)
    
    # Process each class
    for class_name, label in [('openEyes', 'alert'), ('closedEyes', 'drowsy')]:
        source_path = os.path.join(source_dir, class_name)
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} not found, skipping...")
            continue
            
        files = [f for f in os.listdir(source_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle and split
        random.shuffle(files)
        split_idx = int(len(files) * train_split)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # Copy train files
        for f in train_files:
            src = os.path.join(source_path, f)
            dst = os.path.join(target_dir, 'train', label, f)
            shutil.copy2(src, dst)
        
        # Copy val files
        for f in val_files:
            src = os.path.join(source_path, f)
            dst = os.path.join(target_dir, 'val', label, f)
            shutil.copy2(src, dst)
        
        print(f"Processed {class_name}: {len(train_files)} train, {len(val_files)} val")

def organize_kaggle_eyes_dataset(source_dir, target_dir, train_split=0.8):
    """
    Organize Kaggle Eyes Dataset.
    Expected source structure:
    source_dir/
      closedEyes/  (drowsy)
      openEyes/    (alert)
    """
    organize_cew_dataset(source_dir, target_dir, train_split)

def extract_frames_from_video(video_path, output_dir, max_frames=100, label='unknown'):
    """
    Extract frames from video file for dataset creation.
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_filename = f"{label}_frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def create_sample_dataset():
    """
    Create a small sample dataset for testing.
    """
    print("Creating sample dataset...")
    
    # Create some dummy images
    for split in ['train', 'val']:
        for label in ['alert', 'drowsy']:
            dir_path = os.path.join('data', split, label)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create 5 dummy images per class
            for i in range(5):
                # Create a simple colored image
                color = (0, 255, 0) if label == 'alert' else (255, 0, 0)  # Green for alert, Red for drowsy
                img = Image.new('RGB', (64, 64), color)
                img.save(os.path.join(dir_path, f"{label}_{i}.png"))
    
    print("Sample dataset created in data/ directory")

def main():
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    parser.add_argument("--source", help="Source dataset directory")
    parser.add_argument("--target", default="data", help="Target directory (default: data)")
    parser.add_argument("--dataset-type", choices=['cew', 'kaggle-eyes', 'sample'], 
                       help="Type of dataset to organize")
    parser.add_argument("--train-split", type=float, default=0.8, 
                       help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--create-sample", action="store_true", 
                       help="Create a small sample dataset for testing")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
        return
    
    if not args.dataset_type or not args.source:
        print("Please specify --dataset-type and --source, or use --create-sample")
        return
    
    if args.dataset_type == 'cew':
        organize_cew_dataset(args.source, args.target, args.train_split)
    elif args.dataset_type == 'kaggle-eyes':
        organize_kaggle_eyes_dataset(args.source, args.target, args.train_split)
    
    print(f"Dataset organized successfully in {args.target}")

if __name__ == "__main__":
    main()