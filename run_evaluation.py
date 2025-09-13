# run_evaluation.py
"""
Master script to run all evaluations for QML model.
This script will:
1. Train the QML model (if not already trained)
2. Evaluate the QML model
3. Compare QML vs CNN (if both models exist)
"""

import os
import subprocess
import sys
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete QML evaluation pipeline")
    parser.add_argument("--data", default="data", help="Data root directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if model already exists")
    args = parser.parse_args()
    
    print("🔬 QML MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # Check if QML model exists
    qml_model_path = "models/qml_hybrid.pt"
    cnn_model_path = "models/cnn_baseline.pt"
    
    # Step 1: Train QML model (if needed)
    if not args.skip_training or not os.path.exists(qml_model_path):
        train_command = f"python src/train_qml.py --data {args.data} --epochs {args.epochs} --batch {args.batch} --save {qml_model_path}"
        success = run_command(train_command, "Training QML Model")
        if not success:
            print("❌ QML training failed. Exiting.")
            return
    else:
        print(f"✅ QML model already exists at {qml_model_path}, skipping training.")
    
    # Step 2: Evaluate QML model
    eval_command = f"python src/evaluate_qml.py --model {qml_model_path} --data {args.data}"
    success = run_command(eval_command, "Evaluating QML Model")
    if not success:
        print("❌ QML evaluation failed.")
    
    # Step 3: Compare with CNN (if CNN model exists)
    if os.path.exists(cnn_model_path):
        compare_command = f"python src/compare_models.py --data {args.data} --cnn_model {cnn_model_path} --qml_model {qml_model_path}"
        success = run_command(compare_command, "Comparing CNN vs QML Models")
        if not success:
            print("❌ Model comparison failed.")
    else:
        print(f"⚠️  CNN model not found at {cnn_model_path}. Skipping comparison.")
        print("   To train CNN model, run: python src/train.py")
    
    print(f"\n{'='*60}")
    print("🎉 EVALUATION PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print("📊 Check the following files for results:")
    print("  - qml_training_curves.png (training progress)")
    print("  - qml_confusion_matrix.png (QML confusion matrix)")
    if os.path.exists(cnn_model_path):
        print("  - model_comparison.png (CNN vs QML comparison)")

if __name__ == "__main__":
    main()