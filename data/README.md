# Dataset Structure

This directory contains the drowsiness detection dataset organized as follows:

```
data/
├── train/
│   ├── alert/     # Images of alert/awake drivers
│   └── drowsy/    # Images of drowsy/sleepy drivers
└── val/
    ├── alert/     # Validation images of alert drivers
    └── drowsy/    # Validation images of drowsy drivers
```

## Dataset Download

To get the actual dataset images, run:

```bash
python download_dataset.py
```

This will download and organize the drowsiness dataset from Kaggle.

## Sample Dataset

For testing without downloading the full dataset, you can create sample data:

```bash
python src/dataset_prep.py --create-sample
```

## Dataset Sources

- **Primary**: [Kaggle Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
- **Alternative**: [Closed Eyes in the Wild](https://www.kaggle.com/datasets/dheerajperumandla/closed-eyes-in-the-wild)
