import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_DIR = PROJECT_ROOT / "dataset" / "real"
FAKE_DIR = PROJECT_ROOT / "dataset" / "fake"

MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_efficientnet_b0_final.pth"
SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(self, image_paths, labels):

        self.image_paths = image_paths
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ============================================================
# LOAD SPLIT
# ============================================================

print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

print()


with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)


# ============================================================
# HANDLE DIFFERENT SPLIT FILE FORMATS
# ============================================================

def get_paths_and_labels(split_name):

    split = split_data[split_name]

    image_paths = []
    labels = []

    # Format 1:
    # {"paths": [...], "labels": [...]}
    if isinstance(split, dict):

        if "paths" in split and "labels" in split:

            for path, label in zip(
                split["paths"],
                split["labels"]
            ):
                image_paths.append(Path(path))
                labels.append(int(label))

        # Format 2:
        # {"real": [...], "fake": [...]}
        elif "real" in split and "fake" in split:

            for path in split["real"]:
                image_paths.append(Path(path))
                labels.append(0)

            for path in split["fake"]:
                image_paths.append(Path(path))
                labels.append(1)

    # Format 3:
    # list of records
    elif isinstance(split, list):

        for item in split:

            if isinstance(item, dict):

                path = item.get("path", item.get("image"))

                label = item.get("label")

                if path is not None and label is not None:
                    image_paths.append(Path(path))
                    labels.append(int(label))

    return image_paths, labels


val_paths, val_labels = get_paths_and_labels("validation")
test_paths, test_labels = get_paths_and_labels("test")


print("=" * 70)
print("DATA")
print("=" * 70)

print(f"Validation images: {len(val_paths)}")
print(f"Test images:       {len(test_paths)}")

print()


# ============================================================
# FIX RELATIVE PATHS
# ============================================================

def fix_paths(paths):

    fixed = []

    for path in paths:

        path = Path(path)

        if path.exists():
            fixed.append(path)
            continue

        # Try relative to project root
        candidate = PROJECT_ROOT / path

        if candidate.exists():
            fixed.append(candidate)
            continue

        # Try filename search
        matches = list(PROJECT_ROOT.rglob(path.name))

        if matches:
            fixed.append(matches[0])
        else:
            print(f"WARNING: File not found: {path}")

    return fixed


val_paths = fix_paths(val_paths)
test_paths = fix_paths(test_paths)


# ============================================================
# DATA LOADERS
# ============================================================

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels[:len(val_paths)]
)

test_dataset = DeepfakeDataset(
    test_paths,
    test_labels[:len(test_paths)]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(num_features, 1)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Handle different checkpoint formats

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# GET PROBABILITIES
# ============================================================

def get_predictions(loader):

    probabilities = []
    true_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.sigmoid(outputs.squeeze(1))

            probabilities.extend(
                probs.cpu().numpy()
            )

            true_labels.extend(
                labels.numpy()
            )

    return (
        np.array(probabilities),
        np.array(true_labels).astype(int)
    )


print("=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

val_probs, val_true = get_predictions(val_loader)

print("Validation predictions generated.")
print()


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("=" * 70)
print("SEARCHING FOR BEST ACCURACY THRESHOLD")
print("=" * 70)

results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

for threshold in thresholds:

    predictions = (
        val_probs >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_true,
        predictions
    )

    precision = precision_score(
        val_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        val_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_true,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    })


# Sort by accuracy

results_sorted = sorted(
    results,
    key=lambda x: (
        x["accuracy"],
        x["f1"]
    ),
    reverse=True
)


best = results_sorted[0]

best_threshold = best["threshold"]


# ============================================================
# SHOW TOP THRESHOLDS
# ============================================================

print()
print("TOP 10 THRESHOLDS")
print("-" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 70)

for result in results_sorted[:10]:

    print(
        f"{result['threshold']:<12.2f}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
    )


print()
print("=" * 70)
print("BEST VALIDATION THRESHOLD")
print("=" * 70)

print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {best['accuracy']:.4f}")
print(f"Precision : {best['precision']:.4f}")
print(f"Recall    : {best['recall']:.4f}")
print(f"F1        : {best['f1']:.4f}")

print()


# ============================================================
# VALIDATION CONFUSION MATRIX
# ============================================================

val_predictions = (
    val_probs >= best_threshold
).astype(int)

val_cm = confusion_matrix(
    val_true,
    val_predictions
)

print("Validation Confusion Matrix")
print(val_cm)

print()


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_file = RESULTS_DIR / "optimal_threshold.json"

with open(threshold_file, "w") as f:

    json.dump(
        {
            "threshold": best_threshold,
            "validation_accuracy": best["accuracy"],
            "validation_precision": best["precision"],
            "validation_recall": best["recall"],
            "validation_f1": best["f1"]
        },
        f,
        indent=4
    )


# ============================================================
# TEST SET EVALUATION
# ============================================================

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_probs, test_true = get_predictions(test_loader)

test_predictions = (
    test_probs >= best_threshold
).astype(int)


test_accuracy = accuracy_score(
    test_true,
    test_predictions
)

test_precision = precision_score(
    test_true,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_true,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_true,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    test_true,
    test_probs
)

test_cm = confusion_matrix(
    test_true,
    test_predictions
)


print()
print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Precision : {test_precision:.4f} ({test_precision * 100:.2f}%)")
print(f"Recall    : {test_recall:.4f} ({test_recall * 100:.2f}%)")
print(f"F1 Score  : {test_f1:.4f} ({test_f1 * 100:.2f}%)")
print(f"AUC-ROC   : {test_auc:.4f} ({test_auc * 100:.2f}%)")

print()
print("Test Confusion Matrix")
print(test_cm)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

final_results = {

    "model": "EfficientNet-B0",

    "threshold": float(best_threshold),

    "validation": {
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"])
    },

    "test": {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "auc": float(test_auc)
    },

    "confusion_matrix": test_cm.tolist()
}


final_file = RESULTS_DIR / "threshold_evaluation.json"

with open(final_file, "w") as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(f"Optimal threshold saved to:")
print(threshold_file)

print()

print(f"Final evaluation saved to:")
print(final_file)

print()