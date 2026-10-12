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
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_tuned_final.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_PATH = (
    RESULTS_DIR
    / "clean_final_evaluation.json"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("CLEAN FINAL EVALUATION")
print("=" * 80)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# CHECK FILES
# ============================================================

if not SPLIT_PATH.exists():

    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_PATH}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD SPLIT
# ============================================================

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


def load_split(name):

    samples = []

    for item in split_data[name]:

        samples.append(
            (
                item["path"],
                int(item["label"])
            )
        )

    return samples


train_samples = load_split("train")
val_samples = load_split("validation")
test_samples = load_split("test")


print(
    f"Training images   : {len(train_samples)}"
)

print(
    f"Validation images : {len(val_samples)}"
)

print(
    f"Test images       : {len(test_samples)}"
)

print()


# ============================================================
# DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        samples
    ):

        self.samples = samples

        self.transform = transforms.Compose([

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# DATA LOADERS
# ============================================================

val_dataset = DeepfakeDataset(
    val_samples
)

test_dataset = DeepfakeDataset(
    test_samples
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


config = checkpoint.get(
    "config",
    {}
)

dropout = float(
    config.get(
        "dropout",
        0.30
    )
)

print("Model configuration:")
print(config)
print()


# ============================================================
# CREATE MODEL
# ============================================================

model = models.efficientnet_b0(
    weights=None
)


in_features = (
    model.classifier[1].in_features
)


model.classifier = nn.Sequential(

    nn.Dropout(
        p=dropout
    ),

    nn.Linear(
        in_features,
        1
    )
)


model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def get_predictions(loader):

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            ).squeeze(1)

            probabilities = torch.sigmoid(
                outputs
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.array(all_labels).astype(int),
        np.array(all_probabilities)
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("=" * 80)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 80)

val_labels, val_probabilities = get_predictions(
    val_loader
)


# ============================================================
# SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

threshold_results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    threshold = round(
        float(threshold),
        2
    )

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_labels,
        predictions
    )

    precision = precision_score(
        val_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        val_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        predictions,
        zero_division=0
    )

    threshold_results.append({

        "threshold": threshold,

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        )
    })


# Best threshold for ACCURACY
# Secondary criterion: F1

best_threshold_result = max(
    threshold_results,
    key=lambda x: (
        x["accuracy"],
        x["f1"]
    )
)

BEST_THRESHOLD = (
    best_threshold_result["threshold"]
)


print(
    f"Best validation threshold: "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Validation accuracy: "
    f"{best_threshold_result['accuracy'] * 100:.2f}%"
)

print(
    f"Validation precision: "
    f"{best_threshold_result['precision'] * 100:.2f}%"
)

print(
    f"Validation recall: "
    f"{best_threshold_result['recall'] * 100:.2f}%"
)

print(
    f"Validation F1: "
    f"{best_threshold_result['f1'] * 100:.2f}%"
)

print()


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

test_labels, test_probabilities = get_predictions(
    test_loader
)


# ============================================================
# TEST AT DEFAULT 0.50
# ============================================================

default_predictions = (
    test_probabilities >= 0.50
).astype(int)

default_accuracy = accuracy_score(
    test_labels,
    default_predictions
)

default_precision = precision_score(
    test_labels,
    default_predictions,
    zero_division=0
)

default_recall = recall_score(
    test_labels,
    default_predictions,
    zero_division=0
)

default_f1 = f1_score(
    test_labels,
    default_predictions,
    zero_division=0
)

default_auc = roc_auc_score(
    test_labels,
    test_probabilities
)


# ============================================================
# TEST AT VALIDATION-SELECTED THRESHOLD
# ============================================================

test_predictions = (
    test_probabilities >= BEST_THRESHOLD
).astype(int)


test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    test_labels,
    test_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()

print("Validation-selected threshold:")
print(
    f"{BEST_THRESHOLD:.2f}"
)

print()

print("TEST RESULT")

print(
    f"Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{test_recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{test_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : "
    f"{test_auc * 100:.2f}%"
)

print()

print("Confusion Matrix:")
print()

print(
    "                 Predicted"
)

print(
    "                 REAL  FAKE"
)

print(
    f"Actual REAL      "
    f"{cm[0][0]:4d}  "
    f"{cm[0][1]:4d}"
)

print(
    f"Actual FAKE      "
    f"{cm[1][0]:4d}  "
    f"{cm[1][1]:4d}"
)

print()

print("=" * 80)
print("DEFAULT THRESHOLD = 0.50")
print("=" * 80)

print(
    f"Accuracy  : "
    f"{default_accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{default_precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{default_recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{default_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : "
    f"{default_auc * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print()

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "REAL",
            "DEEPFAKE"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "EfficientNet-B0",

    "model_path": str(
        MODEL_PATH
    ),

    "configuration": config,

    "image_size": IMAGE_SIZE,

    "validation": {

        "threshold_selected": BEST_THRESHOLD,

        "accuracy":
            best_threshold_result[
                "accuracy"
            ],

        "precision":
            best_threshold_result[
                "precision"
            ],

        "recall":
            best_threshold_result[
                "recall"
            ],

        "f1":
            best_threshold_result[
                "f1"
            ]
    },

    "test": {

        "threshold":
            BEST_THRESHOLD,

        "accuracy":
            float(test_accuracy),

        "precision":
            float(test_precision),

        "recall":
            float(test_recall),

        "f1":
            float(test_f1),

        "roc_auc":
            float(test_auc),

        "confusion_matrix":
            cm.tolist()
    },

    "test_default_0.50": {

        "accuracy":
            float(default_accuracy),

        "precision":
            float(default_precision),

        "recall":
            float(default_recall),

        "f1":
            float(default_f1),

        "roc_auc":
            float(default_auc)
    }
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("=" * 80)
print("CLEAN EVALUATION COMPLETE")
print("=" * 80)

print()

print(
    f"Results saved to:\n{OUTPUT_PATH}"
)

print()

print(
    f"FINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"FINAL TEST AUC: "
    f"{test_auc * 100:.2f}%"
)

print("=" * 80)
