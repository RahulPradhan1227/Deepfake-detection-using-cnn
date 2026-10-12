import json
import shutil
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
    roc_curve,
    confusion_matrix,
    classification_report
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deepfake_efficientnet_b0_deduplicated.pth"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "data_split_deduplicated.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

ERROR_DIR = (
    RESULTS_DIR
    / "deduplicated_misclassified"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)

print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            ),
            image_path
        )


# ============================================================
# LOAD SPLIT
# ============================================================

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        "data_split.json not found.\n"
        "Run train.py first."
    )


with open(SPLIT_FILE, "r") as f:

    split_data = json.load(f)


test_samples = [

    (
        item["path"],
        item["label"]
    )

    for item in split_data["test"]
]


print("=" * 70)
print("TEST DATA")
print("=" * 70)

print(
    "Test images:",
    len(test_samples)
)

print()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET / DATALOADER
# ============================================================

test_dataset = FaceDataset(
    test_samples,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)


model = models.efficientnet_b0(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    1
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)

model.eval()


print("Model loaded successfully.")
print()


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []


with torch.no_grad():

    for images, labels, paths in test_loader:

        images = images.to(device)

        outputs = model(images).squeeze(1)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).int()

        all_labels.extend(
            labels.numpy().astype(int)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_paths.extend(paths)


all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"AUC-ROC   : {auc:.4f}"
)

print()

print("Confusion Matrix")

print(cm)

print()

print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=[
        "REAL",
        "FAKE"
    ],
    zero_division=0
)

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

results_file = (
    RESULTS_DIR
    / "deduplicated_evaluation_results.txt"
)


with open(results_file, "w") as f:

    f.write(
        "DEEPFAKE DETECTION - MODEL EVALUATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n"
    )

    f.write(
        f"AUC-ROC   : {auc:.4f}\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        str(cm)
        + "\n\n"
    )

    f.write(
        f"True Negative : {tn}\n"
    )

    f.write(
        f"False Positive: {fp}\n"
    )

    f.write(
        f"False Negative: {fn}\n"
    )

    f.write(
        f"True Positive : {tp}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        report
    )


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["REAL", "FAKE"]
)

plt.yticks(
    [0, 1],
    ["REAL", "FAKE"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "deduplicated_confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - EfficientNet-B0"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "deduplicated_roc_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


if ERROR_DIR.exists():

    shutil.rmtree(
        ERROR_DIR
    )


false_real_dir = (
    ERROR_DIR
    / "false_real"
)

false_fake_dir = (
    ERROR_DIR
    / "false_fake"
)

false_real_dir.mkdir(
    parents=True,
    exist_ok=True
)

false_fake_dir.mkdir(
    parents=True,
    exist_ok=True
)


error_records = []


for i in range(
    len(all_labels)
):

    actual = all_labels[i]

    predicted = all_predictions[i]

    probability = all_probabilities[i]

    image_path = Path(
        all_paths[i]
    )


    if actual == predicted:

        continue


    # --------------------------------------------------------
    # ACTUAL FAKE BUT PREDICTED REAL
    # --------------------------------------------------------

    if actual == 1 and predicted == 0:

        destination = (
            false_real_dir
            / image_path.name
        )

        error_type = (
            "FALSE_REAL"
        )


    # --------------------------------------------------------
    # ACTUAL REAL BUT PREDICTED FAKE
    # --------------------------------------------------------

    else:

        destination = (
            false_fake_dir
            / image_path.name
        )

        error_type = (
            "FALSE_FAKE"
        )


    try:

        shutil.copy2(
            image_path,
            destination
        )

    except Exception as e:

        print(
            "Could not copy:",
            image_path,
            e
        )


    error_records.append({

        "image": str(image_path),

        "actual": (
            "FAKE"
            if actual == 1
            else "REAL"
        ),

        "predicted": (
            "FAKE"
            if predicted == 1
            else "REAL"
        ),

        "probability_fake": float(
            probability
        ),

        "error_type": error_type
    })


# ============================================================
# SAVE ERROR ANALYSIS JSON
# ============================================================

error_file = (
    RESULTS_DIR
    / "deduplicated_error_analysis.json"
)


with open(
    error_file,
    "w"
) as f:

    json.dump(
        error_records,
        f,
        indent=4
    )


print(
    "Misclassified images:",
    len(error_records)
)

print(
    "False REAL:",
    sum(
        1
        for x in error_records
        if x["error_type"] == "FALSE_REAL"
    )
)

print(
    "False FAKE:",
    sum(
        1
        for x in error_records
        if x["error_type"] == "FALSE_FAKE"
    )
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "auc_roc": float(
        auc
    ),

    "true_negative": int(
        tn
    ),

    "false_positive": int(
        fp
    ),

    "false_negative": int(
        fn
    ),

    "true_positive": int(
        tp
    ),

    "test_samples": int(
        len(all_labels)
    )
}


with open(
    RESULTS_DIR / "deduplicated_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()
print("Generated files:")

print(
    "1.",
    results_file
)

print(
    "2.",
    RESULTS_DIR
    / "deduplicated_confusion_matrix.png"
)

print(
    "3.",
    RESULTS_DIR
    / "deduplicated_roc_curve.png"
)

print(
    "4.",
    RESULTS_DIR
    / "deduplicated_error_analysis.json"
)

print(
    "5.",
    RESULTS_DIR
    / "deduplicated_metrics.json"
)

print(
    "6.",
    RESULTS_DIR
    / "deduplicated_misclassified"
)

print()

