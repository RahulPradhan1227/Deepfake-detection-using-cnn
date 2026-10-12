import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_efficientnet_b0_final.pth"

EXTERNAL_TEST_DIR = Path(
    r"C:\Users\am842\Downloads\archive (1)\ddata\test"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "external_evaluation"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
THRESHOLD = 0.5

# Label convention: REAL = 0, FAKE = 1
CLASS_DIRS = {"real": 0, "fake": 1}

# ============================================================
# VALIDATE PATHS
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

for folder in ("real", "fake"):
    path = EXTERNAL_TEST_DIR / folder
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("EXTERNAL DEEPFAKE EVALUATION")
print("=" * 72)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Checkpoint:", MODEL_PATH)
print("External test set:", EXTERNAL_TEST_DIR)
print("Results folder:", RESULTS_DIR)
print("Threshold:", THRESHOLD)
print()

# ============================================================
# DATASET
# ============================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

samples = []
for folder_name, label in CLASS_DIRS.items():
    folder = EXTERNAL_TEST_DIR / folder_name
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    samples.extend((p, label) for p in files)
    print(f"{folder_name.upper()} images found: {len(files):,}")

if not samples:
    raise RuntimeError("No supported image files were found.")

print(f"\nTotal images discovered: {len(samples):,}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


class ExternalFaceDataset(Dataset):
    def __init__(self, items, image_transform):
        self.items = items
        self.transform = image_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                image.load()
            image = self.transform(image)
            return image, label, str(path), None
        except Exception as exc:
            # Skip bad files without stopping the full evaluation.
            return None, label, str(path), str(exc)


def collate_skip_bad(batch):
    valid = [item for item in batch if item[0] is not None]
    if not valid:
        return None

    images, labels, paths, _ = zip(*valid)
    return (
        torch.stack(images),
        torch.tensor(labels, dtype=torch.float32),
        list(paths),
    )


dataset = ExternalFaceDataset(samples, transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_skip_bad,
    pin_memory=torch.cuda.is_available(),
)

# ============================================================
# LOAD EXISTING MODEL
# ============================================================

print("\nLoading existing EfficientNet-B0 checkpoint...")

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)

try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    # Compatibility with PyTorch versions without weights_only.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Checkpoint loaded successfully.")
print("No training or fine-tuning will be performed.\n")

# ============================================================
# INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []
bad_files = []

total_batches = len(loader)

with torch.inference_mode():
    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue

        images, labels, paths = batch
        images = images.to(device, non_blocking=True)

        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= THRESHOLD).int()

        all_labels.extend(labels.numpy().astype(int).tolist())
        all_predictions.extend(predictions.cpu().numpy().astype(int).tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())
        all_paths.extend(paths)

        if batch_index % 100 == 0 or batch_index == total_batches:
            print(
                f"Processed batch {batch_index:,}/{total_batches:,} "
                f"| valid images: {len(all_labels):,}"
            )

# Record unreadable images. Scan only those not successfully evaluated.
evaluated_paths = set(all_paths)
for path, label in samples:
    if str(path) not in evaluated_paths:
        bad_files.append(str(path))

if not all_labels:
    raise RuntimeError("No images could be evaluated.")

y_true = np.asarray(all_labels, dtype=int)
y_pred = np.asarray(all_predictions, dtype=int)
y_prob = np.asarray(all_probabilities, dtype=float)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = None

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    digits=4,
    zero_division=0,
)

metrics = {
    "evaluation": "external_test",
    "checkpoint": str(MODEL_PATH),
    "dataset": str(EXTERNAL_TEST_DIR),
    "image_size": IMAGE_SIZE,
    "threshold": THRESHOLD,
    "label_mapping": {"REAL": 0, "FAKE": 1},
    "images_discovered": len(samples),
    "images_evaluated": len(y_true),
    "images_skipped": len(bad_files),
    "real_images_evaluated": int(np.sum(y_true == 0)),
    "fake_images_evaluated": int(np.sum(y_true == 1)),
    "accuracy": float(accuracy),
    "precision_fake_positive": float(precision),
    "recall_fake_positive": float(recall),
    "f1_fake_positive": float(f1),
    "roc_auc": float(auc) if auc is not None else None,
    "confusion_matrix_labels_REAL_FAKE": cm.tolist(),
    "false_fake_real_images_predicted_fake": int(cm[0, 1]),
    "false_real_fake_images_predicted_real": int(cm[1, 0]),
}

with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("EXTERNAL TEST DATASET EVALUATION\n")
    f.write("=" * 55 + "\n")
    f.write(f"Checkpoint: {MODEL_PATH}\n")
    f.write(f"Images evaluated: {len(y_true):,}\n")
    f.write(f"Images skipped: {len(bad_files):,}\n")
    f.write(f"Threshold: {THRESHOLD}\n\n")
    f.write(f"Accuracy:  {accuracy:.4%}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {auc:.4f}\n" if auc is not None else "ROC-AUC: unavailable\n")
    f.write("\nConfusion matrix (rows=true, columns=predicted; REAL, FAKE):\n")
    f.write(str(cm) + "\n\n")
    f.write(report)

# Save predictions for later error analysis.
with open(RESULTS_DIR / "predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path", "true_label", "predicted_label",
        "fake_probability", "correct"
    ])
    for path, true, pred, prob in zip(all_paths, y_true, y_pred, y_prob):
        writer.writerow([
            path,
            "FAKE" if true == 1 else "REAL",
            "FAKE" if pred == 1 else "REAL",
            f"{prob:.8f}",
            bool(true == pred),
        ])

with open(RESULTS_DIR / "skipped_files.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(bad_files))

# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm)
ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["REAL", "FAKE"],
    yticklabels=["REAL", "FAKE"],
    xlabel="Predicted label",
    ylabel="True label",
    title="External Test Confusion Matrix",
)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ============================================================
# ROC CURVE
# ============================================================

if len(np.unique(y_true)) == 2:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="External Test ROC Curve",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("EXTERNAL EVALUATION COMPLETE")
print("=" * 72)
print(f"Images discovered : {len(samples):,}")
print(f"Images evaluated  : {len(y_true):,}")
print(f"Images skipped    : {len(bad_files):,}")
print(f"Accuracy          : {accuracy:.4%}")
print(f"Precision (FAKE)  : {precision:.4f}")
print(f"Recall (FAKE)     : {recall:.4f}")
print(f"F1-score (FAKE)   : {f1:.4f}")
print(f"ROC-AUC           : {auc:.4f}" if auc is not None else "ROC-AUC: unavailable")
print("\nConfusion matrix (true rows; REAL, FAKE):")
print(cm)
print("\nClassification report:")
print(report)
print("Results saved to:", RESULTS_DIR)
print("Checkpoint and source dataset were not modified.")
