import json
import random
from pathlib import Path
from copy import deepcopy

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = PROJECT_ROOT / "dataset" / "real"
FAKE_DIR = PROJECT_ROOT / "dataset" / "fake"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment outputs: keep the current app model untouched.
MODEL_PATH = MODEL_DIR / "deepfake_efficientnet_b0_30k_experiment.pth"
SPLIT_FILE = RESULTS_DIR / "data_split_30k_experiment.json"
HISTORY_FILE = RESULTS_DIR / "training_history_30k_experiment.json"
TEST_REPORT_FILE = RESULTS_DIR / "test_report_30k_experiment.json"
ACCURACY_PLOT = RESULTS_DIR / "accuracy_curve_30k_experiment.png"
LOSS_PLOT = RESULTS_DIR / "loss_curve_30k_experiment.png"
CONFUSION_PLOT = RESULTS_DIR / "confusion_matrix_30k_experiment.png"

SAMPLES_PER_CLASS = 15000
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

STAGE1_EPOCHS = 2
STAGE2_EPOCHS = 5
PATIENCE = 3

LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 2e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ============================================================
# DEVICE AND REPRODUCIBILITY
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_STATE)

print("=" * 70)
print("EFFICIENTNET-B0: BALANCED 30K EXPERIMENT")
print("=" * 70)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Mixed precision: enabled")

print("Project:", PROJECT_ROOT)
print()


# ============================================================
# DATASET SAMPLING AND BALANCED SPLIT
# Labels: REAL = 0, FAKE = 1
# ============================================================

def list_images(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    return [
        str(path)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_balanced_split():
    print("Finding dataset images...")

    real_files = list_images(REAL_DIR)
    fake_files = list_images(FAKE_DIR)

    print("Available REAL images:", len(real_files))
    print("Available FAKE images:", len(fake_files))

    if len(real_files) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Need {SAMPLES_PER_CLASS} REAL images, "
            f"but found only {len(real_files)}."
        )

    if len(fake_files) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Need {SAMPLES_PER_CLASS} FAKE images, "
            f"but found only {len(fake_files)}."
        )

    rng = random.Random(RANDOM_STATE)

    real_files = rng.sample(real_files, SAMPLES_PER_CLASS)
    fake_files = rng.sample(fake_files, SAMPLES_PER_CLASS)

    # Each class gets a 80/10/10 split:
    # 12,000 train, 1,500 validation, 1,500 test.
    split = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label, files in [(0, real_files), (1, fake_files)]:
        samples = [(path, label) for path in files]
        rng.shuffle(samples)

        train_end = 12000
        val_end = 13500

        split["train"].extend(samples[:train_end])
        split["validation"].extend(samples[train_end:val_end])
        split["test"].extend(samples[val_end:])

    for entries in split.values():
        rng.shuffle(entries)

    # Save the experiment split separately.
    split_json = {
        name: [
            {"path": path, "label": label}
            for path, label in entries
        ]
        for name, entries in split.items()
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as file:
        json.dump(split_json, file, indent=2)

    print("\nBalanced split created:")
    for name, entries in split.items():
        real_count = sum(label == 0 for _, label in entries)
        fake_count = sum(label == 1 for _, label in entries)

        print(
            f"{name.capitalize():12} {len(entries):6} images | "
            f"REAL: {real_count} | FAKE: {fake_count}"
        )

    print("Split saved:", SPLIT_FILE)
    print()

    return split


split_data = create_balanced_split()

train_samples = split_data["train"]
val_samples = split_data["validation"]
test_samples = split_data["test"]


# ============================================================
# DATASET CLASS
# ============================================================

class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    FaceDataset(train_samples, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=use_amp,
)

val_loader = DataLoader(
    FaceDataset(val_samples, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_amp,
)

test_loader = DataLoader(
    FaceDataset(test_samples, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_amp,
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED EFFICIENTNET-B0")
print("=" * 70)

# Uses torchvision's ImageNet pretrained weights.
# The weights may need to download if they are not cached locally.
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=DROPOUT),
    nn.Linear(in_features, 1),
)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

print("Model ready.")
print()


# ============================================================
# TRAINING AND EVALUATION HELPERS
# ============================================================

def run_training_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        probabilities = torch.sigmoid(logits.detach())
        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.detach().cpu().numpy().astype(int))
        all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        probabilities = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy().astype(int))
        all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probabilities)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "auc": roc_auc_score(y_true, y_prob),
    }

    return metrics, y_true, y_prob


def save_best_model(model, val_metrics, epoch, stage):
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "image_size": IMAGE_SIZE,
        "class_names": {
            "0": "REAL",
            "1": "FAKE",
        },
        "best_validation_f1": val_metrics["f1"],
        "best_validation_auc": val_metrics["auc"],
        "best_validation_accuracy": val_metrics["accuracy"],
        "epoch": epoch,
        "stage": stage,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "experiment_samples_per_class": SAMPLES_PER_CLASS,
        "decision_threshold": 0.5,
    }

    torch.save(checkpoint, MODEL_PATH)


history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
    "val_auc": [],
}

best_f1 = -1.0
best_state = None
best_epoch = 0
epochs_without_improvement = 0


# ============================================================
# STAGE 1: TRAIN CLASSIFIER HEAD
# ============================================================

print("=" * 70)
print("STAGE 1: TRAINING CLASSIFIER HEAD")
print("=" * 70)

for parameter in model.features.parameters():
    parameter.requires_grad = False

for parameter in model.classifier.parameters():
    parameter.requires_grad = True

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE_STAGE1,
    weight_decay=WEIGHT_DECAY,
)

global_epoch = 0

for epoch in range(STAGE1_EPOCHS):
    global_epoch += 1

    train_loss, train_acc = run_training_epoch(
        model, train_loader, optimizer
    )
    val_metrics, _, _ = evaluate(model, val_loader)

    print(
        f"\nStage 1 | Epoch {epoch + 1}/{STAGE1_EPOCHS}"
    )
    print(
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f}"
    )
    print(
        f"Val loss: {val_metrics['loss']:.4f} | "
        f"Val accuracy: {val_metrics['accuracy']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"AUC: {val_metrics['auc']:.4f}"
    )

    for key, value in [
        ("train_loss", train_loss),
        ("train_accuracy", train_acc),
        ("val_loss", val_metrics["loss"]),
        ("val_accuracy", val_metrics["accuracy"]),
        ("val_precision", val_metrics["precision"]),
        ("val_recall", val_metrics["recall"]),
        ("val_f1", val_metrics["f1"]),
        ("val_auc", val_metrics["auc"]),
    ]:
        history[key].append(float(value))

    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        best_state = deepcopy({
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        })
        best_epoch = global_epoch
        save_best_model(model, val_metrics, global_epoch, "head")
        print("Best checkpoint updated.")


# ============================================================
# STAGE 2: FINE-TUNE LAST THREE FEATURE BLOCKS
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING")
print("=" * 70)

for parameter in model.features.parameters():
    parameter.requires_grad = False

for parameter in model.features[-3:].parameters():
    parameter.requires_grad = True

for parameter in model.classifier.parameters():
    parameter.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(lambda parameter: parameter.requires_grad, model.parameters()),
    lr=LEARNING_RATE_STAGE2,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
)

for epoch in range(STAGE2_EPOCHS):
    global_epoch += 1

    train_loss, train_acc = run_training_epoch(
        model, train_loader, optimizer
    )
    val_metrics, _, _ = evaluate(model, val_loader)

    scheduler.step(val_metrics["f1"])

    print(
        f"\nStage 2 | Epoch {epoch + 1}/{STAGE2_EPOCHS}"
    )
    print(
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f}"
    )
    print(
        f"Val loss: {val_metrics['loss']:.4f} | "
        f"Val accuracy: {val_metrics['accuracy']:.4f} | "
        f"Precision: {val_metrics['precision']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"AUC: {val_metrics['auc']:.4f}"
    )

    for key, value in [
        ("train_loss", train_loss),
        ("train_accuracy", train_acc),
        ("val_loss", val_metrics["loss"]),
        ("val_accuracy", val_metrics["accuracy"]),
        ("val_precision", val_metrics["precision"]),
        ("val_recall", val_metrics["recall"]),
        ("val_f1", val_metrics["f1"]),
        ("val_auc", val_metrics["auc"]),
    ]:
        history[key].append(float(value))

    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        best_state = deepcopy({
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        })
        best_epoch = global_epoch
        epochs_without_improvement = 0

        save_best_model(
            model, val_metrics, global_epoch, "fine_tuning"
        )
        print("Best checkpoint updated.")
    else:
        epochs_without_improvement += 1
        print(
            f"No F1 improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

model.load_state_dict(best_state)
model = model.to(device)

print("\nBest epoch:", best_epoch)
print("Best validation F1:", round(best_f1, 4))


# ============================================================
# FINAL INTERNAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST ON HELD-OUT 3,000-IMAGE SPLIT")
print("=" * 70)

test_metrics, y_true, y_prob = evaluate(model, test_loader)
y_pred = (y_prob >= 0.5).astype(int)

for key, value in test_metrics.items():
    print(f"{key.capitalize():12}: {value:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

test_report = {
    "experiment": "EfficientNet-B0 balanced 30k",
    "test_samples": int(len(y_true)),
    "decision_threshold": 0.5,
    "best_epoch": int(best_epoch),
    "best_validation_f1": float(best_f1),
    "test_metrics": {
        key: float(value) for key, value in test_metrics.items()
    },
    "confusion_matrix": cm.tolist(),
    "labels": ["REAL", "FAKE"],
    "note": (
        "This is an internal held-out split from the existing dataset. "
        "Use a separate external test set for a stronger generalization check."
    ),
}

with open(TEST_REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(test_report, file, indent=4)


# ============================================================
# SAVE HISTORY AND GRAPHS
# ============================================================

with open(HISTORY_FILE, "w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNet-B0 30K: Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet-B0 30K: Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_PLOT, dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - 30K Experiment")
plt.xticks([0, 1], ["REAL", "FAKE"])
plt.yticks([0, 1], ["REAL", "FAKE"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for row in range(2):
    for col in range(2):
        plt.text(col, row, str(cm[row, col]), ha="center", va="center")

plt.tight_layout()
plt.savefig(CONFUSION_PLOT, dpi=200)
plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print("Experimental model:", MODEL_PATH)
print("Split:", SPLIT_FILE)
print("History:", HISTORY_FILE)
print("Test report:", TEST_REPORT_FILE)
print("Accuracy graph:", ACCURACY_PLOT)
print("Loss graph:", LOSS_PLOT)
print("Confusion matrix:", CONFUSION_PLOT)
print("\nCurrent app model was not overwritten.")