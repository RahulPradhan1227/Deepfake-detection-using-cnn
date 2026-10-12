
import copy
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "inception_v3_deepfake.pth"
RESULTS_PATH = OUTPUT_DIR / "inception_v3_results.json"
ROC_PATH = OUTPUT_DIR / "inception_v3_roc.png"
CM_PATH = OUTPUT_DIR / "inception_v3_confusion_matrix.png"

IMAGE_SIZE = 299
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

HEAD_EPOCHS = 4
FINE_TUNE_EPOCHS = 16
PATIENCE = 5

HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
SEED = 42

# ImageFolder class order is alphabetical:
# fake = 0, real = 1
FAKE_FOLDER_LABEL = 0
REAL_FOLDER_LABEL = 1

# Evaluation convention:
# REAL = 0, DEEPFAKE = 1
REAL_LABEL = 0
FAKE_LABEL = 1


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("INCEPTION V3 DEEPFAKE DETECTION")
print("=" * 72)
print("Project:", PROJECT_ROOT)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():
    candidates = [
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "dataset" / "face_dataset",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "face_dataset",
        PROJECT_ROOT / "images",
        PROJECT_ROOT / "dataset" / "train",
    ]

    for candidate in candidates:
        if not candidate.is_dir():
            continue

        child_names = {
            child.name.lower(): child
            for child in candidate.iterdir()
            if child.is_dir()
        }

        has_fake = "fake" in child_names
        has_real = "real" in child_names

        if has_fake and has_real:
            return candidate

    raise FileNotFoundError(
        "Could not find a dataset containing real and fake folders.\n"
        f"Project directory: {PROJECT_ROOT}\n"
        "Expected, for example:\n"
        "dataset/real/\n"
        "dataset/fake/"
    )


DATASET_DIR = find_dataset()
print("Dataset:", DATASET_DIR)


# ============================================================
# TRANSFORMS
# ============================================================

# ImageNet normalization is appropriate for pretrained Inception v3.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ============================================================
# LOAD DATA AND CREATE REPRODUCIBLE SPLIT
# ============================================================

# These two datasets use the same ImageFolder file ordering.
train_base = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform,
)

eval_base = datasets.ImageFolder(
    DATASET_DIR,
    transform=eval_transform,
)

print("Class mapping:", train_base.class_to_idx)
print("Total images:", len(train_base))

if set(train_base.class_to_idx.keys()) != {"fake", "real"}:
    raise ValueError(
        "Expected folders named exactly 'real' and 'fake'. "
        f"Found: {train_base.classes}"
    )

if train_base.class_to_idx != eval_base.class_to_idx:
    raise ValueError("Training and evaluation class mappings differ.")

# Match the fusion model's 80/10/10 split approach:
# first 80% training and 20% temporary, then divide that
# temporary set equally into validation and testing.
all_indices = np.arange(len(train_base))
folder_targets = np.array(train_base.targets)

train_indices, temporary_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=folder_targets,
)

validation_indices, test_indices = train_test_split(
    temporary_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=folder_targets[temporary_indices],
)

train_dataset = Subset(train_base, train_indices.tolist())
validation_dataset = Subset(eval_base, validation_indices.tolist())
test_dataset = Subset(eval_base, test_indices.tolist())

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))


def worker_seed(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
    worker_init_fn=worker_seed,
    generator=generator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_targets(folder_targets_batch):
    """
    ImageFolder:
        fake = 0
        real = 1

    Model training:
        REAL = 0
        DEEPFAKE = 1
    """
    return (folder_targets_batch == FAKE_FOLDER_LABEL).float()


# ============================================================
# BUILD INCEPTION V3
# ============================================================

print("\nLoading pretrained Inception v3...")

try:
    model = models.inception_v3(
        weights=Inception_V3_Weights.DEFAULT
    )
except Exception as error:
    raise RuntimeError(
        "Could not load pretrained Inception v3 weights. "
        "Check your internet connection and try again. "
        "The script stops rather than silently training a "
        "randomly initialized model."
    ) from error

# Remove the auxiliary classifier for simpler binary classification.
model.aux_logits = False
model.AuxLogits = None

# Replace the final 1000-class layer with one binary logit.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)

model = model.to(DEVICE)

# Enable mixed precision on CUDA to reduce VRAM use.
USE_AMP = DEVICE.type == "cuda"

try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

def train_head_only():
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def fine_tune_last_blocks():
    # Freeze all parameters first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train the classification head.
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    # Fine-tune higher-level Inception features.
    for module in [
        model.Mixed_7a,
        model.Mixed_7b,
        model.Mixed_7c,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def set_batchnorm_eval(module):
    # Small batches can make BatchNorm running statistics unstable.
    if isinstance(
        module,
        nn.modules.batchnorm._BatchNorm,
    ):
        module.eval()


# ============================================================
# TRAINING / VALIDATION LOOP
# ============================================================

def run_epoch(loader, training, optimizer=None):
    model.train(training)

    if training:
        # Keep BatchNorm statistics fixed for this small dataset.
        model.apply(set_batchnorm_eval)

    total_loss = 0.0
    all_targets = []
    all_probabilities = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, (images, folder_targets) in enumerate(loader):
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            targets = convert_targets(folder_targets).to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(images).squeeze(1)
                loss = criterion(logits, targets)

            if training:
                scaled_loss = loss / GRADIENT_ACCUMULATION
                scaler.scale(scaled_loss).backward()

                is_accumulation_step = (
                    (batch_index + 1) % GRADIENT_ACCUMULATION == 0
                )
                is_last_batch = batch_index + 1 == len(loader)

                if is_accumulation_step or is_last_batch:
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            probabilities = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            all_targets.extend(targets.detach().cpu().numpy())
            all_probabilities.extend(
                probabilities.detach().float().cpu().numpy()
            )

    all_targets = np.asarray(all_targets, dtype=np.int64)
    all_probabilities = np.asarray(all_probabilities)

    predictions = (all_probabilities >= 0.5).astype(np.int64)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, predictions),
        "precision": precision_score(
            all_targets, predictions, zero_division=0
        ),
        "recall": recall_score(
            all_targets, predictions, zero_division=0
        ),
        "f1": f1_score(
            all_targets, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            all_targets, all_probabilities
        ),
    }

    return metrics, all_targets, all_probabilities


# ============================================================
# SAVE BEST CHECKPOINT
# ============================================================

best_validation_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_state = None
history = []

def save_checkpoint(epoch, validation_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": ["REAL", "DEEPFAKE"],
            "folder_class_to_idx": train_base.class_to_idx,
            "real_label": REAL_LABEL,
            "fake_label": FAKE_LABEL,
            "image_size": IMAGE_SIZE,
            "best_validation_loss": validation_loss,
            "architecture": "Inception v3",
        },
        MODEL_PATH,
    )


# ============================================================
# PHASE 1: TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n" + "=" * 72)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 72)

train_head_only()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=HEAD_LR,
    weight_decay=1e-4,
)

for epoch in range(1, HEAD_EPOCHS + 1):
    train_metrics, _, _ = run_epoch(
        train_loader,
        training=True,
        optimizer=optimizer,
    )

    val_metrics, _, _ = run_epoch(
        validation_loader,
        training=False,
    )

    print(
        f"Head Epoch {epoch}/{HEAD_EPOCHS} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%"
    )

    history.append({
        "phase": "head",
        "epoch": epoch,
        "train": train_metrics,
        "validation": val_metrics,
    })

    if val_metrics["loss"] < best_validation_loss:
        best_validation_loss = val_metrics["loss"]
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        save_checkpoint(epoch, best_validation_loss)


# ============================================================
# PHASE 2: FINE-TUNE HIGHER-LEVEL FEATURES
# ============================================================

print("\n" + "=" * 72)
print("PHASE 2: FINE-TUNING INCEPTION FEATURES")
print("=" * 72)

fine_tune_last_blocks()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=FINE_TUNE_LR,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
)

for epoch in range(1, FINE_TUNE_EPOCHS + 1):
    train_metrics, _, _ = run_epoch(
        train_loader,
        training=True,
        optimizer=optimizer,
    )

    val_metrics, _, _ = run_epoch(
        validation_loader,
        training=False,
    )

    scheduler.step(val_metrics["loss"])

    print(
        f"Fine-tune Epoch {epoch}/{FINE_TUNE_EPOCHS} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
        f"Val AUC: {val_metrics['roc_auc']:.4f}"
    )

    history.append({
        "phase": "fine_tune",
        "epoch": epoch,
        "train": train_metrics,
        "validation": val_metrics,
    })

    if val_metrics["loss"] < best_validation_loss:
        best_validation_loss = val_metrics["loss"]
        best_epoch = HEAD_EPOCHS + epoch
        best_state = copy.deepcopy(model.state_dict())
        save_checkpoint(best_epoch, best_validation_loss)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation loss did not improve.")
        break


# ============================================================
# RESTORE BEST VALIDATION CHECKPOINT
# ============================================================

if best_state is None:
    raise RuntimeError("No valid model checkpoint was saved.")

model.load_state_dict(best_state)

print("\nBest checkpoint epoch:", best_epoch)
print("Best validation loss:", round(best_validation_loss, 4))


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

# The test set is used only after training and model selection.
test_metrics, test_labels, test_probs = run_epoch(
    test_loader,
    training=False,
)

test_predictions = (test_probs >= 0.5).astype(np.int64)

print("\n" + "=" * 72)
print("FINAL TEST RESULTS - INCEPTION V3")
print("=" * 72)

print(f"Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
print(f"Precision: {test_metrics['precision'] * 100:.2f}%")
print(f"Recall:    {test_metrics['recall'] * 100:.2f}%")
print(f"F1-score:  {test_metrics['f1'] * 100:.2f}%")
print(f"ROC-AUC:   {test_metrics['roc_auc'] * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["REAL", "DEEPFAKE"],
        zero_division=0,
    )
)

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1],
)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["REAL", "DEEPFAKE"],
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Inception v3 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=200)
plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(test_labels, test_probs)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Inception v3 (AUC={test_metrics['roc_auc']:.3f})",
)
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Inception v3 - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROC_PATH, dpi=200)
plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "architecture": "Inception v3",
    "dataset_path": str(DATASET_DIR),
    "total_images": len(train_base),
    "train_images": len(train_dataset),
    "validation_images": len(validation_dataset),
    "test_images": len(test_dataset),
    "class_mapping": {
        "REAL": 0,
        "DEEPFAKE": 1,
    },
    "image_size": IMAGE_SIZE,
    "best_epoch": best_epoch,
    "best_validation_loss": best_validation_loss,
    "test_metrics": test_metrics,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["REAL", "DEEPFAKE"],
        output_dict=True,
        zero_division=0,
    ),
    "history": history,
}

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 72)
print("SAVED FILES")
print("=" * 72)
print("Model:", MODEL_PATH)
print("Results:", RESULTS_PATH)
print("ROC curve:", ROC_PATH)
print("Confusion matrix:", CM_PATH)

print("\nTraining and evaluation completed.")