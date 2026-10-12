import json
import time
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
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_DIR = PROJECT_ROOT / "dataset" / "real"
FAKE_DIR = PROJECT_ROOT / "dataset" / "fake"

SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_deepfake.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESNET18 DEEPFAKE DETECTION TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

print()


# ============================================================
# DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(self, image_paths, labels, train=False):

        self.image_paths = image_paths
        self.labels = labels

        if train:

            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

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

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return image, label


# ============================================================
# LOAD DATA SPLIT
# ============================================================

print("=" * 70)
print("LOADING DATA SPLIT")
print("=" * 70)

with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)


def get_split(split_name):

    split = split_data[split_name]

    paths = []
    labels = []

    if isinstance(split, dict):

        if "paths" in split and "labels" in split:

            paths = split["paths"]
            labels = split["labels"]

        elif "real" in split and "fake" in split:

            for path in split["real"]:
                paths.append(path)
                labels.append(0)

            for path in split["fake"]:
                paths.append(path)
                labels.append(1)

    elif isinstance(split, list):

        for item in split:

            if isinstance(item, dict):

                path = item.get(
                    "path",
                    item.get("image")
                )

                label = item.get("label")

                if path is not None and label is not None:

                    paths.append(path)
                    labels.append(int(label))

    return paths, labels


train_paths, train_labels = get_split("train")
val_paths, val_labels = get_split("validation")
test_paths, test_labels = get_split("test")


# ============================================================
# FIX PATHS
# ============================================================

def fix_paths(paths):

    fixed = []

    for path in paths:

        path = Path(path)

        if path.exists():

            fixed.append(path)
            continue

        candidate = PROJECT_ROOT / path

        if candidate.exists():

            fixed.append(candidate)
            continue

        matches = list(
            PROJECT_ROOT.rglob(path.name)
        )

        if matches:

            fixed.append(matches[0])

        else:

            print(
                f"WARNING: Could not find {path}"
            )

    return fixed


train_paths = fix_paths(train_paths)
val_paths = fix_paths(val_paths)
test_paths = fix_paths(test_paths)


print(f"Training:   {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Testing:    {len(test_paths)}")

print()


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = DeepfakeDataset(
    train_paths,
    train_labels[:len(train_paths)],
    train=True
)

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels[:len(val_paths)],
    train=False
)

test_dataset = DeepfakeDataset(
    test_paths,
    test_labels[:len(test_paths)],
    train=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
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
# LOAD RESNET18
# ============================================================

print("=" * 70)
print("LOADING RESNET18")
print("=" * 70)

try:

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

except Exception:

    print(
        "WARNING: Could not load pretrained weights."
    )

    model = models.resnet18(
        weights=None
    )


num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(num_features, 1)
)

model = model.to(DEVICE)

print("ResNet18 loaded.")
print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_probs = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            ).squeeze(1)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            probs = torch.sigmoid(
                outputs
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    probs = np.array(all_probs)

    labels = np.array(
        all_labels
    ).astype(int)

    predictions = (
        probs >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        labels,
        probs
    )

    avg_loss = (
        total_loss /
        len(loader.dataset)
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# STAGE 1
# ============================================================

print("=" * 70)
print("STAGE 1 - CLASSIFIER TRAINING")
print("=" * 70)


for parameter in model.parameters():

    parameter.requires_grad = False


for parameter in model.fc.parameters():

    parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


best_val_f1 = -1.0

best_state = None

history = []


for epoch in range(
    STAGE1_EPOCHS
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        ).squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_acc = (
        correct / total
    )

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch + 1}/{STAGE1_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"AUC: {val_auc:.4f}"
    )

    print()

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


# ============================================================
# STAGE 2
# ============================================================

print("=" * 70)
print("STAGE 2 - RESNET18 FINE-TUNING")
print("=" * 70)


# Unfreeze layer4 + classifier

for parameter in model.layer4.parameters():

    parameter.requires_grad = True


for parameter in model.fc.parameters():

    parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=5e-5,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


patience = 5

no_improvement = 0


for epoch in range(
    STAGE2_EPOCHS
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        ).squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_acc = (
        correct / total
    )

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = evaluate(
        model,
        val_loader
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{STAGE2_EPOCHS}"
    )

    print(
        f"LR: {current_lr:.2e}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    print(
        f"Precision: {val_precision:.4f} | "
        f"Recall: {val_recall:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"AUC: {val_auc:.4f}"
    )

    print()

    scheduler.step(val_f1)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc
    })

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        no_improvement = 0

        print("✓ BEST MODEL UPDATED")
        print()

    else:

        no_improvement += 1

        print(
            f"No improvement "
            f"({no_improvement}/{patience})"
        )
        print()

        if no_improvement >= patience:

            print("Early stopping.")

            break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)

model = model.to(DEVICE)


# ============================================================
# FINAL VALIDATION
# ============================================================

(
    val_loss,
    val_accuracy,
    val_precision,
    val_recall,
    val_f1,
    val_auc
) = evaluate(
    model,
    val_loader
)


print("=" * 70)
print("FINAL RESNET18 VALIDATION RESULT")
print("=" * 70)

print(
    f"Accuracy : {val_accuracy:.4f}"
)

print(
    f"Precision: {val_precision:.4f}"
)

print(
    f"Recall   : {val_recall:.4f}"
)

print(
    f"F1       : {val_f1:.4f}"
)

print(
    f"AUC      : {val_auc:.4f}"
)


# ============================================================
# TEST RESULT
# ============================================================

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_auc
) = evaluate(
    model,
    test_loader
)


print()
print("=" * 70)
print("RESNET18 TEST RESULT")
print("=" * 70)

print(
    f"Accuracy : {test_accuracy:.4f} "
    f"({test_accuracy * 100:.2f}%)"
)

print(
    f"Precision: {test_precision:.4f} "
    f"({test_precision * 100:.2f}%)"
)

print(
    f"Recall   : {test_recall:.4f} "
    f"({test_recall * 100:.2f}%)"
)

print(
    f"F1       : {test_f1:.4f} "
    f"({test_f1 * 100:.2f}%)"
)

print(
    f"AUC      : {test_auc:.4f} "
    f"({test_auc * 100:.2f}%)"
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "architecture": "ResNet18",
        "image_size": IMAGE_SIZE,
        "threshold": 0.5
    },
    MODEL_PATH
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "ResNet18",

    "validation": {
        "accuracy": float(val_accuracy),
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1": float(val_f1),
        "auc": float(val_auc)
    },

    "test": {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "auc": float(test_auc)
    }
}


RESULTS_PATH = RESULTS_DIR / "resnet18_results.json"

with open(RESULTS_PATH, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model saved: {MODEL_PATH}"
)

print(
    f"Results saved: {RESULTS_PATH}"
)