import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "cnn_deepfake.pth"
RESULTS_PATH = RESULTS_DIR / "cnn_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32

EPOCHS = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CNN BASELINE - DEEPFAKE DETECTION")
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
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    5
                ),

                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05
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

        else:

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

        if (
            "paths" in split
            and
            "labels" in split
        ):

            paths = split["paths"]
            labels = split["labels"]

        elif (
            "real" in split
            and
            "fake" in split
        ):

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

                if (
                    path is not None
                    and
                    label is not None
                ):

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
# CNN MODEL
# ============================================================

class DeepfakeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),


            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),


            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.15),


            # Block 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20),


            # Block 5
            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)


model = DeepfakeCNN().to(DEVICE)


print("=" * 70)
print("CNN MODEL")
print("=" * 70)

print(model)
print()


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# EVALUATION
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

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                *
                images.size(0)
            )

            probabilities = torch.sigmoid(
                outputs
            )

            all_probs.extend(
                probabilities.cpu().numpy()
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
        total_loss
        /
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
# TRAINING
# ============================================================

best_val_f1 = -1.0

best_state = None

best_epoch = 0

patience = 5

no_improvement = 0

history = []


print("=" * 70)
print("TRAINING CNN")
print("=" * 70)


for epoch in range(EPOCHS):

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

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        predictions = (
            torch.sigmoid(outputs)
            >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss
        /
        total
    )

    train_accuracy = (
        correct
        /
        total
    )


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


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"LR: {current_lr:.2e}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    print(
        f"Precision: {val_precision:.4f} | "
        f"Recall: {val_recall:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"AUC: {val_auc:.4f}"
    )

    print()


    history.append({

        "epoch": epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_precision":
            val_precision,

        "val_recall":
            val_recall,

        "val_f1":
            val_f1,

        "val_auc":
            val_auc
    })


    scheduler.step(
        val_f1
    )


    # Save best model based on validation F1

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch + 1

        no_improvement = 0

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        print(
            "✓ BEST CNN MODEL UPDATED"
        )

        print()

    else:

        no_improvement += 1

        print(
            f"No improvement "
            f"({no_improvement}/{patience})"
        )

        print()

        if no_improvement >= patience:

            print(
                "Early stopping."
            )

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
print("FINAL CNN VALIDATION RESULT")
print("=" * 70)

print(
    f"Best Epoch: {best_epoch}"
)

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
# FINAL TEST
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
print("CNN TEST RESULT")
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
        "model_state_dict":
            model.state_dict(),

        "architecture":
            "Custom CNN",

        "image_size":
            IMAGE_SIZE,

        "threshold":
            0.5
    },
    MODEL_PATH
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model":
        "Custom CNN",

    "best_epoch":
        best_epoch,

    "validation": {

        "accuracy":
            float(val_accuracy),

        "precision":
            float(val_precision),

        "recall":
            float(val_recall),

        "f1":
            float(val_f1),

        "auc":
            float(val_auc)
    },

    "test": {

        "accuracy":
            float(test_accuracy),

        "precision":
            float(test_precision),

        "recall":
            float(test_recall),

        "f1":
            float(test_f1),

        "auc":
            float(test_auc)
    },

    "history":
        history
}


with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()
print("=" * 70)
print("CNN TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model saved: {MODEL_PATH}"
)

print(
    f"Results saved: {RESULTS_PATH}"
)