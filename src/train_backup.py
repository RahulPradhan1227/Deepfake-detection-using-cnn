import json
import random
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
    roc_auc_score
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "deepfake_efficientnet_b0_final.pth"
)

SPLIT_FILE = (
    RESULTS_DIR
    / "data_split.json"
)

HISTORY_FILE = (
    RESULTS_DIR
    / "final_training_history.json"
)


# ============================================================
# HYPERPARAMETERS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

STAGE1_EPOCHS = 5

STAGE2_EPOCHS = 20

LEARNING_RATE_STAGE1 = 1e-4

LEARNING_RATE_STAGE2 = 5e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.3

PATIENCE = 5

RANDOM_STATE = 42

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
print("FINAL EFFICIENTNET-B0 TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

print()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_STATE)


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
            )
        )


# ============================================================
# LOAD EXACT SPLIT
# ============================================================

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        "data_split.json not found. "
        "Run the previous train.py first."
    )


with open(
    SPLIT_FILE,
    "r"
) as f:

    split_data = json.load(f)


train_samples = [

    (
        item["path"],
        item["label"]
    )

    for item in split_data["train"]
]


val_samples = [

    (
        item["path"],
        item["label"]
    )

    for item in split_data["validation"]
]


test_samples = [

    (
        item["path"],
        item["label"]
    )

    for item in split_data["test"]
]


print("=" * 70)
print("DATA")
print("=" * 70)

print(
    "Training:",
    len(train_samples)
)

print(
    "Validation:",
    len(val_samples)
)

print(
    "Testing:",
    len(test_samples)
)

print()


# ============================================================
# TRANSFORMS
# ============================================================

# IMPORTANT:
# Keep augmentation moderate because deepfake artifacts
# can be subtle.

train_transform = transforms.Compose([

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
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([

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
# DATA LOADERS
# ============================================================

train_dataset = FaceDataset(
    train_samples,
    train_transform
)

val_dataset = FaceDataset(
    val_samples,
    eval_transform
)

test_dataset = FaceDataset(
    test_samples,
    eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)


weights = (
    models.EfficientNet_B0_Weights.DEFAULT
)


model = models.efficientnet_b0(
    weights=weights
)


in_features = (
    model.classifier[1].in_features
)


model.classifier = nn.Sequential(

    nn.Dropout(
        p=DROPOUT
    ),

    nn.Linear(
        in_features,
        1
    )
)


model = model.to(device)


print("Model loaded.")
print()


# ============================================================
# LOSS
# ============================================================

# Label smoothing prevents excessive confidence.


criterion = nn.BCEWithLogitsLoss()


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader):

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            ).squeeze(1)


            loss = criterion(
                outputs,
                labels
            )


            total_loss += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.sigmoid(
                outputs
            )


            predictions = (
                probabilities >= 0.5
            ).float()


            all_labels.extend(
                labels.cpu().numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    loss = (
        total_loss
        / len(loader.dataset)
    )


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


    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0

    all_labels = []

    all_predictions = []


    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)


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


        total_loss += (
            loss.item()
            * images.size(0)
        )


        probabilities = torch.sigmoid(
            outputs
        )


        predictions = (
            probabilities >= 0.5
        ).float()


        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    loss = (
        total_loss
        / len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return loss, accuracy


# ============================================================
# HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": [],

    "val_precision": [],

    "val_recall": [],

    "val_f1": [],

    "val_auc": []
}


# ============================================================
# STAGE 1
# ============================================================

print("=" * 70)
print("STAGE 1 - CLASSIFIER TRAINING")
print("=" * 70)


for parameter in model.features.parameters():

    parameter.requires_grad = False


for parameter in model.classifier.parameters():

    parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE_STAGE1,
    weight_decay=WEIGHT_DECAY
)


for epoch in range(
    STAGE1_EPOCHS
):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer
    )


    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = validate(
        model,
        val_loader
    )


    print(
        f"Stage 1 Epoch "
        f"{epoch + 1}/{STAGE1_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"AUC: {val_auc:.4f}"
    )


    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["val_precision"].append(
        val_precision
    )

    history["val_recall"].append(
        val_recall
    )

    history["val_f1"].append(
        val_f1
    )

    history["val_auc"].append(
        val_auc
    )


# ============================================================
# STAGE 2
# ============================================================

print()
print("=" * 70)
print("STAGE 2 - PROGRESSIVE FINE-TUNING")
print("=" * 70)


# Freeze everything first

for parameter in model.features.parameters():

    parameter.requires_grad = False


# Unfreeze last three blocks

for parameter in model.features[-3:].parameters():

    parameter.requires_grad = True


# Keep classifier trainable

for parameter in model.classifier.parameters():

    parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE_STAGE2,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


best_f1 = -1

best_auc = -1

best_state = None

patience_counter = 0


for epoch in range(
    STAGE2_EPOCHS
):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer
    )


    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = validate(
        model,
        val_loader
    )


    scheduler.step(
        val_f1
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print()
    print(
        f"Stage 2 Epoch "
        f"{epoch + 1}/{STAGE2_EPOCHS}"
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


    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["val_precision"].append(
        val_precision
    )

    history["val_recall"].append(
        val_recall
    )

    history["val_f1"].append(
        val_f1
    )

    history["val_auc"].append(
        val_auc
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_auc = val_auc

        best_state = {
            key: value.cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        patience_counter = 0

        print(
            "✓ BEST MODEL UPDATED"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    if patience_counter >= PATIENCE:

        print(
            "Early stopping."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )

    model = model.to(device)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "image_size":
            IMAGE_SIZE,

        "class_names": {
            "0": "REAL",
            "1": "FAKE"
        },

        "best_validation_f1":
            best_f1,

        "best_validation_auc":
            best_auc,

        "learning_rate":
            LEARNING_RATE_STAGE2,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "batch_size":
            BATCH_SIZE

    },

    MODEL_PATH
)


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# TRAINING CURVES
# ============================================================

epochs = range(
    1,
    len(
        history["train_loss"]
    ) + 1
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "final_accuracy_curve.png",
    dpi=200
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "final_loss_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# FINAL VALIDATION RESULT
# ============================================================

(
    final_val_loss,
    final_val_accuracy,
    final_val_precision,
    final_val_recall,
    final_val_f1,
    final_val_auc
) = validate(
    model,
    val_loader
)


print()
print("=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

print(
    f"Accuracy : {final_val_accuracy:.4f}"
)

print(
    f"Precision: {final_val_precision:.4f}"
)

print(
    f"Recall   : {final_val_recall:.4f}"
)

print(
    f"F1       : {final_val_f1:.4f}"
)

print(
    f"AUC      : {final_val_auc:.4f}"
)

print()

print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print()

print(
    "Training history:"
)

print(
    HISTORY_FILE
)

print()

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)