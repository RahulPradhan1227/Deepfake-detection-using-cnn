import csv
import json
import time
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

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_DIR = PROJECT_ROOT / "dataset" / "real"
FAKE_DIR = PROJECT_ROOT / "dataset" / "fake"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = RESULTS_DIR / "data_split.json"

TUNING_RESULTS = RESULTS_DIR / "hyperparameter_results.csv"

IMAGE_SIZE = 224

NUM_WORKERS = 0

RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HYPERPARAMETERS TO TEST
# ============================================================

CONFIGURATIONS = [

    {
        "name": "config_1",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16
    },

    {
        "name": "config_2",
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16
    },

    {
        "name": "config_3",
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16
    },

    {
        "name": "config_4",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.5,
        "batch_size": 16
    },

    {
        "name": "config_5",
        "learning_rate": 5e-5,
        "weight_decay": 1e-5,
        "dropout": 0.5,
        "batch_size": 16
    },

    {
        "name": "config_6",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "batch_size": 32
    }

]


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
# DEVICE
# ============================================================

print("=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

print("Device:", DEVICE)

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

        path, label = self.samples[index]

        image = Image.open(
            path
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
# LOAD DATA SPLIT
# ============================================================

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        "data_split.json not found. "
        "Run train.py first."
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


print("Training samples:", len(train_samples))

print("Validation samples:", len(val_samples))

print()


# ============================================================
# TRANSFORMS
# ============================================================

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


val_transform = transforms.Compose([

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
# MODEL
# ============================================================

def create_model(dropout):

    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
    )

    model = models.efficientnet_b0(
        weights=weights
    )

    # Freeze entire backbone first

    for param in model.features.parameters():

        param.requires_grad = False


    # Fine-tune last three blocks

    for param in model.features[-3:].parameters():

        param.requires_grad = True


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


    return model.to(DEVICE)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    running_loss = 0.0

    all_labels = []

    all_predictions = []


    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
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
        running_loss
        / len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            ).squeeze(1)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.sigmoid(
                outputs
            )


            predictions = (
                probabilities >= 0.5
            ).int()


            all_labels.extend(
                labels.cpu()
                .numpy()
            )


            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )


            all_probabilities.extend(
                probabilities.cpu()
                .numpy()
            )


    loss = (
        running_loss
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
# TUNING
# ============================================================

results = []


for config_number, config in enumerate(
    CONFIGURATIONS,
    start=1
):

    print()
    print("=" * 70)

    print(
        f"CONFIGURATION "
        f"{config_number}/{len(CONFIGURATIONS)}"
    )

    print("=" * 70)


    print(
        "Name:",
        config["name"]
    )

    print(
        "Learning rate:",
        config["learning_rate"]
    )

    print(
        "Weight decay:",
        config["weight_decay"]
    )

    print(
        "Dropout:",
        config["dropout"]
    )

    print(
        "Batch size:",
        config["batch_size"]
    )

    print()


    set_seed(RANDOM_STATE)


    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_dataset = FaceDataset(
        train_samples,
        train_transform
    )


    val_dataset = FaceDataset(
        val_samples,
        val_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        config["dropout"]
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_f1 = -1

    best_auc = 0

    best_accuracy = 0

    best_precision = 0

    best_recall = 0

    start_time = time.time()


    for epoch in range(8):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
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
            val_loader,
            criterion
        )


        scheduler.step(
            val_f1
        )


        print(
            f"Epoch {epoch + 1}/8 | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"AUC: {val_auc:.4f}"
        )


        if val_f1 > best_f1:

            best_f1 = val_f1

            best_auc = val_auc

            best_accuracy = val_accuracy

            best_precision = val_precision

            best_recall = val_recall


    training_time = (
        time.time()
        - start_time
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    result = {

        "configuration": config["name"],

        "learning_rate":
            config["learning_rate"],

        "weight_decay":
            config["weight_decay"],

        "dropout":
            config["dropout"],

        "batch_size":
            config["batch_size"],

        "accuracy":
            best_accuracy,

        "precision":
            best_precision,

        "recall":
            best_recall,

        "f1":
            best_f1,

        "auc":
            best_auc,

        "training_time_seconds":
            training_time
    }


    results.append(
        result
    )


    print()

    print(
        "BEST RESULT FOR",
        config["name"]
    )

    print(
        f"Accuracy : {best_accuracy:.4f}"
    )

    print(
        f"Precision: {best_precision:.4f}"
    )

    print(
        f"Recall   : {best_recall:.4f}"
    )

    print(
        f"F1       : {best_f1:.4f}"
    )

    print(
        f"AUC      : {best_auc:.4f}"
    )

    print(
        f"Time     : {training_time:.1f} sec"
    )


    # Free GPU memory

    del model

    del train_loader

    del val_loader

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [

    "configuration",
    "learning_rate",
    "weight_decay",
    "dropout",
    "batch_size",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "training_time_seconds"
]


with open(
    TUNING_RESULTS,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

best_result = max(
    results,
    key=lambda x: x["f1"]
)


print()
print("=" * 70)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)

print()

print(
    "BEST CONFIGURATION:",
    best_result["configuration"]
)

print(
    "Learning rate:",
    best_result["learning_rate"]
)

print(
    "Weight decay:",
    best_result["weight_decay"]
)

print(
    "Dropout:",
    best_result["dropout"]
)

print(
    "Batch size:",
    best_result["batch_size"]
)

print()

print(
    f"Accuracy : "
    f"{best_result['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1       : "
    f"{best_result['f1']:.4f}"
)

print(
    f"AUC      : "
    f"{best_result['auc']:.4f}"
)

print()

print(
    "Results saved to:"
)

print(
    TUNING_RESULTS
)

print()