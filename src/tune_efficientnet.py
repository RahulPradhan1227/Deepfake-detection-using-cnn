import json
import copy
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

SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_MODEL_PATH = (
    MODEL_DIR / "efficientnet_b0_tuned_final.pth"
)

RESULTS_PATH = (
    RESULTS_DIR / "efficientnet_tuning_results.json"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================

CONFIGS = [

    {
        "name": "config_1",
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "dropout": 0.30,
        "unfreeze_blocks": 3,
    },

    {
        "name": "config_2",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.30,
        "unfreeze_blocks": 3,
    },

    {
        "name": "config_3",
        "lr": 5e-5,
        "weight_decay": 5e-4,
        "dropout": 0.30,
        "unfreeze_blocks": 3,
    },

    {
        "name": "config_4",
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "dropout": 0.50,
        "unfreeze_blocks": 3,
    },

    {
        "name": "config_5",
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "dropout": 0.30,
        "unfreeze_blocks": 5,
    },

    {
        "name": "config_6",
        "lr": 2e-5,
        "weight_decay": 1e-4,
        "dropout": 0.30,
        "unfreeze_blocks": 3,
    },

]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("EFFICIENTNET-B0 HYPERPARAMETER TUNING")
print("=" * 80)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()

print(
    f"Number of configurations: {len(CONFIGS)}"
)

print()


# ============================================================
# DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        training=False
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.training = training

        if training:

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
# LOAD SPLIT
# ============================================================

with open(
    SPLIT_PATH,
    "r"
) as f:

    split_data = json.load(f)


def get_split(name):

    split = split_data[name]

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


train_paths, train_labels = get_split(
    "train"
)

val_paths, val_labels = get_split(
    "validation"
)


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

            raise FileNotFoundError(
                f"Could not find: {path}"
            )

    return fixed


train_paths = fix_paths(
    train_paths
)

val_paths = fix_paths(
    val_paths
)


print(
    f"Training images:   {len(train_paths)}"
)

print(
    f"Validation images: {len(val_paths)}"
)

print()


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = DeepfakeDataset(
    train_paths,
    train_labels,
    training=True
)

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels,
    training=False
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


# ============================================================
# MODEL
# ============================================================

def create_model(
    dropout
):

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(

        nn.Dropout(
            dropout
        ),

        nn.Linear(
            input_features,
            1
        )
    )

    return model


# ============================================================
# PREPARE MODEL
# ============================================================

def prepare_model(
    config
):

    model = create_model(
        config["dropout"]
    )

    # Freeze everything first

    for parameter in model.parameters():

        parameter.requires_grad = False


    # Classifier is trainable

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True


    # EfficientNet-B0 features contain 9 blocks.
    # Unfreeze the last N blocks.

    feature_blocks = list(
        model.features.children()
    )

    number_to_unfreeze = (
        config["unfreeze_blocks"]
    )

    for block in feature_blocks[
        -number_to_unfreeze:
    ]:

        for parameter in block.parameters():

            parameter.requires_grad = True


    return model.to(DEVICE)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                labels,
                probabilities
            )
    }


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model
):

    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            probabilities = torch.sigmoid(
                outputs
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    probabilities = np.array(
        all_probabilities
    )

    labels = np.array(
        all_labels
    ).astype(int)

    metrics = calculate_metrics(
        labels,
        probabilities
    )

    return metrics


# ============================================================
# TRAIN ONE CONFIGURATION
# ============================================================

def train_configuration(
    config
):

    print("=" * 80)

    print(
        f"RUNNING {config['name']}"
    )

    print(
        f"Learning rate : {config['lr']}"
    )

    print(
        f"Weight decay  : {config['weight_decay']}"
    )

    print(
        f"Dropout       : {config['dropout']}"
    )

    print(
        f"Unfreeze      : {config['unfreeze_blocks']} blocks"
    )

    print("=" * 80)


    model = prepare_model(
        config
    )


    criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.AdamW(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=config["lr"],

        weight_decay=config[
            "weight_decay"
        ]
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2
    )


    best_state = None

    best_f1 = -1

    best_epoch = 0

    patience_counter = 0

    MAX_EPOCHS = 12

    EARLY_STOP = 4


    history = []


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

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
                *
                images.size(0)
            )


            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.50
            ).float()


            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )


        val_metrics = evaluate(
            model
        )


        scheduler.step(
            val_metrics["f1"]
        )


        history.append({

            "epoch": epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_precision":
                val_metrics[
                    "precision"
                ],

            "val_recall":
                val_metrics[
                    "recall"
                ],

            "val_f1":
                val_metrics[
                    "f1"
                ],

            "val_auc":
                val_metrics[
                    "auc"
                ]
        })


        print(

            f"Epoch {epoch:02d} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Train Acc: "
            f"{train_accuracy:.4f} | "

            f"Val Acc: "
            f"{val_metrics['accuracy']:.4f} | "

            f"Val F1: "
            f"{val_metrics['f1']:.4f} | "

            f"Val AUC: "
            f"{val_metrics['auc']:.4f}"
        )


        # Save best by validation F1

        if (
            val_metrics["f1"]
            >
            best_f1
        ):

            best_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if (
            patience_counter
            >=
            EARLY_STOP
        ):

            print(
                "Early stopping."
            )

            break


    model.load_state_dict(
        best_state
    )


    final_val_metrics = evaluate(
        model
    )


    print()

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{final_val_metrics['f1']:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{final_val_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation AUC: "
        f"{final_val_metrics['auc'] * 100:.2f}%"
    )

    print()


    return (
        model,
        final_val_metrics,
        history,
        best_epoch
    )


# ============================================================
# RUN ALL CONFIGURATIONS
# ============================================================

all_results = []

best_overall_model = None
best_overall_config = None
best_overall_score = -1
best_overall_metrics = None


for config in CONFIGS:

    (
        model,
        metrics,
        history,
        best_epoch
    ) = train_configuration(
        config
    )


    result = {

        "config":
            config,

        "best_epoch":
            best_epoch,

        "validation_metrics":
            metrics,

        "history":
            history
    }


    all_results.append(
        result
    )


    # Select using validation F1

    if (
        metrics["f1"]
        >
        best_overall_score
    ):

        best_overall_score = (
            metrics["f1"]
        )

        best_overall_model = copy.deepcopy(
            model.state_dict()
        )

        best_overall_config = config

        best_overall_metrics = metrics


    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# SAVE BEST MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


checkpoint = {

    "model_state_dict":
        best_overall_model,

    "config":
        best_overall_config,

    "validation_metrics":
        best_overall_metrics,

    "image_size":
        IMAGE_SIZE
}


torch.save(
    checkpoint,
    FINAL_MODEL_PATH
)


# ============================================================
# SORT RESULTS
# ============================================================

all_results_sorted = sorted(

    all_results,

    key=lambda x:
        x["validation_metrics"]["f1"],

    reverse=True
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 80)

print()

print(
    f"{'Config':<12}"
    f"{'Val Acc':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'AUC':>12}"
)

print("-" * 72)


for result in all_results_sorted:

    metrics = result[
        "validation_metrics"
    ]

    print(

        f"{result['config']['name']:<12}"

        f"{metrics['accuracy'] * 100:>11.2f}%"

        f"{metrics['precision'] * 100:>11.2f}%"

        f"{metrics['recall'] * 100:>11.2f}%"

        f"{metrics['f1'] * 100:>11.2f}%"

        f"{metrics['auc'] * 100:>11.2f}%"
    )


print()

print("=" * 80)

print(
    "BEST CONFIGURATION"
)

print("=" * 80)

print()

print(
    best_overall_config
)

print()

print(
    f"Validation Accuracy : "
    f"{best_overall_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Validation Precision: "
    f"{best_overall_metrics['precision'] * 100:.2f}%"
)

print(
    f"Validation Recall   : "
    f"{best_overall_metrics['recall'] * 100:.2f}%"
)

print(
    f"Validation F1       : "
    f"{best_overall_metrics['f1'] * 100:.2f}%"
)

print(
    f"Validation AUC      : "
    f"{best_overall_metrics['auc'] * 100:.2f}%"
)

print()

print(
    "Best model saved to:"
)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE JSON
# ============================================================

final_results = {

    "device":
        str(DEVICE),

    "selection_metric":
        "validation_f1",

    "best_configuration":
        best_overall_config,

    "best_validation_metrics":
        best_overall_metrics,

    "configurations":
        all_results_sorted
}


with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


print()

print(
    "Tuning results saved to:"
)

print(
    RESULTS_PATH
)

print("=" * 80)