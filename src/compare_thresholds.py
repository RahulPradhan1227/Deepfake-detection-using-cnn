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
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_PATH = PROJECT_ROOT / "results" / "data_split.json"

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_PATH = RESULTS_DIR / "fair_model_comparison.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODELS
# ============================================================

MODEL_PATHS = {

    "EfficientNet-B0":
        PROJECT_ROOT /
        "models" /
        "deepfake_efficientnet_b0_final.pth",

    "ResNet18":
        PROJECT_ROOT /
        "models" /
        "resnet18_deepfake.pth",

    "MobileNetV3-Large":
        PROJECT_ROOT /
        "models" /
        "mobilenetv3_large_deepfake.pth",

    "CNN":
        PROJECT_ROOT /
        "models" /
        "cnn_deepfake.pth",

    "ConvNeXt-Tiny":
        PROJECT_ROOT /
        "models" /
        "convnext_tiny_deepfake.pth"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("FAIR MODEL COMPARISON WITH VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 75)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels
    ):

        self.image_paths = image_paths

        self.labels = labels

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

                label = item.get(
                    "label"
                )

                if (
                    path is not None
                    and
                    label is not None
                ):

                    paths.append(path)

                    labels.append(
                        int(label)
                    )

    return paths, labels


val_paths, val_labels = get_split(
    "validation"
)

test_paths, test_labels = get_split(
    "test"
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

        candidate = (
            PROJECT_ROOT / path
        )

        if candidate.exists():

            fixed.append(candidate)

            continue

        matches = list(
            PROJECT_ROOT.rglob(
                path.name
            )
        )

        if matches:

            fixed.append(matches[0])

        else:

            raise FileNotFoundError(
                f"Could not find image: {path}"
            )

    return fixed


val_paths = fix_paths(
    val_paths
)

test_paths = fix_paths(
    test_paths
)


print(
    f"Validation images: {len(val_paths)}"
)

print(
    f"Test images:       {len(test_paths)}"
)

print()


# ============================================================
# DATA LOADERS
# ============================================================

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels[:len(val_paths)]
)

test_dataset = DeepfakeDataset(
    test_paths,
    test_labels[:len(test_paths)]
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class DeepfakeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32, 3, padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),

            nn.Conv2d(
                32, 64, 3, padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),

            nn.Conv2d(
                64, 128, 3, padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.15),

            nn.Conv2d(
                128, 256, 3, padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20),

            nn.Conv2d(
                256, 512, 3, padding=1
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

        return self.classifier(
            x
        ).squeeze(1)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(name):

    if name == "EfficientNet-B0":

        model = models.efficientnet_b0(
            weights=None
        )

        model.classifier = nn.Sequential(

            nn.Dropout(0.30),

            nn.Linear(
                model.classifier[1].in_features,
                1
            )
        )

        return model


    if name == "ResNet18":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Sequential(

            nn.Dropout(0.30),

            nn.Linear(
                model.fc.in_features,
                1
            )
        )

        return model


    if name == "MobileNetV3-Large":

        model = models.mobilenet_v3_large(
            weights=None
        )

        model.classifier = nn.Sequential(

            nn.Linear(
                model.classifier[0].in_features,
                1280
            ),

            nn.Hardswish(),

            nn.Dropout(0.30),

            nn.Linear(
                1280,
                1
            )
        )

        return model


    if name == "ConvNeXt-Tiny":

        model = models.convnext_tiny(
            weights=None
        )

        input_features = (
            model.classifier[-1].in_features
        )

        model.classifier = nn.Sequential(

            nn.Flatten(
                start_dim=1
            ),

            nn.LayerNorm(
                input_features
            ),

            nn.Linear(
                input_features,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                1
            )
        )

        return model


    if name == "CNN":

        return DeepfakeCNN()


    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    path
):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict,
        strict=True
    )

    return model


# ============================================================
# GET PROBABILITIES
# ============================================================

def get_predictions(
    model,
    loader
):

    model.eval()

    probabilities = []

    labels = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            probabilities.extend(
                torch.sigmoid(
                    outputs
                ).cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return (
        np.array(probabilities),
        np.array(labels).astype(int)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
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
        probabilities
    )

    cm = confusion_matrix(
        labels,
        predictions
    )


    return {

        "threshold":
            float(threshold),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "auc":
            float(auc),

        "confusion_matrix":
            cm.tolist()
    }


# ============================================================
# FIND BEST VALIDATION THRESHOLD
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50

    best_accuracy = -1.0

    best_metrics = None


    for threshold in np.arange(
        0.10,
        0.901,
        0.01
    ):

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold
        )

        if (
            metrics["accuracy"]
            >
            best_accuracy
        ):

            best_accuracy = (
                metrics["accuracy"]
            )

            best_threshold = (
                threshold
            )

            best_metrics = metrics


    return (
        best_threshold,
        best_metrics
    )


# ============================================================
# RUN COMPARISON
# ============================================================

all_results = {}


for model_name, model_path in MODEL_PATHS.items():

    print("=" * 75)

    print(
        f"EVALUATING: {model_name}"
    )

    print("=" * 75)


    if not model_path.exists():

        print(
            f"WARNING: Model not found:"
        )

        print(
            model_path
        )

        print()

        continue


    # Create model

    model = create_model(
        model_name
    )


    # Load weights

    model = load_checkpoint(
        model,
        model_path
    )


    model = model.to(
        DEVICE
    )


    # Validation predictions

    (
        val_probabilities,
        val_ground_truth
    ) = get_predictions(
        model,
        val_loader
    )


    # Find threshold using validation ONLY

    (
        best_threshold,
        val_metrics
    ) = find_best_threshold(
        val_ground_truth,
        val_probabilities
    )


    print(
        f"Best validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_metrics['accuracy'] * 100:.2f}%"
    )


    # ========================================================
    # TEST EVALUATION
    # ========================================================

    (
        test_probabilities,
        test_ground_truth
    ) = get_predictions(
        model,
        test_loader
    )


    # IMPORTANT:
    # Test uses the threshold selected from validation.

    test_metrics = calculate_metrics(
        test_ground_truth,
        test_probabilities,
        best_threshold
    )


    print()
    print(
        "TEST RESULT"
    )

    print(
        f"Threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{test_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{test_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1        : "
        f"{test_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"AUC       : "
        f"{test_metrics['auc'] * 100:.2f}%"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            test_metrics[
                "confusion_matrix"
            ]
        )
    )

    print()


    all_results[model_name] = {

        "validation": {

            "optimal_threshold":
                float(best_threshold),

            "accuracy":
                val_metrics[
                    "accuracy"
                ],

            "precision":
                val_metrics[
                    "precision"
                ],

            "recall":
                val_metrics[
                    "recall"
                ],

            "f1":
                val_metrics[
                    "f1"
                ],

            "auc":
                val_metrics[
                    "auc"
                ]
        },

        "test": test_metrics
    }


# ============================================================
# RANK MODELS
# ============================================================

ranking = sorted(

    all_results.items(),

    key=lambda item:
        item[1]["test"]["accuracy"],

    reverse=True
)


print("=" * 75)
print("FINAL FAIR MODEL COMPARISON")
print("=" * 75)

print()

print(
    f"{'Model':<22}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'AUC':>12}"
)

print("-" * 82)


for model_name, result in ranking:

    test = result["test"]

    print(

        f"{model_name:<22}"

        f"{test['accuracy'] * 100:>11.2f}%"

        f"{test['precision'] * 100:>11.2f}%"

        f"{test['recall'] * 100:>11.2f}%"

        f"{test['f1'] * 100:>11.2f}%"

        f"{test['auc'] * 100:>11.2f}%"
    )


print()


if ranking:

    winner = ranking[0][0]

    print(
        f"🏆 Best accuracy model: "
        f"{winner}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "evaluation_method":
        "Threshold optimized on validation set; "
        "test set evaluated once using frozen threshold.",

    "models":
        all_results,

    "ranking_by_test_accuracy": [

        {
            "model":
                model_name,

            "accuracy":
                result["test"]["accuracy"],

            "precision":
                result["test"]["precision"],

            "recall":
                result["test"]["recall"],

            "f1":
                result["test"]["f1"],

            "auc":
                result["test"]["auc"],

            "threshold":
                result["test"]["threshold"]
        }

        for model_name, result
        in ranking
    ]
}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


print()
print("=" * 75)

print(
    f"Results saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 75)