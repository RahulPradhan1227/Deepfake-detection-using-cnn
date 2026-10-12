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
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_tuned_final.pth"
)

# IMPORTANT:
# Change this ONLY if your tune_efficientnet.py uses
# a different SPLIT_PATH.
SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "data_split.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "final_test_evaluation.json"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "test_predictions.json"
)

IMAGE_SIZE = 224

BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

print()

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not SPLIT_PATH.exists():

    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_PATH}\n\n"
        "Please make sure SPLIT_PATH matches the one used "
        "in tune_efficientnet.py."
    )


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

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        image = self.transform(
            image
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return image, label


# ============================================================
# LOAD SPLIT FILE
# ============================================================

with open(
    SPLIT_PATH,
    "r"
) as f:

    split_data = json.load(f)


# ============================================================
# GET SPLIT
# ============================================================

def get_split(name):

    if name not in split_data:

        raise KeyError(
            f"'{name}' split was not found in:\n"
            f"{SPLIT_PATH}\n\n"
            f"Available splits: "
            f"{list(split_data.keys())}"
        )

    split = split_data[name]

    paths = []

    labels = []


    # --------------------------------------------------------
    # FORMAT 1:
    # {
    #   "paths": [...],
    #   "labels": [...]
    # }
    # --------------------------------------------------------

    if isinstance(split, dict):

        if (
            "paths" in split
            and
            "labels" in split
        ):

            paths = split["paths"]

            labels = [
                int(label)
                for label in split["labels"]
            ]


        # ----------------------------------------------------
        # FORMAT 2:
        # {
        #   "real": [...],
        #   "fake": [...]
        # }
        # ----------------------------------------------------

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


    # --------------------------------------------------------
    # FORMAT 3:
    # [
    #   {"path": "...", "label": 0},
    #   {"path": "...", "label": 1}
    # ]
    # --------------------------------------------------------

    elif isinstance(split, list):

        for item in split:

            if not isinstance(
                item,
                dict
            ):

                continue


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


    if len(paths) == 0:

        raise ValueError(
            f"No images found in '{name}' split."
        )


    if len(paths) != len(labels):

        raise ValueError(
            f"{name} split has "
            f"{len(paths)} paths but "
            f"{len(labels)} labels."
        )


    return paths, labels


# ============================================================
# FIX PATHS
# ============================================================

def fix_paths(paths):

    fixed = []

    for path in paths:

        path = Path(path)


        # Absolute/existing path

        if path.exists():

            fixed.append(
                path.resolve()
            )

            continue


        # Relative to project root

        candidate = (
            PROJECT_ROOT
            / path
        )

        if candidate.exists():

            fixed.append(
                candidate.resolve()
            )

            continue


        # Search by filename

        matches = list(
            PROJECT_ROOT.rglob(
                path.name
            )
        )


        if matches:

            fixed.append(
                matches[0].resolve()
            )

        else:

            raise FileNotFoundError(
                f"Could not find image:\n{path}"
            )


    return fixed


# ============================================================
# LOAD TEST SPLIT
# ============================================================

test_paths, test_labels = get_split(
    "test"
)

test_paths = fix_paths(
    test_paths
)


print(
    f"Test images: {len(test_paths)}"
)

print()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

real_count = sum(
    label == 0
    for label in test_labels
)

fake_count = sum(
    label == 1
    for label in test_labels
)

print(
    f"Real images: {real_count}"
)

print(
    f"Fake images: {fake_count}"
)

print()


# ============================================================
# DATALOADER
# ============================================================

test_dataset = DeepfakeDataset(
    test_paths,
    test_labels
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    dropout=0.3
):

    model = models.efficientnet_b0(
        weights=None
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
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Get configuration stored inside checkpoint

config = checkpoint.get(
    "config",
    {}
)

dropout = config.get(
    "dropout",
    0.3
)

print("Model configuration:")
print(config)

print()


# ============================================================
# CREATE MODEL
# ============================================================

model = create_model(
    dropout=dropout
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# EVALUATE
# ============================================================

all_probabilities = []

all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        outputs = model(
            images
        ).squeeze(1)


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


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

DEFAULT_THRESHOLD = 0.50

predictions = (
    probabilities >= DEFAULT_THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    labels,
    predictions
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    labels,
    predictions,
    target_names=[
        "REAL",
        "DEEPFAKE"
    ],
    zero_division=0,
    output_dict=True
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print("=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {auc * 100:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "Confusion Matrix:"
)

print()

print(
    "                 Predicted"
)

print(
    "                 REAL  FAKE"
)

print(
    f"Actual REAL      "
    f"{cm[0][0]:4d}  "
    f"{cm[0][1]:4d}"
)

print(
    f"Actual FAKE      "
    f"{cm[1][0]:4d}  "
    f"{cm[1][1]:4d}"
)

print()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

print()

threshold_results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    threshold_predictions = (
        probabilities >= threshold
    ).astype(int)


    threshold_accuracy = (
        accuracy_score(
            labels,
            threshold_predictions
        )
    )


    threshold_precision = (
        precision_score(
            labels,
            threshold_predictions,
            zero_division=0
        )
    )


    threshold_recall = (
        recall_score(
            labels,
            threshold_predictions,
            zero_division=0
        )
    )


    threshold_f1 = (
        f1_score(
            labels,
            threshold_predictions,
            zero_division=0
        )
    )


    threshold_results.append({

        "threshold":
            round(
                float(threshold),
                2
            ),

        "accuracy":
            float(
                threshold_accuracy
            ),

        "precision":
            float(
                threshold_precision
            ),

        "recall":
            float(
                threshold_recall
            ),

        "f1":
            float(
                threshold_f1
            )
    })


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_threshold_result = max(
    threshold_results,
    key=lambda x: x["f1"]
)

best_threshold = (
    best_threshold_result[
        "threshold"
    ]
)


best_threshold_predictions = (
    probabilities >= best_threshold
).astype(int)


best_threshold_cm = (
    confusion_matrix(
        labels,
        best_threshold_predictions
    )
)


print(
    f"Best threshold by F1: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy : "
    f"{best_threshold_result['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best_threshold_result['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{best_threshold_result['recall'] * 100:.2f}%"
)

print(
    f"F1       : "
    f"{best_threshold_result['f1'] * 100:.2f}%"
)

print()


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


final_results = {

    "model":
        str(MODEL_PATH),

    "split_file":
        str(SPLIT_PATH),

    "test_images":
        len(test_paths),

    "real_images":
        real_count,

    "fake_images":
        fake_count,

    "model_configuration":
        config,

    "default_threshold":
        DEFAULT_THRESHOLD,

    "metrics_at_0.50": {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(auc)
    },

    "confusion_matrix": [
        [
            int(cm[0][0]),
            int(cm[0][1])
        ],
        [
            int(cm[1][0]),
            int(cm[1][1])
        ]
    ],

    "classification_report":
        report,

    "best_threshold_by_f1":
        best_threshold_result,

    "best_threshold_confusion_matrix": [
        [
            int(best_threshold_cm[0][0]),
            int(best_threshold_cm[0][1])
        ],
        [
            int(best_threshold_cm[1][0]),
            int(best_threshold_cm[1][1])
        ]
    ],

    "threshold_results":
        threshold_results
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


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_records = []

for path, label, probability in zip(
    test_paths,
    labels,
    probabilities
):

    prediction = int(
        probability >= DEFAULT_THRESHOLD
    )


    prediction_records.append({

        "image":
            str(path),

        "true_label":
            "DEEPFAKE"
            if label == 1
            else "REAL",

        "predicted_label":
            "DEEPFAKE"
            if prediction == 1
            else "REAL",

        "deepfake_probability":
            float(probability),

        "real_probability":
            float(
                1.0 - probability
            ),

        "correct":
            bool(
                prediction == label
            )
    })


with open(
    PREDICTIONS_PATH,
    "w"
) as f:

    json.dump(
        prediction_records,
        f,
        indent=4
    )


# ============================================================
# FINISHED
# ============================================================

print("=" * 80)

print(
    "Evaluation completed successfully."
)

print()

print(
    "Results saved to:"
)

print(
    RESULTS_PATH
)

print()

print(
    "Predictions saved to:"
)

print(
    PREDICTIONS_PATH
)

print("=" * 80)