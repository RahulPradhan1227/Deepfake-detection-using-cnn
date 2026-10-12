import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "deepfake_efficientnet_best.pth"
FINAL_MODEL_PATH = MODEL_DIR / "deepfake_efficientnet_final.pth"
RESULTS_PATH = MODEL_DIR / "training_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

RANDOM_STATE = 42

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DEEPFAKE DETECTION - EFFICIENTNET-B0")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
else:
    print("WARNING: CUDA is not available. Training on CPU.")

print()


# ============================================================
# COLLECT DATASET
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def get_images(folder, label):
    images = []

    for path in folder.iterdir():

        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            images.append((str(path), label))

    return images


real_images = get_images(REAL_DIR, 0)
fake_images = get_images(FAKE_DIR, 1)

all_images = real_images + fake_images

print("Dataset")
print("-" * 70)
print(f"Real images : {len(real_images)}")
print(f"Fake images : {len(fake_images)}")
print(f"Total images: {len(all_images)}")
print()


if len(all_images) == 0:
    raise RuntimeError(
        "No images found. Check dataset/real and dataset/fake."
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

paths = np.array([x[0] for x in all_images])
labels = np.array([x[1] for x in all_images])

# First: 80% train, 20% temporary
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels,
)

# Second: temporary -> 10% validation + 10% test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_labels,
)


print("Dataset Split")
print("-" * 70)
print(f"Training   : {len(train_paths)}")
print(f"Validation : {len(val_paths)}")
print(f"Testing    : {len(test_paths)}")
print()


# ============================================================
# DATASET CLASS
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(self, paths, labels, transform=None):

        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_path = self.paths[index]
        label = self.labels[index]

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as e:

            print(f"\nCould not read: {image_path}")
            print(e)

            image = Image.new(
                "RGB",
                (IMAGE_SIZE, IMAGE_SIZE),
            )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.float32
        )


# ============================================================
# TRANSFORMS
# ============================================================

imagenet_mean = [
    0.485,
    0.456,
    0.406,
]

imagenet_std = [
    0.229,
    0.224,
    0.225,
]


train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std,
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std,
    ),
])


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = DeepfakeDataset(
    train_paths,
    train_labels,
    train_transform,
)

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels,
    eval_transform,
)

test_dataset = DeepfakeDataset(
    test_paths,
    test_labels,
    eval_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

real_count = np.sum(train_labels == 0)
fake_count = np.sum(train_labels == 1)

total_train = len(train_labels)

real_weight = total_train / (2 * real_count)
fake_weight = total_train / (2 * fake_count)

print("Class Weights")
print("-" * 70)
print(f"Real weight: {real_weight:.4f}")
print(f"Fake weight: {fake_weight:.4f}")
print()


# ============================================================
# MODEL
# ============================================================

print("Loading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# Freeze backbone initially
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Replace classifier
in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.40),
    nn.Linear(in_features, 1),
)


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

pos_weight = torch.tensor(
    [fake_weight / real_weight],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_probs = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images).squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        probabilities = torch.sigmoid(
            outputs
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_probs.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(loader.dataset)

    predictions = (
        np.array(all_probs) >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    return epoch_loss, accuracy, f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_probs = []

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

            probabilities = torch.sigmoid(
                outputs
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probabilities.cpu().numpy()
            )

    loss = running_loss / len(loader.dataset)

    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)

    predictions = (
        probs_np >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels_np,
        predictions
    )

    precision = precision_score(
        labels_np,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels_np,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels_np,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        labels_np,
        probs_np
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        auc,
        labels_np,
        probs_np,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0
patience = 5
patience_counter = 0

history = []

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()


for epoch in range(1, EPOCHS + 1):

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader
    )

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc,
        _,
        _,
    ) = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Precision: {val_precision:.4f} | "
        f"Val Recall: {val_recall:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.2e}"
    )

    print()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "learning_rate": current_lr,
    })


    # Save best model
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "class_names": {
                    "0": "real",
                    "1": "fake",
                },
                "best_val_f1": best_val_f1,
            },
            BEST_MODEL_PATH,
        )

        print(
            f"✓ Best model saved "
            f"(Val F1 = {best_val_f1:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{patience})"
        )

        if patience_counter >= patience:

            print()
            print("Early stopping triggered.")
            break

    print("-" * 70)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best validation F1: "
    f"{checkpoint['best_val_f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_auc,
    test_labels_np,
    test_probs_np,
) = evaluate(
    model,
    test_loader
)


test_predictions = (
    test_probs_np >= 0.5
).astype(int)


cm = confusion_matrix(
    test_labels_np,
    test_predictions
)


print()
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_accuracy:.4f}")
print(f"Test Precision : {test_precision:.4f}")
print(f"Test Recall    : {test_recall:.4f}")
print(f"Test F1 Score  : {test_f1:.4f}")
print(f"Test ROC-AUC   : {test_auc:.4f}")

print()
print("Confusion Matrix")
print("-" * 70)
print(cm)

print()
print("Classification Report")
print("-" * 70)

print(
    classification_report(
        test_labels_np,
        test_predictions,
        target_names=[
            "REAL",
            "DEEPFAKE",
        ],
        zero_division=0,
    )
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "image_size": IMAGE_SIZE,
        "class_names": {
            "0": "real",
            "1": "fake",
        },
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auc": test_auc,
    },
    FINAL_MODEL_PATH,
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "dataset": {
        "real": int(len(real_images)),
        "fake": int(len(fake_images)),
        "total": int(len(all_images)),
    },

    "split": {
        "train": int(len(train_paths)),
        "validation": int(len(val_paths)),
        "test": int(len(test_paths)),
    },

    "device": str(DEVICE),

    "best_validation_f1": float(best_val_f1),

    "test": {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "roc_auc": float(test_auc),
    },

    "confusion_matrix": cm.tolist(),

    "history": history,
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(f"Best model : {BEST_MODEL_PATH}")
print(f"Final model: {FINAL_MODEL_PATH}")
print(f"Results    : {RESULTS_PATH}")
