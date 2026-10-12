
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
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
from sklearn.metrics import roc_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def __init__(self):
    super().__init__()

    self.names = [
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v3_small",
        "densenet121",
        "shufflenet_v2",
    ]

    self.backbones = nn.ModuleList([
        build_backbone(name)
        for name in self.names
    ])

    # Freeze all backbone parameters first.
    for model in self.backbones:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # ResNet: fine-tune the final residual block and classifier.
    for p in self.backbones[0].layer4.parameters():
        p.requires_grad = True
    for p in self.backbones[0].fc.parameters():
        p.requires_grad = True

    # EfficientNet: fine-tune the final feature block and classifier.
    for p in self.backbones[1].features[7].parameters():
        p.requires_grad = True
    for p in self.backbones[1].classifier.parameters():
        p.requires_grad = True

    # MobileNetV3: fine-tune its final feature blocks and classifier.
    for p in self.backbones[2].features[10:].parameters():
        p.requires_grad = True
    for p in self.backbones[2].classifier.parameters():
        p.requires_grad = True

    # DenseNet: fine-tune the final dense block and classifier.
    for p in self.backbones[3].features.denseblock4.parameters():
        p.requires_grad = True
    for p in self.backbones[3].features.norm5.parameters():
        p.requires_grad = True
    for p in self.backbones[3].classifier.parameters():
        p.requires_grad = True

    # ShuffleNet: fine-tune the last stage and classifier.
    for p in self.backbones[4].stage4.parameters():
        p.requires_grad = True
    for p in self.backbones[4].conv5.parameters():
        p.requires_grad = True
    for p in self.backbones[4].fc.parameters():
        p.requires_grad = True

    # Trainable fusion layer.
    self.fusion = nn.Linear(len(self.names), 1)

    with torch.no_grad():
        self.fusion.weight.fill_(1.0 / len(self.names))
        self.fusion.bias.zero_()

    print("\nTrainable parameters by backbone:")
    for name, model in zip(self.names, self.backbones):
        count = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        print(f"{name}: {count:,}")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "five_backbone_fusion.pth"
RESULTS_PATH = OUTPUT_DIR / "fusion_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE =2e-5
SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# FIND DATASET
# Expected structure:
# dataset/
#   real/
#   fake/
# ============================================================

def find_dataset():
    candidates = [
        PROJECT_ROOT / "dataset" / "face_dataset",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "face_dataset",
        PROJECT_ROOT / "images",
        PROJECT_ROOT / "dataset" / "train",
    ]

    for root in candidates:
        if not root.is_dir():
            continue

        subdirs = {
            p.name.lower(): p
            for p in root.iterdir()
            if p.is_dir()
        }

        real_names = [
            "real", "real_faces", "training_real"
        ]
        fake_names = [
            "fake", "fake_faces", "deepfake",
            "deepfakes", "training_fake"
        ]

        real_dir = next(
            (subdirs[n] for n in real_names if n in subdirs),
            None,
        )
        fake_dir = next(
            (subdirs[n] for n in fake_names if n in subdirs),
            None,
        )

        if real_dir and fake_dir:
            print(f"Dataset found: {root}")
            print(f"Real images: {real_dir}")
            print(f"Fake images: {fake_dir}")
            return root

    raise FileNotFoundError(
        "\nCould not find the dataset.\n"
        "Expected a folder containing both real and fake folders.\n"
        "Example: dataset/face_dataset/real and dataset/face_dataset/fake\n"
        f"Project root: {PROJECT_ROOT}"
    )


# ============================================================
# DATASET AND TRANSFORMS
# ============================================================

DATASET_ROOT = find_dataset()

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ImageFolder assigns class indices alphabetically.
# We explicitly verify which index represents each class.
dataset_for_labels = datasets.ImageFolder(
    root=str(DATASET_ROOT),
    transform=eval_transform,
)

class_names = dataset_for_labels.classes

if len(class_names) != 2:
    raise ValueError(
        f"Expected exactly two classes, found: {class_names}"
    )

real_class_names = {
    "real", "real_faces", "training_real"
}
fake_class_names = {
    "fake", "fake_faces", "deepfake",
    "deepfakes", "training_fake"
}

real_indices = [
    i for i, name in enumerate(class_names)
    if name.lower() in real_class_names
]
fake_indices = [
    i for i, name in enumerate(class_names)
    if name.lower() in fake_class_names
]

if len(real_indices) != 1 or len(fake_indices) != 1:
    raise ValueError(
        f"Could not identify real/fake class names: {class_names}"
    )

REAL_LABEL = real_indices[0]
FAKE_LABEL = fake_indices[0]

targets = np.array(dataset_for_labels.targets)
indices = np.arange(len(targets))

train_indices, remaining_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=targets,
)

val_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=targets[remaining_indices],
)

# Use augmentation only for the training subset.
train_dataset_full = datasets.ImageFolder(
    root=str(DATASET_ROOT),
    transform=train_transform,
)

eval_dataset_full = datasets.ImageFolder(
    root=str(DATASET_ROOT),
    transform=eval_transform,
)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(eval_dataset_full, val_indices)
test_dataset = Subset(eval_dataset_full, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

print("\n" + "=" * 70)
print("FIVE-BACKBONE TRANSFER-LEARNING FUSION")
print("=" * 70)
print("Project root:", PROJECT_ROOT)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Classes:", class_names)
print("Real label:", REAL_LABEL)
print("Fake label:", FAKE_LABEL)
print("Total images:", len(dataset_for_labels))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION)


# ============================================================
# FIVE-BACKBONE FUSION MODEL
# ============================================================

def build_backbone(name):
    """Create a pretrained backbone with a binary classifier."""

    if name == "resnet18":
        try:
            model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        except Exception:
            model = models.resnet18(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 1)

    elif name == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
        except Exception:
            model = models.efficientnet_b0(weights=None)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 1
        )

    elif name == "mobilenet_v3_small":
        try:
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
        except Exception:
            model = models.mobilenet_v3_small(weights=None)

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, 1
        )

    elif name == "densenet121":
        try:
            model = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT
            )
        except Exception:
            model = models.densenet121(weights=None)

        model.classifier = nn.Linear(
            model.classifier.in_features, 1
        )

    elif name == "shufflenet_v2":
        try:
            model = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            )
        except Exception:
            model = models.shufflenet_v2_x1_0(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 1)

    else:
        raise ValueError(f"Unknown backbone: {name}")

    return model


class FiveBackboneFusion(nn.Module):
    """
    Each backbone produces one fake-class logit.
    The five logits are combined by a trainable fusion layer.
    """

    def __init__(self):
        super().__init__()

        self.names = [
            "resnet18",
            "efficientnet_b0",
            "mobilenet_v3_small",
            "densenet121",
            "shufflenet_v2",
        ]

        self.backbones = nn.ModuleList([
            build_backbone(name)
            for name in self.names
        ])

        self.fusion = nn.Linear(len(self.names), 1)

        # Initially average the five predictions.
        with torch.no_grad():
            self.fusion.weight.fill_(1.0 / len(self.names))
            self.fusion.bias.zero_()

    def forward(self, x):
        predictions = []

        for model in self.backbones:
            logits = model(x)
            predictions.append(logits)

        combined = torch.cat(predictions, dim=1)
        fused_logits = self.fusion(combined)

        return fused_logits


# ============================================================
# TRAINING AND EVALUATION HELPERS
# ============================================================

def get_targets(batch_targets):
    """Convert ImageFolder labels into real=0 and fake=1."""
    return (batch_targets == FAKE_LABEL).float().unsqueeze(1)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_probs = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Training" if training else "Evaluating",
    )

    for step, (images, labels) in progress:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )
        labels = get_targets(labels).to(
            DEVICE,
            non_blocking=True,
        )

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                # Scale loss to accumulate gradients over batches.
                scaled_loss = loss / GRADIENT_ACCUMULATION
                scaled_loss.backward()

                is_accumulation_step = (
                    (step + 1) % GRADIENT_ACCUMULATION == 0
                )
                is_last_batch = step + 1 == len(loader)

                if is_accumulation_step or is_last_batch:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        true_labels = labels.detach().cpu().numpy().ravel()

        all_probs.extend(probs.tolist())
        all_labels.extend(true_labels.astype(int).tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(loader.dataset)
    predictions = (np.array(all_probs) >= 0.5).astype(int)
    labels_array = np.array(all_labels)

    accuracy = accuracy_score(labels_array, predictions)
    precision = precision_score(
        labels_array, predictions, zero_division=0
    )
    recall = recall_score(
        labels_array, predictions, zero_division=0
    )
    f1 = f1_score(
        labels_array, predictions, zero_division=0
    )

    try:
        auc = roc_auc_score(labels_array, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }

    return metrics, labels_array, np.array(all_probs)


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("\nLoading five pretrained backbones...")

    model = FiveBackboneFusion().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 70)

        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        with torch.no_grad():
            val_metrics, _, _ = run_epoch(
                model,
                val_loader,
                criterion,
            )

        scheduler.step(val_metrics["loss"])

        print("\nTRAIN:", json.dumps(train_metrics, indent=2))
        print("VALIDATION:", json.dumps(val_metrics, indent=2))
        print("Learning rate:", optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            best_state = copy.deepcopy({
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            })

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "backbone_names": model.names,
                    "class_names": class_names,
                    "real_label": REAL_LABEL,
                    "fake_label": FAKE_LABEL,
                    "image_size": IMAGE_SIZE,
                    "best_validation_loss": best_val_loss,
                },
                MODEL_PATH,
            )

            print("Best model saved:", MODEL_PATH)

        else:
            patience_counter += 1
            print(
                f"No validation improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    # Evaluate the selected best checkpoint, not the final epoch.
    model.load_state_dict(best_state)
    model.to(DEVICE)

    print(f"\nBest epoch: {best_epoch}")
    print("Best validation loss:", best_val_loss)

    with torch.no_grad():
        test_metrics, test_labels, test_probs = run_epoch(
            model,
            test_loader,
            criterion,
        )

    test_predictions = (test_probs >= 0.5).astype(int)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(json.dumps(test_metrics, indent=2))

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test_labels, test_predictions, labels=[0, 1]))

    print("\nCLASSIFICATION REPORT WITH ROC-AUC")
    print("=" * 70)
    
    print(classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["REAL", "DEEPFAKE"],
        zero_division=0,
    ))
    
    real_probs = 1.0 - test_probs
    fake_probs = test_probs
    
    real_auc = roc_auc_score(
        (test_labels == 0).astype(int),
        real_probs,
    )
    
    fake_auc = roc_auc_score(
        (test_labels == 1).astype(int),
        fake_probs,
    )
    
    print(f"REAL ROC-AUC:     {real_auc:.4f} ({real_auc * 100:.2f}%)")
    print(f"DEEPFAKE ROC-AUC: {fake_auc:.4f} ({fake_auc * 100:.2f}%)")
    print(f"OVERALL ROC-AUC:  {test_metrics['roc_auc']:.4f} "
          f"({test_metrics['roc_auc'] * 100:.2f}%)")
    
    real_fpr, real_tpr, _ = roc_curve(
        (test_labels == 0).astype(int),
        real_probs,
    )
    
    fake_fpr, fake_tpr, _ = roc_curve(
        (test_labels == 1).astype(int),
        fake_probs,
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(real_fpr, real_tpr, label=f"REAL (AUC={real_auc:.3f})")
    plt.plot(fake_fpr, fake_tpr, label=f"DEEPFAKE (AUC={fake_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Five-Backbone Fusion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = OUTPUT_DIR / "fusion_roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    
    print("ROC curve saved:", roc_path)

    results = {
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "test_metrics": test_metrics,
        "history": history,
        "dataset_root": str(DATASET_ROOT),
        "class_names": class_names,
        "real_label": REAL_LABEL,
        "fake_label": FAKE_LABEL,
        "train_count": len(train_dataset),
        "validation_count": len(val_dataset),
        "test_count": len(test_dataset),
        "backbones": model.names,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSaved model:", MODEL_PATH)
    print("Saved results:", RESULTS_PATH)
    print("\nTraining and testing completed.")


if __name__ == "__main__":
    main()