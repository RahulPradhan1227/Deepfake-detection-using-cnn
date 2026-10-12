
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = Path(r"C:\Users\am842\Downloads\archive (1)\ddata")

OLD_MODEL = ROOT / "models" / "deepfake_efficientnet_b0_final.pth"
NEW_MODEL = ROOT / "models" / "deepfake_efficientnet_b0_30k_experiment.pth"

OUTPUT = ROOT / "results" / "external_threshold_calibration"

IMAGE_SIZE = 224
BATCH_SIZE = 16
CALIBRATION_PER_CLASS = 3000
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("NEW MODEL THRESHOLD CALIBRATION")
print("=" * 72)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

for path in [OLD_MODEL, NEW_MODEL]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

for folder in [
    EXTERNAL_ROOT / "train" / "real",
    EXTERNAL_ROOT / "train" / "fake",
    EXTERNAL_ROOT / "test" / "real",
    EXTERNAL_ROOT / "test" / "fake",
]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

OUTPUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def list_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_samples(root, balanced_count=None, seed=SEED):
    samples = []
    rng = random.Random(seed)

    for class_name, label in [("real", 0), ("fake", 1)]:
        paths = list_images(root / class_name)

        if not paths:
            raise RuntimeError(f"No images found in {root / class_name}")

        if balanced_count is not None:
            if len(paths) < balanced_count:
                raise RuntimeError(
                    f"Only {len(paths)} images in {root / class_name}; "
                    f"need {balanced_count}"
                )
            paths = rng.sample(paths, balanced_count)

        samples.extend((path, label) for path in paths)

    rng.shuffle(samples)
    return samples


class ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        try:
            with Image.open(path) as im:
                image = self_transform(im.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Cannot read image: {path}") from exc

        return image, label


# Keep the transform shared and consistent for both models.
self_transform = transform


def build_model(checkpoint_path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def infer(model, samples, description):
    dataset = ImageDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    probabilities = []
    labels = []

    print(f"\n{description}: {len(dataset)} images")

    with torch.inference_mode():
        for i, (images, batch_labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            logits = model(images).reshape(-1)
            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())

            if i % 100 == 0 or i == len(loader):
                print(f"\rBatches: {i}/{len(loader)}", end="", flush=True)

    print()
    return np.asarray(labels, dtype=np.int64), np.asarray(
        probabilities, dtype=np.float64
    )


# ============================================================
# CALIBRATION DATA: EXTERNAL TRAIN ONLY
# ============================================================

calibration_samples = make_samples(
    EXTERNAL_ROOT / "train",
    balanced_count=CALIBRATION_PER_CLASS,
    seed=SEED,
)

print("\nCalibration sample:")
print("REAL:", sum(label == 0 for _, label in calibration_samples))
print("FAKE:", sum(label == 1 for _, label in calibration_samples))

# Load the new model and calibrate its threshold on training data only.
new_model = build_model(NEW_MODEL)
cal_labels, cal_probs = infer(
    new_model,
    calibration_samples,
    "Calibrating new model on external TRAIN sample",
)

precision, recall, thresholds = precision_recall_curve(cal_labels, cal_probs)

# precision_recall_curve returns one more precision/recall entry than thresholds.
f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))
calibrated_threshold = float(thresholds[best_index])
calibration_best_f1 = float(f1_values[best_index])

print("\nThreshold selected using calibration data only:")
print(f"New model threshold: {calibrated_threshold:.6f}")
print(f"Calibration FAKE F1: {calibration_best_f1:.4%}")

del new_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# EXTERNAL TEST DATA: NOT USED FOR THRESHOLD SELECTION
# ============================================================

test_samples = make_samples(EXTERNAL_ROOT / "test")

print("\nExternal test set:")
print("REAL:", sum(label == 0 for _, label in test_samples))
print("FAKE:", sum(label == 1 for _, label in test_samples))
print("TOTAL:", len(test_samples))

# Evaluate both models on exactly the same test images.
old_model = build_model(OLD_MODEL)
test_labels, old_probs = infer(old_model, test_samples, "Testing old model")
del old_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()

new_model = build_model(NEW_MODEL)
test_labels_new, new_probs = infer(new_model, test_samples, "Testing new model")
del new_model

if not np.array_equal(test_labels, test_labels_new):
    raise RuntimeError("Model test labels do not match.")

# ============================================================
# METRICS
# ============================================================

def metrics_at_threshold(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_fake": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_fake": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1_fake": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "true_real_predicted_real": int(tn),
        "real_predicted_fake": int(fp),
        "fake_predicted_real": int(fn),
        "true_fake_predicted_fake": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "total_images": int(len(labels)),
    }


results = {
    "calibration_source": str(EXTERNAL_ROOT / "train"),
    "calibration_images_per_class": CALIBRATION_PER_CLASS,
    "calibration_seed": SEED,
    "selected_new_model_threshold": calibrated_threshold,
    "calibration_best_fake_f1": calibration_best_f1,
    "test_source": str(EXTERNAL_ROOT / "test"),
    "test_images": int(len(test_labels)),
    "models": {
        "old_model": {
            "checkpoint": str(OLD_MODEL),
            "threshold_results": {
                "0.5": metrics_at_threshold(test_labels, old_probs, 0.5),
                "0.008": metrics_at_threshold(test_labels, old_probs, 0.008),
            },
        },
        "new_model": {
            "checkpoint": str(NEW_MODEL),
            "threshold_results": {
                "0.5": metrics_at_threshold(test_labels, new_probs, 0.5),
                "calibrated": metrics_at_threshold(
                    test_labels, new_probs, calibrated_threshold
                ),
            },
        },
    },
}

# Save report in a new folder, without overwriting previous comparison results.
report_path = OUTPUT / "threshold_calibration_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

summary_path = OUTPUT / "threshold_calibration_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("EXTERNAL TEST: THRESHOLD CALIBRATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"Calibration threshold: {calibrated_threshold:.6f}\n")
    f.write(f"Calibration best FAKE F1: {calibration_best_f1:.6f}\n")
    f.write(f"External test images: {len(test_labels)}\n\n")

    for model_name, model_data in results["models"].items():
        f.write(f"\n{model_name.upper()}\n")
        f.write("-" * 40 + "\n")

        for threshold_name, m in model_data["threshold_results"].items():
            f.write(f"\nThreshold: {threshold_name}\n")
            for key in [
                "accuracy", "precision_fake", "recall_fake",
                "f1_fake", "roc_auc", "fake_predicted_real",
                "real_predicted_fake",
            ]:
                f.write(f"{key}: {m[key]}\n")
            f.write(f"Confusion matrix: {m['confusion_matrix']}\n")

# ============================================================
# FINAL SUMMARY
# ============================================================

old_05 = results["models"]["old_model"]["threshold_results"]["0.5"]
old_008 = results["models"]["old_model"]["threshold_results"]["0.008"]
new_05 = results["models"]["new_model"]["threshold_results"]["0.5"]
new_cal = results["models"]["new_model"]["threshold_results"]["calibrated"]

print("\n" + "=" * 78)
print("EXTERNAL TEST COMPARISON")
print("=" * 78)
print(f"{'Model / threshold':<28}{'Accuracy':>11}{'FAKE recall':>13}{'FAKE F1':>11}{'AUC':>10}")
print("-" * 78)

rows = [
    ("Old model / 0.5", old_05),
    ("Old model / 0.008", old_008),
    ("New model / 0.5", new_05),
    (f"New model / {calibrated_threshold:.4f}", new_cal),
]

for name, m in rows:
    print(
        f"{name:<28}{m['accuracy']:>10.2%}"
        f"{m['recall_fake']:>12.2%}"
        f"{m['f1_fake']:>10.2%}"
        f"{m['roc_auc']:>10.2%}"
    )
    print(
        f"    FAKE predicted REAL: {m['fake_predicted_real']} | "
        f"REAL predicted FAKE: {m['real_predicted_fake']}"
    )

print("\nSaved report:", report_path)
print("Saved summary:", summary_path)
print("\nDone. No model checkpoints or app files were changed.")
