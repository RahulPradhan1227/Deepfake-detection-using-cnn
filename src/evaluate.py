
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import models, transforms
from facenet_pytorch import MTCNN

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = Path(
    r"C:\Users\am842\Downloads\archive (1)\ddata"
)

OLD_MODEL = ROOT / "models" / "deepfake_efficientnet_b0_final.pth"
NEW_MODEL = ROOT / "models" / "deepfake_efficientnet_b0_30k_experiment.pth"

OUTPUT = ROOT / "results" / "mtcnn_model_comparison"

IMAGE_SIZE = 224
BATCH_SIZE = 16
CALIBRATION_PER_CLASS = 3000
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("MTCNN + EFFICIENTNET-B0 MODEL COMPARISON")
print("=" * 72)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

for path in (OLD_MODEL, NEW_MODEL):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

for folder in (
    EXTERNAL_ROOT / "train" / "real",
    EXTERNAL_ROOT / "train" / "fake",
    EXTERNAL_ROOT / "test" / "real",
    EXTERNAL_ROOT / "test" / "fake",
):
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

# Same image transform as app.py
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Same MTCNN settings as app.py
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    keep_all=True,
    post_process=False,
    device=device,
)

# ============================================================
# DATASET HELPERS
# REAL = 0, FAKE = 1
# ============================================================

def list_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_samples(root, balanced_count=None, seed=SEED):
    rng = random.Random(seed)
    samples = []

    for class_name, label in (("real", 0), ("fake", 1)):
        paths = list_images(root / class_name)

        if not paths:
            raise RuntimeError(f"No images found in {root / class_name}")

        if balanced_count is not None:
            if len(paths) < balanced_count:
                raise RuntimeError(
                    f"Not enough images in {root / class_name}: "
                    f"{len(paths)} available, {balanced_count} required"
                )
            paths = rng.sample(paths, balanced_count)

        samples.extend((path, label) for path in paths)

    rng.shuffle(samples)
    return samples


# ============================================================
# MODEL LOADING
# ============================================================

def load_checkpoint(path):
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": checkpoint}


def build_model(path, verify_new_model=False):
    checkpoint = load_checkpoint(path)

    if verify_new_model:
        class_names = checkpoint.get("class_names")
        expected = {"0": "REAL", "1": "FAKE"}

        if class_names != expected:
            raise RuntimeError(
                f"Unexpected new-model class mapping: {class_names}. "
                f"Expected {expected}."
            )

        print("Verified new-model mapping:", class_names)

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(model.classifier[1].in_features, 1),
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ============================================================
# MTCNN FACE EXTRACTION
# Matches the app's largest-face logic
# ============================================================

def extract_largest_face(image):
    boxes, probabilities = mtcnn.detect(image)

    if boxes is None:
        return None

    boxes = np.asarray(boxes, dtype=np.float64)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return None

    valid = []

    for index, box in enumerate(boxes):
        if not np.isfinite(box).all():
            continue

        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if area > 0:
            valid.append((area, index, box))

    if not valid:
        return None

    _, _, largest_box = max(valid, key=lambda item: item[0])

    faces = mtcnn.extract(
        image,
        np.asarray([largest_box], dtype=np.float64),
        save_path=None,
    )

    if faces is None:
        return None

    if torch.is_tensor(faces):
        if faces.ndim == 4:
            if faces.shape[0] == 0:
                return None
            face = faces[0]
        elif faces.ndim == 3:
            face = faces
        else:
            return None
    elif isinstance(faces, (list, tuple)):
        if not faces or faces[0] is None:
            return None
        face = faces[0]
    else:
        return None

    if not torch.is_tensor(face) or face.ndim != 3:
        return None

    face = face.detach().cpu()

    if face.shape[0] in (1, 3):
        face = face.permute(1, 2, 0)

    face = face.numpy()

    if not np.isfinite(face).all():
        return None

    if face.max() <= 1.0:
        face = face * 255.0

    face = np.clip(face, 0, 255).astype(np.uint8)
    return Image.fromarray(face).convert("RGB")


# ============================================================
# EVALUATION PIPELINE
# Each image is face-cropped once and shared by all models.
# ============================================================

def evaluate_samples(samples, model_map, description):
    labels = []
    probability_lists = {name: [] for name in model_map}

    no_face = 0
    unreadable = 0
    processing_errors = 0
    batch_images = []
    batch_labels = []

    total = len(samples)
    print(f"\n{description}: {total:,} original images")

    def infer_batch():
        if not batch_images:
            return

        inputs = torch.stack(batch_images).to(device)

        with torch.inference_mode():
            for name, model in model_map.items():
                logits = model(inputs).reshape(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                probability_lists[name].extend(probs.tolist())

        labels.extend(batch_labels)
        batch_images.clear()
        batch_labels.clear()

    for index, (path, label) in enumerate(samples, start=1):
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")

            face = extract_largest_face(image)

            if face is None:
                no_face += 1
                continue

            batch_images.append(image_transform(face))
            batch_labels.append(label)

            if len(batch_images) >= BATCH_SIZE:
                infer_batch()

        except (OSError, ValueError):
            unreadable += 1
        except Exception as exc:
            processing_errors += 1
            if processing_errors <= 5:
                print(f"\nProcessing error for {path}: {exc}")

        if index % 100 == 0 or index == total:
            print(
                f"\rProcessed {index:,}/{total:,} | "
                f"faces: {len(labels) + len(batch_labels):,} | "
                f"no face: {no_face:,} | "
                f"errors: {unreadable + processing_errors:,}",
                end="",
                flush=True,
            )

    infer_batch()
    print()

    valid_labels = np.asarray(labels, dtype=np.int64)
    probabilities = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in probability_lists.items()
    }

    counts = {
        "original_images": int(total),
        "face_detected": int(len(valid_labels)),
        "no_face_detected": int(no_face),
        "unreadable_images": int(unreadable),
        "processing_errors": int(processing_errors),
        "face_detection_coverage": (
            float(len(valid_labels) / total) if total else 0.0
        ),
    }

    print("Face detection coverage: {:.2%}".format(
        counts["face_detection_coverage"]
    ))
    print("Faces evaluated:", len(valid_labels))
    print("No face detected:", no_face)
    print("Unreadable images:", unreadable)
    print("Processing errors:", processing_errors)

    return valid_labels, probabilities, counts


# ============================================================
# CALIBRATION: EXTERNAL TRAIN IMAGES ONLY
# ============================================================

calibration_samples = make_samples(
    EXTERNAL_ROOT / "train",
    balanced_count=CALIBRATION_PER_CLASS,
    seed=SEED,
)

print("\nCalibration images per class:", CALIBRATION_PER_CLASS)

new_model = build_model(NEW_MODEL, verify_new_model=True)

cal_labels, cal_results, cal_counts = evaluate_samples(
    calibration_samples,
    {"new_model": new_model},
    "MTCNN threshold calibration",
)

cal_probs = cal_results["new_model"]

if len(cal_labels) == 0 or len(np.unique(cal_labels)) < 2:
    raise RuntimeError(
        "Calibration did not produce detected faces from both classes. "
        "Cannot safely select a threshold."
    )

precision, recall, thresholds = precision_recall_curve(
    cal_labels, cal_probs
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))
calibrated_threshold = float(thresholds[best_index])
calibration_f1 = float(f1_values[best_index])

print("\nCalibration threshold:", f"{calibrated_threshold:.6f}")
print("Calibration FAKE F1:", f"{calibration_f1:.2%}")

del new_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# EXTERNAL TEST SET
# ============================================================

test_samples = make_samples(EXTERNAL_ROOT / "test")

print("\nTest set REAL:", sum(y == 0 for _, y in test_samples))
print("Test set FAKE:", sum(y == 1 for _, y in test_samples))
print("Test set total:", len(test_samples))

old_model = build_model(OLD_MODEL)
new_model = build_model(NEW_MODEL, verify_new_model=True)

test_labels, test_results, test_counts = evaluate_samples(
    test_samples,
    {
        "old_model": old_model,
        "new_model": new_model,
    },
    "MTCNN external test evaluation",
)

del old_model, new_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if len(test_labels) == 0 or len(np.unique(test_labels)) < 2:
    raise RuntimeError(
        "Test evaluation did not produce detected faces from both classes."
    )

old_probs = test_results["old_model"]
new_probs = test_results["new_model"]


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
        "f1_fake": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "real_correct": int(tn),
        "real_predicted_fake": int(fp),
        "fake_predicted_real": int(fn),
        "fake_correct": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "evaluated_faces": int(len(labels)),
    }


old_05 = metrics_at_threshold(test_labels, old_probs, 0.5)
old_008 = metrics_at_threshold(test_labels, old_probs, 0.008)
new_05 = metrics_at_threshold(test_labels, new_probs, 0.5)
new_cal = metrics_at_threshold(
    test_labels, new_probs, calibrated_threshold
)

results = {
    "method": (
        "MTCNN largest-face extraction followed by the app's "
        "224x224 resize and ImageNet normalization."
    ),
    "class_mapping": {"0": "REAL", "1": "FAKE"},
    "calibration_source": str(EXTERNAL_ROOT / "train"),
    "calibration_per_class_requested": CALIBRATION_PER_CLASS,
    "calibration_counts": cal_counts,
    "calibration_threshold": calibrated_threshold,
    "calibration_fake_f1": calibration_f1,
    "test_source": str(EXTERNAL_ROOT / "test"),
    "test_counts": test_counts,
    "threshold_results": {
        "old_model_0.5": old_05,
        "old_model_0.008": old_008,
        "new_model_0.5": new_05,
        "new_model_calibrated": new_cal,
    },
    "checkpoints": {
        "old_model": str(OLD_MODEL),
        "new_model": str(NEW_MODEL),
    },
}

report_path = OUTPUT / "mtcnn_evaluation_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


# ============================================================
# SAVE RESULT TABLE
# ============================================================

rows = [
    ("Old model / 0.5", old_05),
    ("Old model / 0.008", old_008),
    ("New model / 0.5", new_05),
    (f"New model / {calibrated_threshold:.4f}", new_cal),
]

summary_path = OUTPUT / "mtcnn_evaluation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("MTCNN MODEL COMPARISON\n")
    f.write("=" * 72 + "\n")
    f.write("REAL = 0, FAKE = 1\n")
    f.write(f"Calibration threshold: {calibrated_threshold:.6f}\n")
    f.write(f"Calibration FAKE F1: {calibration_f1:.4%}\n\n")
    f.write(f"Original test images: {test_counts['original_images']}\n")
    f.write(f"Faces evaluated: {test_counts['face_detected']}\n")
    f.write(f"No face detected: {test_counts['no_face_detected']}\n")
    f.write(f"Unreadable: {test_counts['unreadable_images']}\n")
    f.write(f"Processing errors: {test_counts['processing_errors']}\n")
    f.write(
        f"Face coverage: {test_counts['face_detection_coverage']:.2%}\n\n"
    )

    f.write(
        f"{'Model / threshold':<28}{'Accuracy':>11}"
        f"{'Precision':>12}{'Recall':>10}{'F1':>10}{'AUC':>10}\n"
    )

    for name, m in rows:
        f.write(
            f"{name:<28}{m['accuracy']:>10.2%}"
            f"{m['precision_fake']:>11.2%}"
            f"{m['recall_fake']:>9.2%}"
            f"{m['f1_fake']:>9.2%}"
            f"{m['roc_auc']:>9.2%}\n"
        )
        f.write(f"  Confusion matrix: {m['confusion_matrix']}\n")

print("\n" + "=" * 90)
print("MTCNN EXTERNAL TEST COMPARISON")
print("=" * 90)
print(
    f"{'Model / threshold':<28}{'Accuracy':>11}"
    f"{'Precision':>12}{'Recall':>10}{'F1':>10}{'AUC':>10}"
)
print("-" * 90)

for name, m in rows:
    print(
        f"{name:<28}{m['accuracy']:>10.2%}"
        f"{m['precision_fake']:>11.2%}"
        f"{m['recall_fake']:>9.2%}"
        f"{m['f1_fake']:>9.2%}"
        f"{m['roc_auc']:>9.2%}"
    )


# ============================================================
# CONFUSION MATRIX PLOTS
# ============================================================

def save_confusion_plot(metrics, title, filename):
    cm = np.asarray(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title,
    )

    midpoint = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > midpoint else "black",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=160)
    plt.close(fig)


save_confusion_plot(
    new_cal,
    f"New Model Confusion Matrix (threshold {calibrated_threshold:.4f})",
    "new_model_confusion_matrix.png",
)

save_confusion_plot(
    old_008,
    "Old Model Confusion Matrix (threshold 0.008)",
    "old_model_confusion_matrix.png",
)


# ============================================================
# ROC CURVE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

for name, probabilities in (
    ("Old EfficientNet-B0", old_probs),
    ("New EfficientNet-B0", new_probs),
):
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    auc_value = roc_auc_score(test_labels, probabilities)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.4f})")

ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="ROC Curve — MTCNN Face Pipeline",
)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "roc_curve.png", dpi=160)
plt.close(fig)

print("\nSaved results folder:", OUTPUT)
print("Report:", report_path)
print("Summary:", summary_path)
print("Graphs: confusion matrices and ROC curve")
print("\nEvaluation completed. Model checkpoints were not modified.")
