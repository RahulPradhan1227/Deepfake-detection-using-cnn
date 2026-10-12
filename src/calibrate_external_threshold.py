import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "deepfake_efficientnet_b0_final.pth"
TRAIN_DIR = Path(r"C:\Users\am842\Downloads\archive (1)\ddata\train")
RESULTS_DIR = ROOT / "results" / "external_evaluation"
TEST_PREDICTIONS = RESULTS_DIR / "predictions.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_PER_CLASS = 5000
SEED = 42
DEFAULT_THRESHOLD = 0.50

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_samples():
    rng = random.Random(SEED)
    samples = []

    for class_name, label in [("real", 0), ("fake", 1)]:
        folder = TRAIN_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if len(files) > MAX_PER_CLASS:
            files = rng.sample(files, MAX_PER_CLASS)

        samples.extend((p, label) for p in files)
        print(f"Calibration {class_name.upper()} images: {len(files):,}")

    rng.shuffle(samples)
    return samples


class CalibrationDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                image.load()
            return self.transform(image), label, str(path)
        except Exception as exc:
            raise RuntimeError(f"Cannot read image {path}: {exc}") from exc


def calculate_metrics(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_fake": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_REAL_FAKE": cm.tolist(),
        "real_misclassified_as_fake": int(cm[0, 1]),
        "fake_misclassified_as_real": int(cm[1, 0]),
    }


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    if not TEST_PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"Missing {TEST_PREDICTIONS}. Run evaluate_external.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the already recorded external test predictions.
    test_labels = []
    test_probabilities = []

    with open(TEST_PREDICTIONS, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            test_labels.append(1 if row["true_label"] == "FAKE" else 0)
            test_probabilities.append(float(row["fake_probability"]))

    y_test = np.asarray(test_labels, dtype=int)
    p_test = np.asarray(test_probabilities, dtype=float)

    if len(y_test) == 0:
        raise RuntimeError("No external test predictions were found.")

    print("=" * 70)
    print("THRESHOLD CALIBRATION")
    print("=" * 70)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Calibration source:", TRAIN_DIR)
    print("Test set: existing predictions.csv (test folder only)")
    print("Test images:", f"{len(y_test):,}")
    print("The test set will NOT be used to select the threshold.\n")

    samples = collect_samples()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    loader = DataLoader(
        CalibrationDataset(samples, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    try:
        checkpoint = torch.load(
            MODEL_PATH, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    cal_labels = []
    cal_probabilities = []

    with torch.inference_mode():
        for batch_num, (images, labels, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images).squeeze(1))
            cal_labels.extend(labels.numpy().astype(int).tolist())
            cal_probabilities.extend(probs.cpu().numpy().tolist())

            if batch_num % 50 == 0 or batch_num == len(loader):
                print(
                    f"Calibration batches: {batch_num:,}/{len(loader):,}"
                )

    y_cal = np.asarray(cal_labels, dtype=int)
    p_cal = np.asarray(cal_probabilities, dtype=float)

    # Pick threshold using calibration data only: maximize FAKE F1.
    # If F1 ties, prefer higher fake recall.
    candidates = np.round(np.arange(0.001, 0.201, 0.001), 3)
    candidate_results = [
        calculate_metrics(y_cal, p_cal, t) for t in candidates
    ]
    best = max(
        candidate_results,
        key=lambda r: (r["f1_fake"], r["recall_fake"]),
    )
    chosen_threshold = best["threshold"]

    # Now apply the already-selected threshold to the untouched test results.
    default_test = calculate_metrics(
        y_test, p_test, DEFAULT_THRESHOLD
    )
    calibrated_test = calculate_metrics(
        y_test, p_test, chosen_threshold
    )

    try:
        test_auc = float(roc_auc_score(y_test, p_test))
    except ValueError:
        test_auc = None

    # Save the calibration threshold search, not just the winning row.
    with open(
        RESULTS_DIR / "threshold_comparison.csv",
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold", "accuracy", "precision_fake",
                "recall_fake", "f1_fake",
                "real_misclassified_as_fake",
                "fake_misclassified_as_real",
            ],
        )
        writer.writeheader()
        for result in candidate_results:
            writer.writerow({
                k: result[k] for k in writer.fieldnames
            })

    summary = {
        "checkpoint": str(MODEL_PATH),
        "calibration_source": str(TRAIN_DIR),
        "calibration_images_used": int(len(y_cal)),
        "calibration_real_images": int(np.sum(y_cal == 0)),
        "calibration_fake_images": int(np.sum(y_cal == 1)),
        "threshold_selection_rule": "maximize FAKE F1; break ties by higher FAKE recall",
        "selected_threshold": chosen_threshold,
        "selected_calibration_metrics": best,
        "external_test_images": int(len(y_test)),
        "external_test_roc_auc": test_auc,
        "external_test_at_threshold_0.50": default_test,
        "external_test_at_calibrated_threshold": calibrated_test,
        "note": (
            "Threshold was selected using the calibration subset of the train folder. "
            "The test folder was not used to select it. The model checkpoint was not changed."
        ),
    }

    with open(
        RESULTS_DIR / "calibration_summary.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"Selected threshold: {chosen_threshold:.2f}")
    print(
        f"Calibration F1 (FAKE): {best['f1_fake']:.4f} | "
        f"Recall (FAKE): {best['recall_fake']:.4f}"
    )

    for title, result in [
        ("EXTERNAL TEST AT DEFAULT THRESHOLD 0.50", default_test),
        ("EXTERNAL TEST AT CALIBRATED THRESHOLD", calibrated_test),
    ]:
        print(f"\n{title}")
        print("-" * 55)
        print(f"Threshold:         {result['threshold']:.2f}")
        print(f"Accuracy:          {result['accuracy']:.4%}")
        print(f"Precision (FAKE):  {result['precision_fake']:.4f}")
        print(f"Recall (FAKE):     {result['recall_fake']:.4f}")
        print(f"F1-score (FAKE):   {result['f1_fake']:.4f}")
        print("Confusion matrix (true rows; REAL, FAKE):")
        print(np.array(result["confusion_matrix_REAL_FAKE"]))
        print(
            "Fake images predicted REAL:",
            result["fake_misclassified_as_real"],
        )
        print(
            "Real images predicted FAKE:",
            result["real_misclassified_as_fake"],
        )

    print("\nROC-AUC (threshold-independent):", test_auc)
    print("Saved:", RESULTS_DIR / "threshold_comparison.csv")
    print("Saved:", RESULTS_DIR / "calibration_summary.json")
    print("Checkpoint unchanged.")


if __name__ == "__main__":
    main()

