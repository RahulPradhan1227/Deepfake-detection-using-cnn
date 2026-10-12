import json
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
REAL_DIR = PROJECT_ROOT / "dataset" / "real"
FAKE_DIR = PROJECT_ROOT / "dataset" / "fake"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = RESULTS_DIR / "data_split.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

real_images = sorted(
    [
        str(p)
        for p in REAL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
)

fake_images = sorted(
    [
        str(p)
        for p in FAKE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
)

paths = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

split_data = {
    "train": [
        {"path": p, "label": int(y)}
        for p, y in zip(train_paths, train_labels)
    ],
    "validation": [
        {"path": p, "label": int(y)}
        for p, y in zip(val_paths, val_labels)
    ],
    "test": [
        {"path": p, "label": int(y)}
        for p, y in zip(test_paths, test_labels)
    ]
}

with open(SPLIT_FILE, "w", encoding="utf-8") as f:
    json.dump(split_data, f, indent=4)

print("=" * 70)
print("DATA SPLIT CREATED")
print("=" * 70)
print(f"Real images       : {len(real_images)}")
print(f"Fake images       : {len(fake_images)}")
print(f"Total images      : {len(paths)}")
print()
print(f"Training images   : {len(train_paths)}")
print(f"Validation images : {len(val_paths)}")
print(f"Test images       : {len(test_paths)}")
print()
print(f"Saved to          : {SPLIT_FILE}")
