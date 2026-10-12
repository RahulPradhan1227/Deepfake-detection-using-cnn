import cv2
import numpy as np
from pathlib import Path

# Image folder paths
REAL_DIR = Path(
    r"C:\Users\susha\Downloads\archive\real_and_fake_face_detection\real_and_fake_face\training_real"
)

FAKE_DIR = Path(
    r"C:\Users\susha\Downloads\archive\real_and_fake_face_detection\real_and_fake_face\training_fake"
)

IMG_SIZE = 128

X = []
y = []

# Supported image formats
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_images(folder, label):
    count = 0

    # Check whether the folder exists
    if not folder.exists():
        print("Folder not found:", folder)
        return

    for file in folder.rglob("*"):

        if file.suffix.lower() not in EXTENSIONS:
            continue

        image = cv2.imread(str(file))

        if image is None:
            print("Skipped:", file.name)
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        X.append(image)
        y.append(label)

        count += 1

    print(f"{folder.name}: {count} images loaded")


# Load images: 0 = Real, 1 = Fake
load_images(REAL_DIR, 0)
load_images(FAKE_DIR, 1)

# Convert to NumPy arrays
X = np.array(X, dtype=np.uint8)
y = np.array(y, dtype=np.int32)

# Display results
print("\n========== DATASET LOADING COMPLETE ==========")

print("Total images:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Real images:", np.sum(y == 0))
print("Fake images:", np.sum(y == 1))
np.savez(
    r"C:\Users\susha\deepfake detector\dataset.npz",
    X=X,
    y=y
)
print("Dataset saved successfully!")