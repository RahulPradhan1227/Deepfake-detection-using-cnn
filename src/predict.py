import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# THIS IS THE MODEL THAT ACHIEVED 75.98% TEST ACCURACY
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deepfake_efficientnet_b0_final.pth"
)

IMAGE_SIZE = 224

# Optimized on validation set
THRESHOLD = 0.48


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEEPFAKE IMAGE DETECTION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("\nERROR: Model file not found:")
    print(MODEL_PATH)
    sys.exit(1)


# ============================================================
# CREATE EXACT EFFICIENTNET-B0 ARCHITECTURE
# ============================================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier = nn.Sequential(

    nn.Dropout(
        p=0.30
    ),

    nn.Linear(
        1280,
        1
    )
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading EfficientNet-B0...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")

print(
    f"Model: EfficientNet-B0"
)

print(
    f"Threshold: {THRESHOLD}"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

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


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        print("\nERROR: Image not found:")
        print(image_path)

        return


    print("\n" + "=" * 70)
    print("IMAGE ANALYSIS")
    print("=" * 70)

    print(
        f"\nImage: {image_path.name}"
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as e:

        print("\nERROR: Could not open image.")
        print(e)

        return


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_tensor = transform(
        image
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        fake_probability = torch.sigmoid(
            output
        ).item()


    real_probability = (
        1.0 - fake_probability
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if fake_probability >= THRESHOLD:

        prediction = "DEEPFAKE"

        confidence = fake_probability

    else:

        prediction = "REAL"

        confidence = real_probability


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"\nPrediction : {prediction}"
    )

    print(
        f"\nREAL       : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"DEEPFAKE   : "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"\nConfidence : "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Threshold  : "
        f"{THRESHOLD:.2f}"
    )

    print("\n" + "=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")

    print(
        'python src\\predict.py '
        '"C:\\path\\to\\image.jpg"'
    )

    print("\nExample:")

    print(
        'python src\\predict.py '
        '"C:\\Users\\am842\\Pictures\\test.jpg"'
    )

    sys.exit(1)
    
    


IMAGE_PATH = sys.argv[1]

predict_image(
    IMAGE_PATH
)