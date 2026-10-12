# ============================================================
# Deepfake Detection - MTCNN Face Preprocessing
# ============================================================
# This script:
# 1. Reads the original REAL and FAKE images.
# 2. Detects faces using MTCNN.
# 3. Selects the largest face in each image.
# 4. Crops and resizes the face to 224 x 224.
# 5. Saves the processed images into:
#
#       dataset/
#       ├── real/
#       └── fake/
#
# MTCNN will use the NVIDIA GPU if CUDA is available.
# ============================================================

from pathlib import Path
import shutil

import torch
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN
from tqdm import tqdm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original dataset - DO NOT MODIFY
ORIGINAL_DATASET = Path(
    r"C:\Users\am842\Downloads\archive\real_and_fake_face"
)

ORIGINAL_REAL = ORIGINAL_DATASET / "training_real"
ORIGINAL_FAKE = ORIGINAL_DATASET / "training_fake"

# Processed dataset
OUTPUT_REAL = PROJECT_ROOT / "dataset" / "real"
OUTPUT_FAKE = PROJECT_ROOT / "dataset" / "fake"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224

# Minimum face size that MTCNN will detect
MIN_FACE_SIZE = 40

# Extra pixels around detected face
MARGIN = 20


# ============================================================
# 3. SELECT DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("\nGPU detected:")
    print(torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print("\nCUDA not available.")
    print("Using CPU.")


# ============================================================
# 4. CREATE MTCNN
# ============================================================

print("\nLoading MTCNN...")

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=MARGIN,
    min_face_size=MIN_FACE_SIZE,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    keep_all=True,
    post_process=False,
    device=DEVICE,
)

print("MTCNN loaded successfully.")


# ============================================================
# 5. PREPARE OUTPUT FOLDERS
# ============================================================

OUTPUT_REAL.mkdir(parents=True, exist_ok=True)
OUTPUT_FAKE.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 7. PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, output_path):
    """
    Detect the largest face using MTCNN and save the crop.

    Returns:
        "processed"  -> face found and saved
        "no_face"    -> no face detected
        "unreadable" -> image could not be opened
        "error"      -> unexpected processing error
    """

    try:
        # Open image
        image = Image.open(image_path).convert("RGB")

        # Detect all faces
        boxes, probabilities = mtcnn.detect(image)

        # No face detected
        if boxes is None or len(boxes) == 0:
            return "no_face"

        # ----------------------------------------------------
        # Select the largest detected face
        # ----------------------------------------------------

        largest_index = 0
        largest_area = 0

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            if area > largest_area:
                largest_area = area
                largest_index = i

        # Get largest face coordinates
        x1, y1, x2, y2 = boxes[largest_index]

        # ----------------------------------------------------
        # Add margin around face
        # ----------------------------------------------------

        x1 = max(0, int(x1) - MARGIN)
        y1 = max(0, int(y1) - MARGIN)
        x2 = min(image.width, int(x2) + MARGIN)
        y2 = min(image.height, int(y2) + MARGIN)

        # Crop face
        face = image.crop((x1, y1, x2, y2))

        # Resize to 224 x 224
        face = face.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # Save as JPEG
        face.save(
            output_path,
            format="JPEG",
            quality=95
        )

        return "processed"

    except (UnidentifiedImageError, OSError):
        return "unreadable"

    except Exception as e:
        print(f"\nError processing {image_path.name}: {e}")
        return "error"


# ============================================================
# 8. PROCESS DATASET
# ============================================================

def process_dataset(source_folder, output_folder, class_name):
    """
    Process all images in one class.
    """

    image_files = [
        file
        for file in source_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]

    processed = 0
    no_face = 0
    unreadable = 0
    errors = 0

    print(f"\n{'=' * 60}")
    print(f"Processing {class_name} images")
    print(f"Source : {source_folder}")
    print(f"Output : {output_folder}")
    print(f"Images : {len(image_files)}")
    print(f"{'=' * 60}")

    for index, image_path in enumerate(
        tqdm(image_files, desc=class_name),
        start=1
    ):

        # Create a unique output filename
        output_name = f"{class_name.lower()}_{index:05d}.jpg"

        output_path = output_folder / output_name

        result = process_image(
            image_path,
            output_path
        )

        if result == "processed":
            processed += 1

        elif result == "no_face":
            no_face += 1

        elif result == "unreadable":
            unreadable += 1

        elif result == "error":
            errors += 1

    print(f"\n{class_name} Summary")
    print(f"Total images      : {len(image_files)}")
    print(f"Faces processed   : {processed}")
    print(f"No face detected  : {no_face}")
    print(f"Unreadable images : {unreadable}")
    print(f"Other errors      : {errors}")

    return {
        "total": len(image_files),
        "processed": processed,
        "no_face": no_face,
        "unreadable": unreadable,
        "errors": errors,
    }


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION - MTCNN FACE PREPROCESSING")
    print("=" * 60)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nOriginal dataset:")
    print(ORIGINAL_DATASET)

    print(f"\nOutput dataset:")
    print(PROJECT_ROOT / "dataset")

    print(f"\nImage size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    # --------------------------------------------------------
    # Check original dataset
    # --------------------------------------------------------

    if not ORIGINAL_REAL.exists():
        print("\nERROR: REAL dataset folder not found:")
        print(ORIGINAL_REAL)
        return

    if not ORIGINAL_FAKE.exists():
        print("\nERROR: FAKE dataset folder not found:")
        print(ORIGINAL_FAKE)
        return

    # --------------------------------------------------------
    # Process REAL images
    # --------------------------------------------------------

    real_stats = process_dataset(
        ORIGINAL_REAL,
        OUTPUT_REAL,
        "REAL"
    )

    # --------------------------------------------------------
    # Process FAKE images
    # --------------------------------------------------------

    fake_stats = process_dataset(
        ORIGINAL_FAKE,
        OUTPUT_FAKE,
        "FAKE"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print("\nREAL:")
    print(f"  Original images : {real_stats['total']}")
    print(f"  Faces saved     : {real_stats['processed']}")
    print(f"  No face         : {real_stats['no_face']}")
    print(f"  Unreadable      : {real_stats['unreadable']}")

    print("\nFAKE:")
    print(f"  Original images : {fake_stats['total']}")
    print(f"  Faces saved     : {fake_stats['processed']}")
    print(f"  No face         : {fake_stats['no_face']}")
    print(f"  Unreadable      : {fake_stats['unreadable']}")

    total_original = (
        real_stats["total"] +
        fake_stats["total"]
    )

    total_processed = (
        real_stats["processed"] +
        fake_stats["processed"]
    )

    print("\nTOTAL:")
    print(f"  Original images : {total_original}")
    print(f"  Faces saved     : {total_processed}")

    print("\nOutput folders:")
    print(f"  REAL: {OUTPUT_REAL}")
    print(f"  FAKE: {OUTPUT_FAKE}")

    print("\nNext step will be PyTorch EfficientNet-B0 training.")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()