import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the split dataset
data = np.load(
    r"C:\Users\susha\deepfake detector\split_dataset.npz"
)

print("Available keys:", data.files)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

# Normalization
# Convert pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Data augmentation (training images only)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# Build augmentation layers using input shape
data_augmentation.build((None, 128, 128, 3))

# Display model summary
data_augmentation.summary()
  
# Test augmentation on one training image
sample_image = X_train[:1]

augmented_image = data_augmentation(
    sample_image,
    training=True
)

print("Original image shape:", sample_image.shape)
print("Augmented image shape:", augmented_image.shape)

# Display dataset information
print("========== PREPROCESSING COMPLETE ==========")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

print("\nPixel value range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())

print("\nData augmentation:")
print(data_augmentation.summary())

# Save normalized datasets
np.savez_compressed(
    r"C:\Users\susha\deepfake detector\normalized_dataset.npz",
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)

print("\nNormalized dataset saved successfully!")