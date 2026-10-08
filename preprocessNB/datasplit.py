
import numpy as np
from sklearn.model_selection import train_test_split

# Import dataset
import numpy as np

data = np.load(r"C:\Users\susha\deepfake detector\dataset.npz")
X = data["X"]
y = data["y"]

print("Dataset loaded:", X.shape, y.shape)

print("Original Dataset")
print("X shape:", X.shape)
print("y shape:", y.shape)

# First Split: 70% Training, 30% Remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second Split: 15% Validation, 15% Testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Print dataset information
print("\n========== DATA SPLITTING COMPLETE ==========")

print("\nTraining Dataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation Dataset:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting Dataset:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Class distribution
print("\n========== CLASS DISTRIBUTION ==========")

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:
    real = np.sum(labels == 0)
    fake = np.sum(labels == 1)

    print(f"\n{name}:")
    print("Real:", real)
    print("Fake:", fake)
    print("Total:", len(labels))

print("\nData splitting successful!")
# Save split datasets
output_path = r"C:\Users\susha\deepfake detector\split_dataset.npz"

np.savez_compressed(
    output_path,
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)

print("Split dataset saved at:", output_path)