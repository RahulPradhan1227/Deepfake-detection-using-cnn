import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load test dataset
data = np.load(
    r"C:\Users\susha\deepfake detector\normalized_dataset.npz"
)

X_test = data["X_test"]
y_test = data["y_test"]

# Load trained model
model = tf.keras.models.load_model(
    r"C:\Users\susha\deepfake detector\best_model.keras"
)

# Predict probabilities
predictions = model.predict(X_test)

# Convert probabilities into labels
y_pred = (predictions >= 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Real", "Fake"]
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPrediction distribution:")
print("Predicted Real:", np.sum(y_pred == 0))
print("Predicted Fake:", np.sum(y_pred == 1))

print("\nActual distribution:")
print("Actual Real:", np.sum(y_test == 0))
print("Actual Fake:", np.sum(y_test == 1))

wrong_indices = np.where(y_pred != y_test)[0]

print("\nFirst 10 incorrect predictions:")
for i in wrong_indices[:10]:
    print(
        f"Index: {i}, "
        f"Actual: {'Fake' if y_test[i] == 1 else 'Real'}, "
        f"Predicted: {'Fake' if y_pred[i] == 1 else 'Real'}, "
        f"Confidence: {predictions[i][0]:.4f}"
    )