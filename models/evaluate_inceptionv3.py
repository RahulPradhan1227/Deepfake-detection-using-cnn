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

# Load trained InceptionV3 model
model = tf.keras.models.load_model(
    r"C:\Users\susha\deepfake detector\inceptionv3_best.keras"
)

# Predict
predictions = model.predict(X_test)
y_pred = (predictions >= 0.5).astype(int).flatten()

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n========== INCEPTIONV3 EVALUATION ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Real", "Fake"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))