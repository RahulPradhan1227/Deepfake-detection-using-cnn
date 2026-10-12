import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load normalized dataset (values are between 0 and 1)
data = np.load(
    r"C:\Users\susha\deepfake detector\normalized_dataset.npz"
)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

# Data augmentation
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# Load pre-trained InceptionV3
base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze pre-trained layers
base_model.trainable = False

# Build model
inputs = layers.Input(shape=(128, 128, 3))

x = augmentation(inputs)

# Convert [0, 1] pixels to [0, 255], then apply InceptionV3 preprocessing
x = layers.Rescaling(scale=2.0, offset=-1.0)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    r"C:\Users\susha\deepfake detector\inceptionv3_best.keras",
    monitor="val_loss",
    save_best_only=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=16,
    callbacks=[early_stopping, checkpoint]
)

model.save(
    r"C:\Users\susha\deepfake detector\inceptionv3_final.keras"
)

print("InceptionV3 training completed!")