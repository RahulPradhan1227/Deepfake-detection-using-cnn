import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

tf.keras.utils.set_random_seed(42)

# ============================================================
# SETTINGS
# ============================================================
DATA_PATH   = "normalized_dataset.npz"
SRC_SIZE    = 128      # size stored in your npz
MODEL_SIZE  = 160      # resized inside the model (224 is better if you have a GPU)
BATCH_SIZE  = 16
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 30

# ============================================================
# LOAD DATA
# ============================================================
d = np.load(DATA_PATH)
X_train, y_train = d["X_train"].astype("float32"), d["y_train"].astype("float32")
X_val,   y_val   = d["X_val"].astype("float32"),   d["y_val"].astype("float32")
X_test,  y_test  = d["X_test"].astype("float32"),  d["y_test"].astype("float32")

# ConvNeXt expects raw pixels in [0, 255]. Detect the range and convert.
lo, hi = X_train.min(), X_train.max()
print("Pixel range:", lo, "to", hi)
if hi <= 1.0 and lo >= 0.0:        # [0, 1]
    scale = lambda a: a * 255.0
elif lo < 0.0 and hi <= 1.0:       # [-1, 1]
    scale = lambda a: (a + 1.0) * 127.5
elif hi > 1.0 and lo >= 0.0:       # already 0-255
    scale = lambda a: a
else:                               # mean/std normalized: cannot be recovered reliably
    raise ValueError("Data looks mean/std normalized. Rebuild the npz as raw 0-255 uint8.")
X_train, X_val, X_test = scale(X_train), scale(X_val), scale(X_test)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Train class balance (mean label):", y_train.mean())

# Class weights to counter the Real/Fake bias
cw = compute_class_weight("balanced", classes=np.array([0., 1.]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}
print("Class weights:", class_weight)

# ============================================================
# tf.data PIPELINE (shuffles every epoch, keeps order for val/test)
# ============================================================
def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds   = make_ds(X_val, y_val)
test_ds  = make_ds(X_test, y_test)

# ============================================================
# MODEL
# ============================================================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(0.1, value_range=(0, 255)),
    layers.RandomContrast(0.1),
], name="augmentation")

base = ConvNeXtTiny(include_top=False, weights="imagenet",
                    input_shape=(MODEL_SIZE, MODEL_SIZE, 3))

inputs = layers.Input(shape=(SRC_SIZE, SRC_SIZE, 3))
x = layers.Resizing(MODEL_SIZE, MODEL_SIZE)(inputs)   # match ImageNet scale
x = augmentation(x)                                   # only active in training
x = base(x)                                           # ConvNeXt normalizes internally
x = layers.GlobalAveragePooling2D()(x)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="gelu",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs, name="Deepfake_ConvNeXt")

metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

def make_callbacks():
    return [
        EarlyStopping(monitor="val_auc", mode="max", patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

# ============================================================
# STAGE 1: warm up the head only
# ============================================================
base.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS,
          class_weight=class_weight, callbacks=make_callbacks())

# ============================================================
# STAGE 2: fine-tune the WHOLE network at a low LR
# ============================================================
base.trainable = True
model.compile(tf.keras.optimizers.AdamW(learning_rate=3e-5, weight_decay=1e-4),
              loss=loss, metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS,
          class_weight=class_weight, callbacks=make_callbacks())

# ============================================================
# PREDICT with flip test-time augmentation
# ============================================================
def predict_tta(X):
    p1 = model.predict(X, batch_size=BATCH_SIZE, verbose=0).ravel()
    p2 = model.predict(X[:, :, ::-1, :], batch_size=BATCH_SIZE, verbose=0).ravel()
    return (p1 + p2) / 2.0

val_probs  = predict_tta(X_val)
test_probs = predict_tta(X_test)

# Choose the threshold on VALIDATION data (never on test)
fpr, tpr, thr = roc_curve(y_val, val_probs)
best_thr = thr[(tpr - fpr).argmax()]
print(f"\nBest threshold from validation: {best_thr:.3f}")

# ============================================================
# EVALUATION
# ============================================================
for name, t in [("Default 0.5", 0.5), ("Tuned", best_thr)]:
    preds = (test_probs > t).astype(int)
    acc = (preds == y_test).mean() * 100
    print(f"\n===== {name} threshold ({t:.3f}) | Test Accuracy: {acc:.2f}% =====")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))
    print(confusion_matrix(y_test, preds))

print("\nTest AUC:", round(roc_auc_score(y_test, test_probs), 4))
model.save("convnext_final.keras")