import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "final_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD TUNING RESULTS
# ============================================================

TUNING_FILE = RESULTS_DIR / "efficientnet_tuning_results.json"

with open(TUNING_FILE, "r", encoding="utf-8") as f:
    tuning_data = json.load(f)


print("=" * 75)
print("GENERATING FINAL EVALUATION PACKAGE")
print("=" * 75)


# ============================================================
# BEST CONFIGURATION
# ============================================================

best_config = tuning_data["best_configuration"]
best_metrics = tuning_data["best_validation_metrics"]

print("\nBest configuration:")
print(best_config)

print("\nBest validation metrics:")
print(best_metrics)


# ============================================================
# FIND CONFIGURATION USING HYPERPARAMETERS
# ============================================================

configurations = tuning_data["configurations"]

selected_config = None


if isinstance(configurations, list):

    for config in configurations:

        if not isinstance(config, dict):
            continue

        # ----------------------------------------------------
        # Get possible nested configuration information
        # ----------------------------------------------------

        candidate = config

        if "config" in config and isinstance(config["config"], dict):
            candidate = config["config"]

        # ----------------------------------------------------
        # Match against the known best configuration
        # ----------------------------------------------------

        lr_match = (
            float(candidate.get("lr", -1))
            == float(best_config.get("lr", -2))
        )

        wd_match = (
            float(candidate.get("weight_decay", -1))
            == float(best_config.get("weight_decay", -2))
        )

        dropout_match = (
            float(candidate.get("dropout", -1))
            == float(best_config.get("dropout", -2))
        )

        blocks_match = (
            int(candidate.get("unfreeze_blocks", -1))
            == int(best_config.get("unfreeze_blocks", -2))
        )

        if (
            lr_match
            and wd_match
            and dropout_match
            and blocks_match
        ):

            selected_config = config
            break


# ============================================================
# FALLBACK: FIND CONFIGURATION BY POSITION
# ============================================================

if selected_config is None:

    # config_5 is the fifth configuration according to
    # the tuning results and best_configuration.
    if len(configurations) >= 5:

        selected_config = configurations[4]

        print(
            "\nConfiguration matched by position: "
            "configuration 5"
        )


# ============================================================
# STOP IF NOT FOUND
# ============================================================

if selected_config is None:

    print("\nERROR: Could not identify the best configuration.")

    print("\nConfiguration structure:")

    for i, config in enumerate(configurations, start=1):

        print(f"\nConfiguration {i}:")
        print(config)

    raise SystemExit(1)


print("\nSelected configuration:")
print(selected_config)


# ============================================================
# FIND TRAINING HISTORY
# ============================================================

history = None


# Direct history
if isinstance(selected_config.get("history"), list):

    history = selected_config["history"]


# Nested history
if history is None:

    for key in [
        "result",
        "results",
        "training",
        "training_history",
        "metrics"
    ]:

        value = selected_config.get(key)

        if isinstance(value, dict):

            if isinstance(value.get("history"), list):

                history = value["history"]
                break


# ============================================================
# HISTORY NOT FOUND
# ============================================================

if history is None:

    print("\nERROR: Training history was not found.")

    print("\nSelected configuration keys:")
    print(list(selected_config.keys()))

    print("\nSelected configuration:")
    print(json.dumps(selected_config, indent=4))

    raise SystemExit(1)


print("\nTraining history found successfully.")
print("Number of epochs:", len(history))


# ============================================================
# EXTRACT HISTORY
# ============================================================

epochs = []

train_loss = []
val_loss = []

train_accuracy = []
val_accuracy = []

val_precision = []
val_recall = []
val_f1 = []
val_auc = []


for i, item in enumerate(history):

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------

    epoch = item.get("epoch", i + 1)

    epochs.append(epoch)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    train_loss.append(
        item.get("train_loss", np.nan)
    )

    val_loss.append(
        item.get("val_loss", np.nan)
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    train_accuracy.append(
        item.get("train_accuracy", np.nan)
    )

    val_accuracy.append(
        item.get("val_accuracy", np.nan)
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    val_precision.append(
        item.get("val_precision", np.nan)
    )

    val_recall.append(
        item.get("val_recall", np.nan)
    )

    val_f1.append(
        item.get("val_f1", np.nan)
    )

    val_auc.append(
        item.get("val_auc", np.nan)
    )


# Convert to numpy arrays

epochs = np.array(epochs)

train_loss = np.array(train_loss, dtype=float)
val_loss = np.array(val_loss, dtype=float)

train_accuracy = np.array(
    train_accuracy,
    dtype=float
)

val_accuracy = np.array(
    val_accuracy,
    dtype=float
)

val_precision = np.array(
    val_precision,
    dtype=float
)

val_recall = np.array(
    val_recall,
    dtype=float
)

val_f1 = np.array(
    val_f1,
    dtype=float
)

val_auc = np.array(
    val_auc,
    dtype=float
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 1. TRAINING / VALIDATION LOSS
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

if not np.all(np.isnan(val_loss)):

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        linewidth=2,
        label="Validation Loss"
    )

plt.title(
    "EfficientNet-B0 Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_plot(
    "training_validation_loss.png"
)


# ============================================================
# 2. TRAINING / VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    train_accuracy * 100,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy * 100,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.title(
    "EfficientNet-B0 Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_plot(
    "training_validation_accuracy.png"
)


# ============================================================
# 3. VALIDATION F1
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    val_f1 * 100,
    marker="o",
    linewidth=2,
    label="Validation F1"
)

best_f1_index = np.nanargmax(val_f1)

plt.scatter(
    epochs[best_f1_index],
    val_f1[best_f1_index] * 100,
    s=100,
    label=(
        f"Best F1: "
        f"{val_f1[best_f1_index] * 100:.2f}%"
    )
)

plt.title(
    "EfficientNet-B0 Validation F1 Score"
)

plt.xlabel("Epoch")
plt.ylabel("F1 Score (%)")

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_plot(
    "validation_f1.png"
)


# ============================================================
# 4. VALIDATION ROC-AUC
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    val_auc * 100,
    marker="o",
    linewidth=2,
    label="Validation ROC-AUC"
)

best_auc_index = np.nanargmax(val_auc)

plt.scatter(
    epochs[best_auc_index],
    val_auc[best_auc_index] * 100,
    s=100,
    label=(
        f"Best AUC: "
        f"{val_auc[best_auc_index] * 100:.2f}%"
    )
)

plt.title(
    "EfficientNet-B0 Validation ROC-AUC"
)

plt.xlabel("Epoch")
plt.ylabel("ROC-AUC (%)")

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_plot(
    "validation_auc.png"
)


# ============================================================
# 5. PRECISION / RECALL
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    val_precision * 100,
    marker="o",
    linewidth=2,
    label="Validation Precision"
)

plt.plot(
    epochs,
    val_recall * 100,
    marker="o",
    linewidth=2,
    label="Validation Recall"
)

plt.title(
    "EfficientNet-B0 Validation Precision and Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Score (%)")

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

save_plot(
    "validation_precision_recall.png"
)


# ============================================================
# SAVE CLEAN HISTORY
# ============================================================

clean_history = {

    "configuration": best_config,

    "epochs": epochs.tolist(),

    "train_loss": train_loss.tolist(),

    "val_loss": val_loss.tolist(),

    "train_accuracy": train_accuracy.tolist(),

    "val_accuracy": val_accuracy.tolist(),

    "val_precision": val_precision.tolist(),

    "val_recall": val_recall.tolist(),

    "val_f1": val_f1.tolist(),

    "val_auc": val_auc.tolist()
}


with open(
    OUTPUT_DIR / "config_5_training_history.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        clean_history,
        f,
        indent=4
    )


# ============================================================
# TRAINING CURVE SUMMARY
# ============================================================

best_f1 = float(np.nanmax(val_f1))

best_auc = float(np.nanmax(val_auc))

best_val_accuracy = float(
    np.nanmax(val_accuracy)
)

best_f1_epoch = int(
    epochs[np.nanargmax(val_f1)]
)

best_auc_epoch = int(
    epochs[np.nanargmax(val_auc)]
)

best_accuracy_epoch = int(
    epochs[np.nanargmax(val_accuracy)]
)


summary = {

    "configuration": best_config,

    "epochs_trained": len(epochs),

    "best_validation_accuracy":
        best_val_accuracy,

    "best_validation_f1":
        best_f1,

    "best_validation_auc":
        best_auc,

    "best_accuracy_epoch":
        best_accuracy_epoch,

    "best_f1_epoch":
        best_f1_epoch,

    "best_auc_epoch":
        best_auc_epoch
}


with open(
    OUTPUT_DIR / "training_curve_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("TRAINING CURVES GENERATED SUCCESSFULLY")
print("=" * 75)

print(
    f"\nConfiguration: {best_config['name']}"
)

print(
    f"Epochs: {len(epochs)}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}% "
    f"(Epoch {best_accuracy_epoch})"
)

print(
    f"Best Validation F1: "
    f"{best_f1 * 100:.2f}% "
    f"(Epoch {best_f1_epoch})"
)

print(
    f"Best Validation AUC: "
    f"{best_auc * 100:.2f}% "
    f"(Epoch {best_auc_epoch})"
)

print("\nFiles generated:")

for file in sorted(OUTPUT_DIR.iterdir()):

    print("  " + file.name)

print("\nOutput directory:")
print(OUTPUT_DIR)