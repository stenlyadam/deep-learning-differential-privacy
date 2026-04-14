"""
HW2 - Deep Learning with Differential Privacy
Group Name: SecureBytes
Members:
1. Alim Misbullah D11415803
2. Laina Farsiah D11415802
3. Stenly Ibrahim Adam D11215809
4. Aurelio Naufal Effendy M11415802

CIFAR-10 Experiment — 3-Block VGG Model
----------------------------------------
Trains a baseline CNN and DP-SGD CNNs on CIFAR-10 using a deeper
3-block VGG-style architecture for better convergence.

Model: Three VGG-style convolutional blocks (32→64→128 filters).
No BatchNorm — BatchNorm is incompatible with DP-SGD per-example
gradient clipping.

Config:
  Epochs          : 20
  Batch size      : 250
  Learning rate   : 0.15  (SGD)
  L2 norm clip    : 1.0
  Delta           : 1e-5
  Num microbatches: 25    (safe for GPU; per-batch clipping)
  Noise multipliers: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

Note: num_microbatches=25 means gradients are clipped per group of 10
examples (250/25), not per individual example. 

References:
- TensorFlow Privacy tutorial
- Deep Learning with Differential Privacy (Abadi et al., 2016)
"""

import os
import re
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore")

# =========================================================
# TensorFlow Privacy imports
# =========================================================
try:
    from tensorflow_privacy import (
        DPKerasSGDOptimizer,
        compute_dp_sgd_privacy_statement,
    )
except Exception:
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
    from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
        compute_dp_sgd_privacy_statement,
    )

# =========================================================
# Global settings
# =========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE        = 250
EPOCHS            = 20
LEARNING_RATE     = 0.15
L2_NORM_CLIP      = 1.0
DELTA             = 1e-5
NUM_MICROBATCHES  = 25       
NOISE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# File prefix to avoid overwriting 2-block results
PREFIX = "cifar10_3block"


# =========================================================
# Data loading
# =========================================================
def load_cifar10() -> Tuple:
    """Load and preprocess CIFAR-10 (50 000 train / 10 000 test, 32x32 RGB)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0   # [N, 32, 32, 3]
    x_test  = x_test.astype("float32")  / 255.0

    y_train = y_train.reshape(-1)                  # [N,1] → [N]
    y_test  = y_test.reshape(-1)

    return (x_train, y_train), (x_test, y_test)


# =========================================================
# Model architecture — 3-block VGG-style
# =========================================================
def create_model() -> tf.keras.Model:
    """
    3-block VGG-style CNN for CIFAR-10 (32x32x3).
    Deeper than the 2-block MNIST model for better convergence on colour images.
    No BatchNorm (incompatible with DP-SGD per-example gradient clipping).
    Final layer has no softmax — we use from_logits=True.

    Block 1: Conv2D(32, 3x3) + MaxPool
    Block 2: Conv2D(64, 3x3) + MaxPool
    Block 3: Conv2D(128, 3x3) + MaxPool
    Head:    Dense(256) + Dense(10)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        # Block 1
        tf.keras.layers.Conv2D(32, 3, strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10),
    ])
    return model


# =========================================================
# Epsilon computation
# =========================================================
def get_epsilon(num_examples: int, batch_size: int, noise_multiplier: float,
                epochs: int, delta: float) -> float:
    """Compute epsilon using TF Privacy and return the numeric value."""
    try:
        statement = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=noise_multiplier,
            delta=delta,
            used_microbatching=False,
        )
    except TypeError:
        statement = compute_dp_sgd_privacy_statement(
            num_examples, batch_size, epochs, noise_multiplier, delta,
            used_microbatching=False,
        )

    statement = str(statement)
    m = re.search(
        r"Epsilon with each example occurring once per epoch[^:]*:\s*([0-9.]+)",
        statement,
    )
    return float(m.group(1)) if m else np.nan


# =========================================================
# Train baseline
# =========================================================
def train_baseline(x_train, y_train, x_test, y_test) -> Tuple[Dict, Dict]:
    """Train a non-private baseline model."""
    tf.keras.backend.clear_session()
    print("\n==============================")
    print("Training baseline model (3-block)")
    print("==============================")

    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    start = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=2,
    )
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    result = {
        "model":             "Baseline",
        "noise_multiplier":  0.0,
        "l2_norm_clip":      None,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "delta":             DELTA,
        "epsilon":           None,
        "test_loss":         float(test_loss),
        "test_accuracy":     float(test_acc),
        "training_time_sec": round(train_time, 2),
    }
    return result, history.history


# =========================================================
# Train DP-SGD model
# =========================================================
def train_dp_model(x_train, y_train, x_test, y_test,
                   noise_multiplier: float) -> Tuple[Dict, Dict]:
    """
    Train a DP-SGD model.
    - Per-microbatch gradients clipped to L2_NORM_CLIP
    - Gaussian noise scaled by noise_multiplier added to gradients
    - num_microbatches=25 (groups of 10 examples per microbatch)
    """
    tf.keras.backend.clear_session()
    print("\n==============================")
    print(f"Training DP model (3-block) | noise_multiplier={noise_multiplier}")
    print("==============================")

    model = create_model()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,   # required for DP
    )
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=L2_NORM_CLIP,
        noise_multiplier=noise_multiplier,
        num_microbatches=NUM_MICROBATCHES,
        learning_rate=LEARNING_RATE,
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    start = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=2,
    )
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    epsilon = get_epsilon(
        num_examples=len(x_train),
        batch_size=BATCH_SIZE,
        noise_multiplier=noise_multiplier,
        epochs=EPOCHS,
        delta=DELTA,
    )

    result = {
        "model":             f"DP-SGD_sigma_{noise_multiplier}",
        "noise_multiplier":  noise_multiplier,
        "l2_norm_clip":      L2_NORM_CLIP,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "delta":             DELTA,
        "epsilon":           round(float(epsilon), 3),
        "test_loss":         float(test_loss),
        "test_accuracy":     float(test_acc),
        "training_time_sec": round(train_time, 2),
    }
    return result, history.history


# =========================================================
# Plot helpers
# =========================================================
def plot_accuracies(results_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["test_accuracy"])
    plt.ylabel("Test Accuracy")
    plt.xlabel("Model")
    plt.title("CIFAR-10 (3-Block): Baseline vs Differential Privacy Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{PREFIX}_accuracy_comparison.png"), dpi=200)
    plt.close()


def plot_privacy_tradeoff(results_df: pd.DataFrame):
    dp_df = results_df.dropna(subset=["epsilon"]).copy()
    dp_df["epsilon"] = pd.to_numeric(dp_df["epsilon"], errors="coerce")
    dp_df = dp_df.dropna(subset=["epsilon"])
    if len(dp_df) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(dp_df["epsilon"], dp_df["test_accuracy"], marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy")
    plt.title("CIFAR-10 (3-Block): Privacy-Utility Tradeoff")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{PREFIX}_privacy_utility_tradeoff.png"), dpi=200)
    plt.close()


def plot_learning_curves(baseline_history: Dict, dp_history: Dict):
    """Plot per-epoch train/val accuracy for baseline and DP (sigma=1.0)."""
    epochs = range(len(baseline_history["accuracy"]))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, baseline_history["accuracy"],     "b-",  label="Baseline Train")
    plt.plot(epochs, baseline_history["val_accuracy"], color="orange", linestyle="--",
             label="Baseline Val")
    plt.plot(epochs, dp_history["accuracy"],           "g-",  label="DP (σ=1.0) Train")
    plt.plot(epochs, dp_history["val_accuracy"],       "r--", label="DP (σ=1.0) Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 (3-Block): Learning Behavior Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{PREFIX}_learning_behavior_comparison.png"), dpi=200)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(f"  Train: {x_train.shape}  Test: {x_test.shape}")

    all_results: List[Dict] = []

    # 1) Baseline
    result, baseline_hist = train_baseline(x_train, y_train, x_test, y_test)
    all_results.append(result)
    pd.DataFrame(baseline_hist).to_csv(
        os.path.join(RESULT_DIR, f"{PREFIX}_baseline_history.csv"), index=False
    )

    # 2) DP runs
    dp_hist_sigma1: Dict = {}
    for sigma in NOISE_MULTIPLIERS:
        result, history = train_dp_model(
            x_train, y_train, x_test, y_test, noise_multiplier=sigma
        )
        all_results.append(result)
        pd.DataFrame(history).to_csv(
            os.path.join(RESULT_DIR, f"{PREFIX}_dp_history_sigma_{sigma}.csv"),
            index=False,
        )
        if sigma == 1.0:
            dp_hist_sigma1 = history

    # 3) Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(RESULT_DIR, f"{PREFIX}_experiment_results.csv"), index=False
    )

    # 4) Print table
    print("\n==============================")
    print("CIFAR-10 (3-Block) Results")
    print("==============================")
    print(results_df.to_string(index=False))

    # 5) Plots
    plot_accuracies(results_df)
    plot_privacy_tradeoff(results_df)
    if dp_hist_sigma1:
        plot_learning_curves(baseline_hist, dp_hist_sigma1)

    print(f"\nAll results saved in: {RESULT_DIR}/")


if __name__ == "__main__":
    main()
