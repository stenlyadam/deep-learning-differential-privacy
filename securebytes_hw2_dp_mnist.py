"""
HW2 - Deep Learning with Differential Privacy
Group Name: SecureBytes
Members:
1. Alim Misbullah D11415803
2. Laina Farsiah D11415802
3. Stenly Ibrahim Adam D11215809
4. Aurelio Naufal Effendy M11415802

This script:
1. Trains a baseline CNN on MNIST
2. Trains DP-SGD CNNs with different noise multipliers
3. Computes test accuracy
4. Computes privacy budget epsilon
5. Saves results to CSV

References:
- TensorFlow Privacy MNIST tutorial
- Deep Learning with Differential Privacy (Abadi et al., 2016)
"""

import os
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Silence excessive TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# =========================================================
# Try imports for TensorFlow Privacy
# Official docs expose these from tf_privacy / tensorflow_privacy.
# Some versions differ slightly, so we try a fallback path too.
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

BATCH_SIZE = 250
EPOCHS = 5
LEARNING_RATE = 0.15
L2_NORM_CLIP = 1.0
DELTA = 1e-5

# You can change these to compare different privacy strengths
NOISE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# =========================================================
# Data loading
# =========================================================
def load_mnist():
    """Load and preprocess MNIST."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: [N, 28, 28, 1]
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


# =========================================================
# Model
# =========================================================
def create_model() -> tf.keras.Model:
    """
    Create a small CNN similar to the TensorFlow Privacy tutorial style.
    Final layer has no softmax because we use from_logits=True.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 8, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=1),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding="valid", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10)
    ])
    return model


# =========================================================
# Epsilon computation helper
# =========================================================
def get_epsilon(num_examples: int, batch_size: int, noise_multiplier: float,
                epochs: int, delta: float) -> float:
    """
    Compute epsilon using TensorFlow Privacy helper.

    Official docs expose:
    compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta)
    """
    try:
        eps = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=noise_multiplier,
            delta=delta,
            used_microbatching=True
        )
    except TypeError:
        # fallback for older/newer signatures
        eps = compute_dp_sgd_privacy_statement(
            num_examples,
            batch_size,
            epochs,
            noise_multiplier,
            delta,
            used_microbatching=True
        )
    return str(eps)

# =========================================================
# Train baseline
# =========================================================
def train_baseline(x_train, y_train, x_test, y_test) -> Dict:
    """Train a non-private baseline model."""
    print("\n==============================")
    print("Training baseline model")
    print("==============================")

    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=2
    )
    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    result = {
        "model": "Baseline",
        "noise_multiplier": 0.0,
        "l2_norm_clip": None,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "delta": DELTA,
        "epsilon": None,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "training_time_sec": round(train_time, 2)
    }

    return result, history.history


# =========================================================
# Train DP model
# =========================================================
def train_dp_model(x_train, y_train, x_test, y_test, noise_multiplier: float) -> Dict:
    """
    Train a DP-SGD model.

    Key DP idea:
    - per-example gradients are clipped to L2_NORM_CLIP
    - Gaussian noise is added during optimization
    """
    print("\n==============================")
    print(f"Training DP model | noise_multiplier={noise_multiplier}")
    print("==============================")

    model = create_model()

    # For DP optimizers, use unreduced loss so microbatches/per-example handling works
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=L2_NORM_CLIP,
        noise_multiplier=noise_multiplier,
        num_microbatches=BATCH_SIZE,   # one microbatch per example for simplicity
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=2
    )
    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    epsilon = get_epsilon(
        num_examples=len(x_train),
        batch_size=BATCH_SIZE,
        noise_multiplier=noise_multiplier,
        epochs=EPOCHS,
        delta=DELTA
    )

    result = {
        "model": f"DP-SGD_sigma_{noise_multiplier}",
        "noise_multiplier": noise_multiplier,
        "l2_norm_clip": L2_NORM_CLIP,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "delta": DELTA,
        "epsilon": epsilon,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "training_time_sec": round(train_time, 2)
    }

    return result, history.history


# =========================================================
# Plot helper
# =========================================================
def plot_accuracies(results_df: pd.DataFrame):
    """Save a simple accuracy comparison plot."""
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["test_accuracy"])
    plt.ylabel("Test Accuracy")
    plt.xlabel("Model")
    plt.title("Baseline vs Differential Privacy Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "accuracy_comparison.png"), dpi=200)
    plt.close()


def plot_privacy_tradeoff(results_df: pd.DataFrame):
    """Plot epsilon vs accuracy for DP runs only."""
    dp_df = results_df.dropna(subset=["epsilon"]).copy()
    if len(dp_df) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(dp_df["epsilon"], dp_df["test_accuracy"], marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy")
    plt.title("Privacy-Utility Tradeoff")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "privacy_utility_tradeoff.png"), dpi=200)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    all_results: List[Dict] = []

    # 1) Baseline
    baseline_result, baseline_history = train_baseline(
        x_train, y_train, x_test, y_test
    )
    all_results.append(baseline_result)

    # Save baseline history
    pd.DataFrame(baseline_history).to_csv(
        os.path.join(RESULT_DIR, "baseline_history.csv"),
        index=False
    )

    # 2) DP runs
    for sigma in NOISE_MULTIPLIERS:
        dp_result, dp_history = train_dp_model(
            x_train, y_train, x_test, y_test, noise_multiplier=sigma
        )
        all_results.append(dp_result)

        pd.DataFrame(dp_history).to_csv(
            os.path.join(RESULT_DIR, f"dp_history_sigma_{sigma}.csv"),
            index=False
        )

    # 3) Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "experiment_results.csv"), index=False)

    # 4) Print final table
    print("\n==============================")
    print("Final Results")
    print("==============================")
    print(results_df.to_string(index=False))

    # 5) Save plots
    plot_accuracies(results_df)
    plot_privacy_tradeoff(results_df)

    print(f"\nAll results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
