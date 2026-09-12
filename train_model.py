"""
train_model.py  —  Train a Random Forest Classifier on Hand Landmarks
======================================================================

This script reads the hand-landmark dataset (CSV) created by
collect_data.py, trains a Random Forest classifier, evaluates it,
and saves the trained model for real-time prediction.

Usage
-----
    python train_model.py

Output
------
    models/sign_language_model.pkl   — the trained model
    models/label_classes.npy         — ordered class labels

Console output includes accuracy, classification report, and a
confusion-matrix heatmap saved to  models/confusion_matrix.png.
"""

# ──────────────────────────── Imports ────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # Use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ──────────────────────────── Configuration ──────────────────────
DATASET_FILE = os.path.join("dataset", "landmarks.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sign_language_model.pkl")
LABELS_FILE = os.path.join(MODEL_DIR, "label_classes.npy")
CONFUSION_MATRIX_FILE = os.path.join(MODEL_DIR, "confusion_matrix.png")

TEST_SIZE = 0.20          # 80 % training, 20 % testing
RANDOM_STATE = 42         # For reproducibility

# Random Forest hyper-parameters (simple defaults work well)
N_ESTIMATORS = 100        # Number of trees in the forest
MAX_DEPTH = None          # Trees grow until leaves are pure


# ──────────────────────────── Helper Functions ───────────────────

def load_dataset(filepath):
    """
    Load the CSV dataset and separate features (X) from labels (y).

    Parameters
    ----------
    filepath : str
        Path to the landmarks CSV file.

    Returns
    -------
    X : np.ndarray   — shape (n_samples, 63)
    y : np.ndarray   — shape (n_samples,)  with string labels like 'A', 'B', …
    """
    print(f"[INFO] Loading dataset from '{filepath}' …")

    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset file not found: {filepath}")
        print("        Please run  collect_data.py  first to create the dataset.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")

    # The last column is the label; everything else is a feature
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values

    # Quick sanity check
    unique_labels = sorted(set(y))
    print(f"[INFO] Classes found: {unique_labels}")
    for label in unique_labels:
        count = (y == label).sum()
        print(f"       {label}: {count} samples")

    return X, y


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.

    Why Random Forest?
    ──────────────────
    • Works well with small-to-medium tabular datasets.
    • Handles 63-dimensional feature vectors efficiently.
    • Provides built-in feature-importance and probability estimates.
    • Robust against overfitting (ensemble of many decision trees).
    • Does NOT require a GPU or large computational resources.

    Parameters
    ----------
    X_train : np.ndarray  — training features
    y_train : np.ndarray  — training labels

    Returns
    -------
    model : RandomForestClassifier  — the trained model
    """
    print(f"\n[INFO] Training Random Forest with {N_ESTIMATORS} trees …")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,            # Use all CPU cores for speed
    )

    model.fit(X_train, y_train)
    print("[INFO] Training complete.")

    return model


def evaluate_model(model, X_test, y_test, labels):
    """
    Evaluate the trained model on the test set.

    Prints:
        • Overall accuracy
        • Per-class precision, recall, F1
        • Confusion matrix (saved as a heatmap image)

    Parameters
    ----------
    model  : trained classifier
    X_test : np.ndarray  — test features
    y_test : np.ndarray  — true labels
    labels : list[str]   — ordered class labels
    """
    y_pred = model.predict(X_test)

    # ── Accuracy ──
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc * 100:.2f}%")

    # ── Macro-averaged metrics ──
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")
    print(f"{'='*50}\n")

    # ── Per-class report ──
    print("[INFO] Per-class classification report:\n")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    # ── Confusion matrix heatmap ──
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    print(f"[INFO] Confusion matrix saved to '{CONFUSION_MATRIX_FILE}'")


# ──────────────────────────── Main ───────────────────────────────

def main():
    print("=" * 55)
    print("  SIGN LANGUAGE MODEL TRAINER")
    print("=" * 55)

    # ── 1. Load dataset ──
    X, y = load_dataset(DATASET_FILE)

    # ── 2. Train / test split ──
    # IMPORTANT: collect_data.py records each class in one continuous
    # take, so consecutive rows are near-duplicate frames (tiny hand
    # jitter apart). A random shuffle split leaks near-duplicates into
    # both train and test, which is why evaluation can look ~100%
    # accurate while the live app still mismatches. Instead we hold
    # out the LAST chunk of each class's recording as test data, which
    # is a more honest (if still imperfect) estimate — the real fix
    # is recording each class across multiple separate sessions at
    # different distances/angles/lighting.
    labels = sorted(set(y))
    train_idx, test_idx = [], []
    for label in labels:
        idx = np.where(y == label)[0]
        cut = int(len(idx) * (1 - TEST_SIZE))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\n[INFO] Training set : {X_train.shape[0]} samples")
    print(f"[INFO] Testing set  : {X_test.shape[0]} samples")

    # ── 3. Train ──
    model = train_random_forest(X_train, y_train)

    # ── 4. Evaluate ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    evaluate_model(model, X_test, y_test, labels)

    # ── 5. Save model ──
    joblib.dump(model, MODEL_FILE)
    np.save(LABELS_FILE, np.array(labels))
    print(f"\n[INFO] Model saved to '{MODEL_FILE}'")
    print(f"[INFO] Label classes saved to '{LABELS_FILE}'")
    print("\n[INFO] Training pipeline complete.  You can now run:")
    print("           python main.py")


# ──────────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    main()
