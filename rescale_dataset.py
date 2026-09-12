"""
rescale_dataset.py — One-time migration for existing datasets
===============================================================

Your original collect_data.py only made features translation-
invariant (wrist-relative), not scale-invariant. This script fixes
your EXISTING landmarks.csv in place, without needing to re-record
anything, by using landmark 9 (middle-finger MCP), which is already
stored as wrist-relative coordinates, as the scale reference.

Usage
-----
    python rescale_dataset.py
"""
import os
import numpy as np
import pandas as pd

DATASET_FILE = os.path.join("dataset", "landmarks.csv")


def main():
    df = pd.read_csv(DATASET_FILE)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols].values.astype(np.float64)

    # landmark 9 = feat_27, feat_28, feat_29 (columns 27,28,29 -> index 9*3..9*3+2)
    ref = X[:, 27:30]
    scale = np.linalg.norm(ref, axis=1)
    scale[scale < 1e-6] = 1e-6

    X_scaled = X / scale[:, None]

    df[feat_cols] = X_scaled
    df.to_csv(DATASET_FILE, index=False)
    print(f"[SUCCESS] Rescaled {len(df)} samples in '{DATASET_FILE}' "
          f"to be scale-invariant.")


if __name__ == "__main__":
    main()
