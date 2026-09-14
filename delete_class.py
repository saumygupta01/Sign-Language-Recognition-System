"""
delete_class.py — Remove a Specific Class from the Dataset
===========================================================

Deletes all collected landmark samples for a specific gesture/letter
class from the dataset CSV file.

Usage
-----
    python delete_class.py <CLASS_LABEL>

Example
-------
    python delete_class.py B
"""

import os
import sys
import pandas as pd

DATASET_FILE = os.path.join("dataset", "landmarks.csv")

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Please specify the class label to delete.")
        print("Usage: python delete_class.py <CLASS_LABEL>")
        print("Example: python delete_class.py B")
        return

    target_class = sys.argv[1].upper()

    if not os.path.exists(DATASET_FILE):
        print(f"[ERROR] Dataset file not found at '{DATASET_FILE}'.")
        return

    df = pd.read_csv(DATASET_FILE)

    if 'label' not in df.columns:
        print("[ERROR] CSV format is invalid (no 'label' column found).")
        return

    initial_count = len(df)
    class_exist = target_class in df['label'].values

    if not class_exist:
        print(f"[INFO] No samples found for class '{target_class}' in dataset.")
        return

    df_filtered = df[df['label'] != target_class]
    removed_count = initial_count - len(df_filtered)

    df_filtered.to_csv(DATASET_FILE, index=False)
    print(f"[SUCCESS] Removed {removed_count} samples for class '{target_class}' from '{DATASET_FILE}'.")
    print(f"          New dataset size: {len(df_filtered)} samples.")

if __name__ == "__main__":
    main() 