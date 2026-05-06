"""Preprocess mac_timeseries.csv and save as the test dataset.

Usage
-----
    python prepare_mac_test.py

Applies the same feature selection and normalization as prepare_data.py,
using the already-saved selected_features.txt and scaler.joblib so that
the test data is in exactly the same space as the training data.

Windows are created with majority-vote labeling and allow boundary crossing,
because class 4 runs in this dataset are only 6-7 timesteps (shorter than
window_size=10), so strict no-crossing would produce zero class 4 windows.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import shutil
import numpy as np
import pandas as pd
from collections import Counter
import joblib

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CSV_PATH = SCRIPT_DIR / "mac_timeseries.csv"
TEST_STATES_DIR = DATA_DIR / "test_states_normalized"
SELECTED_FEATURES_FILE = DATA_DIR / "selected_features.txt"
SCALER_PATH = DATA_DIR / "scaler.joblib"

WINDOW_SIZE = 10


def make_windows_majority_vote(feature_array: np.ndarray, labels: np.ndarray,
                               window_size: int = WINDOW_SIZE):
    """Sliding window over all rows with majority-vote labeling.

    Unlike prepare_data.py, this allows windows to cross label boundaries.
    This is necessary here because class 4 runs are only 6-7 timesteps,
    shorter than window_size, so strict no-crossing produces no class 4 windows.
    A window with >=6 out of 10 class-4 timesteps is labeled class 4.
    """
    n = len(labels)
    windows, window_labels = [], []
    for start in range(n - window_size + 1):
        window = feature_array[start:start + window_size]
        majority = Counter(labels[start:start + window_size]).most_common(1)[0][0]
        windows.append(window)
        window_labels.append(majority)
    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)


def main() -> None:
    # ------------------------------------------------------------------ load
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

    labels = df["anomaly_type"].values.astype(np.int64)
    meta_cols = [c for c in ["timestamp", "original_timestamp", "anomaly_type", "is_anomaly"]
                 if c in df.columns]
    feature_df = df.drop(columns=meta_cols)

    # ------------------------------------------------------------------ select same features as training
    with open(SELECTED_FEATURES_FILE) as f:
        selected_features = [l.strip() for l in f if l.strip()]

    missing = [c for c in selected_features if c not in feature_df.columns]
    if missing:
        raise ValueError(f"mac_timeseries.csv is missing {len(missing)} selected features: {missing[:5]}")

    feature_df = feature_df[selected_features]
    print(f"Selected {len(selected_features)} features (same as training).")

    # ------------------------------------------------------------------ NaN fill + scale
    col_medians = feature_df.median()
    feature_df = feature_df.fillna(col_medians)
    print(f"NaN values after fill: {feature_df.isnull().sum().sum()}")

    scaler = joblib.load(SCALER_PATH)
    feature_array = scaler.transform(feature_df.values).astype(np.float32)
    print(f"Feature range after scaling — min: {feature_array.min():.3f}, max: {feature_array.max():.3f}")

    # ------------------------------------------------------------------ windows
    print("Creating windows (majority-vote, boundary crossing allowed)...")
    windows, window_labels = make_windows_majority_vote(feature_array, labels, WINDOW_SIZE)
    windows = np.nan_to_num(windows, nan=0.0)
    print(f"Total windows: {len(windows)}")
    print(f"Label distribution: {Counter(window_labels.tolist())}")

    # ------------------------------------------------------------------ save
    if TEST_STATES_DIR.exists():
        shutil.rmtree(TEST_STATES_DIR)
    TEST_STATES_DIR.mkdir(parents=True)

    print(f"Saving {len(windows)} test windows to {TEST_STATES_DIR} ...")
    for idx, (window, label) in enumerate(zip(windows, window_labels)):
        np.savez(TEST_STATES_DIR / f"state_{idx:07d}.npz", state=window, action=int(label))

    print(f"Done. {len(windows)} test files saved to {TEST_STATES_DIR}")


if __name__ == "__main__":
    main()
