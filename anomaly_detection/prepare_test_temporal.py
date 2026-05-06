"""Create a leak-free test set from combined_timeseries_8000.csv using temporal split.

Usage
-----
    python prepare_test_temporal.py

Strategy
--------
The original prepare_data.py randomly shuffled all stride-1 windows before the
80/20 split, so every test window is within 1 timestep of a training window —
severe leakage. This script fixes that by using a temporal split:

For each contiguous run of the same label, the last 20% of timesteps (with a
gap of window_size-1 rows at the boundary) are reserved exclusively for testing.
Training windows come from the first 80%; test windows from the last 20%.
The gap ensures no test window shares any timestep with a boundary training window.

Note: the training data in states_normalized/ stays unchanged. A small number
of training windows were randomly sampled from the "test zone" rows (due to the
original random split), but these are far fewer than in the random-split test set.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import shutil
import numpy as np
import pandas as pd
from collections import Counter
import joblib

SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / "data"
CSV_PATH     = SCRIPT_DIR / "combined_timeseries_8000.csv"
TEST_DIR     = DATA_DIR / "test_states_normalized"
SELECTED_FEATURES_FILE = DATA_DIR / "selected_features.txt"
SCALER_PATH  = DATA_DIR / "scaler.joblib"

WINDOW_SIZE    = 10
TRAIN_FRACTION = 0.8
GAP            = WINDOW_SIZE - 1   # 9 rows: ensures no overlap at the boundary


def find_runs(labels: np.ndarray):
    """Return list of (start_row, end_row_inclusive, label) for each contiguous run."""
    runs = []
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        runs.append((i, j - 1, int(labels[i])))
        i = j
    return runs


def make_windows(feature_array: np.ndarray, labels: np.ndarray):
    """Stride-1 windows within a single run (no boundary crossing)."""
    n = len(labels)
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, feature_array.shape[1]), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)
    windows, window_labels = [], []
    for start in range(n - WINDOW_SIZE + 1):
        windows.append(feature_array[start:start + WINDOW_SIZE])
        window_labels.append(labels[start])   # run is single-label, no vote needed
    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)


def main() -> None:
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    raw_labels = df["anomaly_type"].values.astype(np.int64)
    meta_cols  = [c for c in ["timestamp", "original_timestamp", "anomaly_type", "is_anomaly"]
                  if c in df.columns]
    feature_df = df.drop(columns=meta_cols)

    # Use exactly the same features as training
    with open(SELECTED_FEATURES_FILE) as f:
        selected = [l.strip() for l in f if l.strip()]
    feature_df = feature_df[selected]

    # NaN fill then apply training scaler (no refit — avoids leakage)
    feature_df = feature_df.fillna(feature_df.median())
    scaler = joblib.load(SCALER_PATH)
    feature_array = scaler.transform(feature_df.values).astype(np.float32)

    runs = find_runs(raw_labels)
    print(f"\nRuns found: {len(runs)}")
    print(f"{'Class':>6} | {'Run rows':>12} | {'Test zone rows':>14} | {'Test windows':>12}")
    print("-" * 55)

    all_windows, all_labels = [], []

    for run_start, run_end, label in runs:
        run_len       = run_end - run_start + 1
        test_zone_row = run_start + int(TRAIN_FRACTION * run_len) + GAP
        if test_zone_row > run_end:
            print(f"{label:>6} | {run_start:>5}-{run_end:<5} | run too short, skipped")
            continue

        test_feat   = feature_array[test_zone_row : run_end + 1]
        test_labels = raw_labels[test_zone_row : run_end + 1]

        w, wl = make_windows(test_feat, test_labels)
        n_win = len(w)
        print(f"{label:>6} | {run_start:>5}-{run_end:<5} | {test_zone_row:>5}-{run_end:<5}      | {n_win:>12}")
        all_windows.append(w)
        all_labels.append(wl)

    if not all_windows:
        print("No windows created — check run lengths.")
        return

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels  = np.concatenate(all_labels,  axis=0)
    all_windows = np.nan_to_num(all_windows, nan=0.0)

    print(f"\nTotal test windows : {len(all_windows)}")
    print(f"Label distribution : {Counter(all_labels.tolist())}")

    # Save
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    for idx, (window, label) in enumerate(zip(all_windows, all_labels)):
        np.savez(TEST_DIR / f"state_{idx:07d}.npz", state=window, action=int(label))

    print(f"\nSaved {len(all_windows)} test files to {TEST_DIR}")


if __name__ == "__main__":
    main()
