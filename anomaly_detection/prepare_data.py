"""Prepare anomaly detection data: feature reduction, windowing, and saving .npz files.

Usage
-----
    python prepare_data.py

Reads ``combined_timeseries_8000.csv``, applies feature reduction, creates
10-timestep sliding windows (no boundary crossing), shuffles, splits 80/20,
and saves state_XXXXXXX.npz files to data/states/ and data/test_states/.
Also saves the selected feature names to data/selected_features.txt.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CSV_PATH = SCRIPT_DIR / "combined_timeseries_8000.csv"
STATES_DIR = DATA_DIR / "states_normalized"
TEST_STATES_DIR = DATA_DIR / "test_states_normalized"
SELECTED_FEATURES_FILE = DATA_DIR / "selected_features.txt"

WINDOW_SIZE = 10
SEED = 42
TRAIN_FRACTION = 0.8


def reduce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply three-stage feature reduction to the dataframe.

    Stages
    ------
    1. Drop zero-variance columns.
    2. Drop ``plat_*_total_events`` and ``plat_*_total_runtime`` columns.
    3. Drop highly correlated features (|correlation| > 0.95), keeping one from each group.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-only dataframe (no metadata columns).

    Returns
    -------
    pd.DataFrame
        Reduced dataframe.
    """
    # Stage 1: drop zero-variance columns
    std = df.std()
    zero_var_cols = std[std == 0].index.tolist()
    df = df.drop(columns=zero_var_cols)
    print(f"Dropped {len(zero_var_cols)} zero-variance columns. Remaining: {df.shape[1]}")

    # Stage 2: drop plat_*_total_events and plat_*_total_runtime
    redundant_plat = [c for c in df.columns
                      if c.startswith("plat_") and (c.endswith("_total_events") or c.endswith("_total_runtime"))]
    df = df.drop(columns=redundant_plat)
    print(f"Dropped {len(redundant_plat)} plat_*_total_events/total_runtime columns. Remaining: {df.shape[1]}")

    # Stage 3: drop highly correlated features (|corr| > 0.90)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    df = df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated columns. Remaining: {df.shape[1]}")

    # Stage 4: per signal group, keep only mean and std; drop max/min/range/var/skewness/kurtosis/irq/outliers
    stat_suffixes_to_drop = ("_max", "_min", "_range", "_var",
                             "_skewness", "_kurtosis", "_irq", "_outliers")
    stat_suffixes_to_keep = ("_mean", "_std")
    all_stat_suffixes = stat_suffixes_to_drop + stat_suffixes_to_keep

    drop_stat_cols = []
    for col in df.columns:
        for suffix in stat_suffixes_to_drop:
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                # Only drop if the mean or std variant also exists — confirms it's a stat group
                has_companion = any((base + k) in df.columns for k in stat_suffixes_to_keep)
                if has_companion:
                    drop_stat_cols.append(col)
                break
    df = df.drop(columns=drop_stat_cols)
    print(f"Dropped {len(drop_stat_cols)} redundant stat columns (kept mean+std). Remaining: {df.shape[1]}")

    return df


def make_windows(feature_array: np.ndarray, labels: np.ndarray, window_size: int = WINDOW_SIZE):
    """Create non-boundary-crossing windows with majority-vote labels.

    Parameters
    ----------
    feature_array : np.ndarray
        Shape (N, n_features).
    labels : np.ndarray
        Integer anomaly_type labels, shape (N,).
    window_size : int
        Number of consecutive timesteps per window.

    Returns
    -------
    tuple of np.ndarray
        ``(windows, window_labels)`` where windows has shape
        ``(M, window_size, n_features)`` and window_labels has shape ``(M,)``.
    """
    windows = []
    window_labels = []
    n = len(labels)

    # Find consecutive runs of same label
    i = 0
    while i < n:
        current_label = labels[i]
        run_start = i
        # Extend run while same label
        while i < n and labels[i] == current_label:
            i += 1
        run_end = i  # exclusive

        run_length = run_end - run_start
        if run_length < window_size:
            continue  # Not enough samples for even one window

        # Slide windows within this run
        for w_start in range(run_start, run_end - window_size + 1):
            w_end = w_start + window_size
            window = feature_array[w_start:w_end]
            window_label_counts = Counter(labels[w_start:w_end])
            majority_label = window_label_counts.most_common(1)[0][0]
            windows.append(window)
            window_labels.append(majority_label)

    windows = np.array(windows, dtype=np.float32)
    window_labels = np.array(window_labels, dtype=np.int64)
    return windows, window_labels


def main() -> None:
    """Run the full data preparation pipeline."""
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    TEST_STATES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Separate metadata
    labels = df["anomaly_type"].values.astype(np.int64)
    meta_cols = ["timestamp", "anomaly_type", "is_anomaly"]
    feature_df = df.drop(columns=meta_cols)

    print(f"Feature columns before reduction: {feature_df.shape[1]}")
    feature_df = reduce_features(feature_df)
    print(f"Feature columns after reduction: {feature_df.shape[1]}")

    # Save selected feature names
    SELECTED_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTED_FEATURES_FILE, "w") as f:
        for col in feature_df.columns:
            f.write(col + "\n")
    print(f"Saved {len(feature_df.columns)} selected features to {SELECTED_FEATURES_FILE}")

    # Print unique anomaly types
    unique_labels = sorted(np.unique(labels).tolist())
    print(f"Unique anomaly_type values: {unique_labels}")
    print(f"N_ACTIONS = {len(unique_labels)}")

    # Fill NaN with column median (computed before scaling)
    col_medians = feature_df.median()
    feature_df = feature_df.fillna(col_medians)
    print(f"NaN values remaining after fill: {feature_df.isnull().sum().sum()}")

    # Fit StandardScaler on full dataset and save for inference
    scaler = StandardScaler()
    feature_array = scaler.fit_transform(feature_df.values).astype(np.float32)
    scaler_path = DATA_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    print(f"Feature range after scaling — min: {feature_array.min():.3f}, max: {feature_array.max():.3f}")

    print("Creating windows...")
    windows, window_labels = make_windows(feature_array, labels, window_size=WINDOW_SIZE)
    # Replace any residual NaN (from edge cases) with 0
    windows = np.nan_to_num(windows, nan=0.0)
    print(f"Total windows created: {len(windows)}")

    # Shuffle with seed
    rng = np.random.RandomState(seed=SEED)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    windows = windows[indices]
    window_labels = window_labels[indices]

    # 80/20 split
    n_train = int(TRAIN_FRACTION * len(windows))
    train_windows = windows[:n_train]
    train_labels = window_labels[:n_train]
    test_windows = windows[n_train:]
    test_labels = window_labels[n_train:]

    print(f"Train windows: {len(train_windows)}, Test windows: {len(test_windows)}")

    # Save training windows
    print("Saving training windows...")
    for idx, (window, label) in enumerate(zip(train_windows, train_labels)):
        save_path = STATES_DIR / f"state_{idx:07d}.npz"
        np.savez(save_path, state=window, action=int(label))

    # Save test windows
    print("Saving test windows...")
    for idx, (window, label) in enumerate(zip(test_windows, test_labels)):
        save_path = TEST_STATES_DIR / f"state_{idx:07d}.npz"
        np.savez(save_path, state=window, action=int(label))

    print(f"Done. Train: {len(train_windows)} files in {STATES_DIR}")
    print(f"      Test:  {len(test_windows)} files in {TEST_STATES_DIR}")
    print(f"Label distribution (train): {Counter(train_labels.tolist())}")
    print(f"Label distribution (test):  {Counter(test_labels.tolist())}")


if __name__ == "__main__":
    main()
