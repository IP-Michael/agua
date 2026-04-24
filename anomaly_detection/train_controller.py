"""Train the 1D CNN anomaly detection controller.

Usage
-----
    python train_controller.py

Trains AnomalyModel on the pre-saved .npz windows from data/states/,
evaluates on data/test_states/, prints a classification report, and saves
trained weights to data/anomaly_model.pt.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import global_constants as GC
from global_constants import AnomalyModel

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
N_EPOCHS = 30
LR = 1e-3
SEED = GC.SEED


def load_split(directory: Path, max_samples: int = None):
    """Load all .npz state files from a directory.

    Parameters
    ----------
    directory : Path
        Directory containing state_XXXXXXX.npz files.
    max_samples : int, optional
        Maximum number of files to load.

    Returns
    -------
    tuple of np.ndarray
        ``(states, actions)`` arrays.
    """
    files = sorted(directory.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]
    states_list = []
    actions_list = []
    for f in files:
        data = np.load(f)
        states_list.append(data["state"])
        actions_list.append(int(data["action"]))
    states = np.stack(states_list, axis=0).astype(np.float32)
    actions = np.array(actions_list, dtype=np.int64)
    return states, actions


def build_dataloader(states: np.ndarray, actions: np.ndarray,
                     batch_size: int, shuffle: bool = True) -> DataLoader:
    """Build a DataLoader from arrays.

    Parameters
    ----------
    states : np.ndarray
        Shape (N, window_size, n_features).
    actions : np.ndarray
        Shape (N,) integer labels.
    batch_size : int
    shuffle : bool

    Returns
    -------
    DataLoader
    """
    state_tensor = th.as_tensor(states, dtype=th.float32)
    action_tensor = th.as_tensor(actions, dtype=th.long)
    dataset = TensorDataset(state_tensor, action_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def remap_labels(actions: np.ndarray):
    """Remap sparse integer labels to 0-indexed contiguous class indices.

    Parameters
    ----------
    actions : np.ndarray
        Integer anomaly_type labels (may be non-contiguous).

    Returns
    -------
    tuple
        ``(remapped_actions, label_map)`` where label_map is a dict
        mapping original label -> new index.
    """
    unique_labels = sorted(np.unique(actions).tolist())
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    remapped = np.array([label_map[a] for a in actions], dtype=np.int64)
    return remapped, label_map


def main() -> None:
    """Train and evaluate the anomaly detection controller."""
    th.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------ load
    normalized_train_path = GC.SAVE_PATH / "states_normalized"
    normalized_test_path = GC.SAVE_PATH / "test_states_normalized"

    print("Loading training data...")
    train_states, train_actions = load_split(normalized_train_path)
    print(f"  Train: {train_states.shape}, actions: {np.unique(train_actions)}")

    print("Loading test data...")
    test_states, test_actions = load_split(normalized_test_path)
    print(f"  Test: {test_states.shape}, actions: {np.unique(test_actions)}")

    # The anomaly_type labels are 0..4 (contiguous), but verify
    all_labels = np.unique(np.concatenate([train_actions, test_actions]))
    print(f"All unique labels: {all_labels}")

    n_features = train_states.shape[2]
    window_size = train_states.shape[1]
    n_actions = len(all_labels)
    print(f"n_features={n_features}, window_size={window_size}, n_actions={n_actions}")

    # ------------------------------------------------------------------ model
    model = AnomalyModel(
        window_size=window_size,
        n_features=n_features,
        n_actions=n_actions,
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
    )

    # ------------------------------------------------------------------ data loaders
    train_loader = build_dataloader(train_states, train_actions, BATCH_SIZE, shuffle=True)
    test_loader = build_dataloader(test_states, test_actions, BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------ training
    optimizer = th.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining for {N_EPOCHS} epochs...")
    for epoch_idx in tqdm(range(N_EPOCHS), desc="Epoch"):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for state_batch, action_batch in train_loader:
            features = model.features_extractor(state_batch)
            logits = model.action_net(model.policy_net(features))
            loss = loss_fn(logits, action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        tqdm.write(f"Epoch {epoch_idx+1}/{N_EPOCHS} — train loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------ evaluation
    model.eval()
    all_true = []
    all_pred = []
    with th.no_grad():
        for state_batch, action_batch in test_loader:
            pred = model(state_batch)
            all_true.extend(action_batch.tolist())
            all_pred.extend(pred.tolist())

    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(all_true, all_pred, digits=4))

    # ------------------------------------------------------------------ save
    GC.SAVE_PATH.mkdir(parents=True, exist_ok=True)
    th.save(model.state_dict(), GC.CONTROLLER_PATH)
    print(f"Saved model weights to {GC.CONTROLLER_PATH}")


if __name__ == "__main__":
    main()
