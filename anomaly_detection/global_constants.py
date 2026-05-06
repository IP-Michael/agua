"""Global constants and model definitions for the anomaly detection Agua domain."""

from pathlib import Path
from typing import Tuple, List
import torch as th
from torch import nn
import numpy as np
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

SELECTED_FEATURES_FILE = Path(__file__).parent / "data" / "selected_features.txt"
if SELECTED_FEATURES_FILE.exists():
    with open(SELECTED_FEATURES_FILE) as _f:
        N_FEATURES = len([l for l in _f if l.strip()])
else:
    N_FEATURES = 185


class AnomalyFeaturesExtractor(nn.Module):
    """1D convolutional feature extractor for anomaly detection windows.

    Parameters
    ----------
    window_size : int, optional
        Number of timesteps per window (default 10).
    n_features : int, optional
        Number of features per timestep (default N_FEATURES).
    policy_embedding_size : int, optional
        Output dimensionality (default 128).
    """

    def __init__(self, window_size: int = 10, n_features: int = None,
                 policy_embedding_size: int = 128):
        if n_features is None:
            n_features = N_FEATURES
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.policy_embedding_size = policy_embedding_size

        # Two Conv1d layers over the time dimension
        # Input: (B, window_size, n_features) -> permute -> (B, n_features, window_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # After flatten: 64 * window_size
        cnn_out_dim = 64 * window_size
        self.projection = nn.Sequential(
            nn.Linear(cnn_out_dim, policy_embedding_size),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Extract features from a batch of windows.

        Parameters
        ----------
        observations : torch.Tensor
            Shape ``(B, window_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Feature embedding of shape ``(B, policy_embedding_size)``.
        """
        # (B, T, F) -> (B, F, T) for Conv1d
        x = observations.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.projection(x)
        return x


class AnomalyModel(nn.Module):
    """Anomaly detection policy network with 1D CNN feature extractor.

    Parameters
    ----------
    window_size : int, optional
        Number of timesteps per window (default 10).
    n_features : int, optional
        Number of features per timestep (default 185).
    n_actions : int, optional
        Number of discrete anomaly classes (default 5).
    policy_embedding_size : int, optional
        Dimensionality of feature extractor output (default 128).
    """

    def __init__(self, window_size: int = 10, n_features: int = None,
                 n_actions: int = 5, policy_embedding_size: int = 128):
        if n_features is None:
            n_features = N_FEATURES
        super().__init__()
        self.features_extractor = AnomalyFeaturesExtractor(
            window_size=window_size,
            n_features=n_features,
            policy_embedding_size=policy_embedding_size,
        )
        self.policy_net = nn.Sequential(
            nn.Linear(policy_embedding_size, 128),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(128, n_actions)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        """Compute greedy (argmax) action for a batch of observations.

        Parameters
        ----------
        observation : torch.Tensor
            Batched window tensor of shape ``(B, window_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Long tensor of chosen action indices with shape ``(B,)``.
        """
        features = self.features_extractor(observation)
        action_logits = self.action_net(self.policy_net(features))
        action = th.argmax(action_logits, dim=1)
        return action


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAVE_PATH = Path(__file__).parent / "data"
CONCEPTS_FILE = SAVE_PATH / "concepts.txt"
CONCEPT_NAME_FILE = SAVE_PATH / "concept_names.txt"
PROMPT_FILE = SAVE_PATH / "prompt_skeleton.txt"
CONTROLLER_PATH = SAVE_PATH / "anomaly_model.pt"
SCALER_PATH = SAVE_PATH / "scaler.joblib"

STATE_SAVE_PATH = SAVE_PATH / "states_normalized"
TEST_STATE_SAVE_PATH = SAVE_PATH / "test_states_normalized"

STATE_DESCRIPTION_SAVE_PATH = SAVE_PATH / "input_descriptions"
CONCEPT_EMBEDDING_SAVE_PATH = SAVE_PATH / "concept_embeddings"
STATE_EMBEDDING_SAVE_PATH = SAVE_PATH / "input_embeddings"

OUTPUT_PROJECTION_SAVE_PATH = SAVE_PATH / "final_projection.pt"
EMBED_PROJECTION_SAVE_PATH = SAVE_PATH / "embed_projection.pt"
TRAINING_LOG_FILE = SAVE_PATH / "train_log.txt"

# ---------------------------------------------------------------------------
# Hyperparameters and model settings
# ---------------------------------------------------------------------------

N_ACTIONS = 5
IS_ACTION_DISCRETE = True

POLICY_EMBEDDING_SIZE = 128
EMBEDDING_SIZE = 100
BINS = [20, 60, 100]

MAX_NUM_STATES = 6335
NUM_TEST_STATES = 1584
N_QUERY_TOGETHER = 1

LLM_MODEL = "deepseek-r1:32b"
QUERY_EMBEDDING_MODEL = "BAAI/bge-m3"
DOC_EMBEDDING_MODEL = "BAAI/bge-m3"

TEST_FRACTION = 0.1
SEED = 42

MAX_INTRA_CONCEPT_SIMILARITY = 0.875

if CONCEPT_EMBEDDING_SAVE_PATH.exists():
    N_CONCEPTS = len(list(CONCEPT_EMBEDDING_SAVE_PATH.iterdir()))
else:
    N_CONCEPTS = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def split_state_files() -> Tuple[List[Path], List[Path]]:
    """Return shuffled (train, validation) state file paths.

    Returns
    -------
    tuple of list[Path]
        ``(train_files, val_files)`` after random shuffle with seed ``SEED``.
    """
    state_files = sorted(list(STATE_SAVE_PATH.iterdir()))
    rand = np.random.RandomState(seed=SEED)
    rand.shuffle(state_files)
    n_train_samples = int((1 - TEST_FRACTION) * len(state_files))
    train_states = state_files[:n_train_samples]
    val_states = state_files[n_train_samples:]
    return (train_states, val_states)


def load_test_states() -> List[Path]:
    """Return sorted list of test state file paths."""
    state_files = sorted(list(TEST_STATE_SAVE_PATH.iterdir()))
    return state_files
