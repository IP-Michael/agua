"""Generate natural language descriptions of anomaly detection windows via LLM.

Usage
-----
    python input_to_text.py

Iterates over training state files, converts each 10-timestep window to a
human-readable text summary, queries the LLM for a concept-grounded
description, and saves results to data/input_descriptions/.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
from openai import OpenAI
import openai
from tqdm.auto import tqdm
from multiprocessing.dummy import Pool
from time import sleep
from typing import Tuple, List

import global_constants as GC

# ---------------------------------------------------------------------------
# Load feature names and concepts at import time
# ---------------------------------------------------------------------------

def _load_feature_names() -> List[str]:
    """Load selected feature names from disk.

    Returns
    -------
    list of str
        Feature names in order.
    """
    with open(GC.SELECTED_FEATURES_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _load_concepts() -> str:
    """Load concept list as a single string.

    Returns
    -------
    str
        All concepts joined into one block.
    """
    with open(GC.CONCEPTS_FILE, "r") as f:
        return "".join(f.readlines())


FEATURE_NAMES: List[str] = _load_feature_names()
CONCEPTS: str = _load_concepts()

with open(GC.PROMPT_FILE, "r") as _pf:
    PROMPT: str = "".join(_pf.readlines())

CUSTOM_INSTRUCTIONS = (
    "You are a network engineer and systems analyst specializing in 5G RAN "
    "anomaly detection. You are gathering key information to use in an embedding "
    "model to identify patterns. Be straight to the point and avoid unnecessary words."
)


# ---------------------------------------------------------------------------
# Window to text
# ---------------------------------------------------------------------------

def window_to_str(window: np.ndarray, feature_names: List[str]) -> str:
    """Convert a (window_size, n_features) array to a human-readable description.

    Shows per-feature statistics: min, max, mean, and trend (last - first).

    Parameters
    ----------
    window : np.ndarray
        Shape ``(window_size, n_features)``.
    feature_names : list of str
        Feature names aligned with the second axis of ``window``.

    Returns
    -------
    str
        Multi-line textual description of the window.
    """
    lines = [f"Network telemetry window ({window.shape[0]} timesteps, {window.shape[1]} features):"]
    for feat_idx, feat_name in enumerate(feature_names):
        vals = window[:, feat_idx]
        feat_min = float(np.min(vals))
        feat_max = float(np.max(vals))
        feat_mean = float(np.mean(vals))
        trend = float(vals[-1] - vals[0])
        # Only include features with non-zero activity to keep the description compact
        if feat_max == 0.0 and feat_min == 0.0:
            continue
        lines.append(
            f"  {feat_name}: min={feat_min:.4g}, max={feat_max:.4g}, "
            f"mean={feat_mean:.4g}, trend={trend:+.4g}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM description
# ---------------------------------------------------------------------------

def get_llm_description(window: np.ndarray, client: OpenAI) -> str:
    """Query the LLM for a concept-grounded description of a window.

    Parameters
    ----------
    window : np.ndarray
        Shape ``(window_size, n_features)``.
    client : OpenAI
        OpenAI-compatible client pointing to the Ollama server.

    Returns
    -------
    str
        LLM-generated description.
    """
    window_text = window_to_str(window, FEATURE_NAMES)
    prompt = PROMPT.format(concepts=CONCEPTS, state_data=window_text)
    response = client.chat.completions.create(
        model=GC.LLM_MODEL,
        messages=[
            {"role": "system", "content": CUSTOM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _llm_wrapper(data: Tuple) -> Tuple[int, str]:
    """Thread-pool worker: generate or load description for one state file.

    Parameters
    ----------
    data : tuple
        ``(file_path, client)`` where file_path is a Path to a .npz state file.

    Returns
    -------
    tuple
        ``(state_idx, description)``
    """
    file_path, client = data
    # Parse index from filename
    stem = file_path.stem  # e.g. state_0000001
    state_idx = int(stem.split("_")[1])
    description_file = GC.STATE_DESCRIPTION_SAVE_PATH / f"state_{state_idx:07d}.txt"

    if description_file.exists():
        return state_idx, description_file.read_text().rstrip()

    window_data = np.load(file_path)
    window = window_data["state"]

    msg = None
    while msg is None:
        try:
            msg = get_llm_description(window=window, client=client)
            sleep(0.5)
        except (openai.RateLimitError, openai.APITimeoutError):
            sleep(1.5)

    return state_idx, msg


# ---------------------------------------------------------------------------
# Save descriptions
# ---------------------------------------------------------------------------

def save_input_descriptions() -> None:
    """Generate and save LLM descriptions for all training state files.

    Skips files that already have a corresponding description. Uses a thread
    pool of size ``GC.N_QUERY_TOGETHER`` for parallelism.
    """
    GC.STATE_DESCRIPTION_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=os.environ["OLLAMA_BASE_URL"],
        api_key="ollama",
        timeout=150,
    )

    state_files = sorted(GC.STATE_SAVE_PATH.glob("*.npz"))
    # Filter to files that don't yet have descriptions
    pending = []
    for f in state_files:
        stem = f.stem
        state_idx = int(stem.split("_")[1])
        desc_file = GC.STATE_DESCRIPTION_SAVE_PATH / f"state_{state_idx:07d}.txt"
        pending.append((f, client))

    pool = Pool(processes=GC.N_QUERY_TOGETHER)
    with tqdm(total=len(pending), desc="Querying LLM", leave=True) as pbar:
        for state_idx, description in pool.imap_unordered(_llm_wrapper, pending):
            description_file = GC.STATE_DESCRIPTION_SAVE_PATH / f"state_{state_idx:07d}.txt"
            if not description_file.exists():
                with open(description_file, "w") as f:
                    print(description, file=f)
            pbar.update()


if __name__ == "__main__":
    save_input_descriptions()
