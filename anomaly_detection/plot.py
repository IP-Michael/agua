"""Generate concept-based explanation bar charts for anomaly detection states.

Usage
-----
    python plot.py --idx 0 1 2
    python plot.py --idx 5 --class 3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from torch import nn

import global_constants as GC
from global_constants import AnomalyModel
from agua.concept_viz import get_concept_weights, create_plotting_data, load_concepts
from agua.linear_policy_model import ConceptPredictor


def load_embeddings(indices: List[int]) -> th.Tensor:
    """Load concept-projected controller embeddings for given test state indices.

    Parameters
    ----------
    indices : list of int
        State indices (matching filename numbering) in data/test_states/.

    Returns
    -------
    torch.Tensor
        Projected concept-bin embedding tensor of shape
        ``(N, N_CONCEPTS * len(BINS))``.
    """
    controller = AnomalyModel(
        window_size=10,
        n_features=GC.N_FEATURES,
        n_actions=GC.N_ACTIONS,
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
    )
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    controller.eval()

    embedding_projector = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS,
    )
    embedding_projector.load_state_dict(
        th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    )
    embedding_projector.eval()

    embeddings = []
    for idx in indices:
        file = GC.TEST_STATE_SAVE_PATH / f"state_{idx:07d}.npz"
        data = np.load(file)
        with th.no_grad():
            state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
            embedding = controller.features_extractor(state).detach().cpu()
        embeddings.append(embedding)

    embeddings = th.cat(embeddings, dim=0)
    return embedding_projector(embeddings)


def main() -> None:
    """Generate and save a concept-weight bar chart for the given test state indices."""
    parser = argparse.ArgumentParser(
        description="Plot concept-based explanation for anomaly detection states."
    )
    parser.add_argument(
        "--idx",
        type=int,
        nargs="+",
        required=True,
        help="One or more test state indices to explain.",
    )
    parser.add_argument(
        "--class",
        dest="class_idx",
        type=int,
        required=False,
        default=None,
        help="Target class index for explanation (default: argmax).",
    )
    args = parser.parse_args()
    indices = args.idx

    concept_embedding = load_embeddings(indices)

    final_projector = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    state_dict = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projector.load_state_dict(
        {"weight": state_dict["weight"], "bias": state_dict["bias"]}
    )
    final_projector.eval()

    concepts = load_concepts(GC.CONCEPT_NAME_FILE)
    weights = get_concept_weights(final_projector, concept_embedding, class_idx=args.class_idx)
    df = create_plotting_data(weights, concepts, GC.BINS)

    ax = df.plot(kind="barh", x="Concept", y="Weight")
    plt.tight_layout()

    idx_tag = "-".join(str(i) for i in indices)
    out_file = f"explanation_{idx_tag}_{args.class_idx}.png"
    plt.savefig(out_file)
    plt.close(ax.figure)
    print(f"Saved explanation plot to {out_file}")


if __name__ == "__main__":
    main()
