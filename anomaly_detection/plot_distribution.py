"""Plot concept distribution across test states (similar to Figure 5 in the Agua paper).

Usage
-----
    python plot_distribution.py                 # overall proportion per concept
    python plot_distribution.py --by_class      # one subplot per anomaly class
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm

import global_constants as GC
from global_constants import AnomalyModel
from agua.concept_viz import get_concept_weights, load_concepts
from agua.linear_policy_model import ConceptPredictor


def _load_models():
    controller = AnomalyModel(
        window_size=10,
        n_features=GC.N_FEATURES,
        n_actions=GC.N_ACTIONS,
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
    )
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    controller.eval()

    embed_projection = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS,
    )
    embed_projection.load_state_dict(
        th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    )
    embed_projection.eval()

    final_projector = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    state_dict = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projector.load_state_dict(
        {"weight": state_dict["weight"], "bias": state_dict["bias"]}
    )
    final_projector.eval()

    return controller, embed_projection, final_projector


def compute_concept_proportions(controller, embed_projection, final_projector):
    """Return (per_sample_concept_proportions, labels) over all test states.

    For each sample, concept weights are summed over bins to get a per-concept
    score, then normalized so each sample contributes a probability distribution
    over concepts. The returned array has shape (N, n_concepts).
    """
    test_files = sorted(GC.TEST_STATE_SAVE_PATH.glob("*.npz"))

    all_proportions = []
    labels = []

    n_bins = len(GC.BINS)
    n_concepts = GC.N_CONCEPTS

    for f in tqdm(test_files, desc="Processing test states"):
        data = np.load(f)
        label = int(data["action"])
        with th.no_grad():
            state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
            embedding = controller.features_extractor(state).detach().cpu()
            concept_embed = embed_projection(embedding)
            weights = get_concept_weights(final_projector, concept_embed, class_idx=None)

        # weights: (1, n_bins * n_concepts) — layout is (bin0_c0, bin0_c1, ..., bin1_c0, ...)
        w = weights.abs().squeeze(0)                        # (n_bins * n_concepts,)
        w = w.reshape(n_bins, n_concepts).sum(dim=0)        # (n_concepts,) — sum over bins
        w = w / (w.sum() + 1e-12)                          # normalise to proportion
        all_proportions.append(w.numpy())
        labels.append(label)

    return np.stack(all_proportions), np.array(labels)      # (N, n_concepts), (N,)


def plot_overall(mean_proportions, concept_names, out_file="concept_distribution.png"):
    fig, ax = plt.subplots(figsize=(max(10, len(concept_names) * 0.9), 5))
    x = np.arange(len(concept_names))
    ax.bar(x, mean_proportions, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Proportion")
    ax.set_title("Concept Distribution — Test Set")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved {out_file}")


CLASS_LABELS = {
    0: "Normal (No Anomaly)",
    1: "Radio",   # fill in actual name, e.g. "Link Failure"
    2: "Network",   # fill in actual name, e.g. "High Load"
    3: "PDCP",   # fill in actual name
    4: "MAC",   # fill in actual name
}


def plot_by_class(all_proportions, labels, concept_names):
    classes = sorted(np.unique(labels).tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 2)))
    x = np.arange(len(concept_names))

    saved = []
    for cls, color in zip(classes, colors):
        mask = labels == cls
        mean_props = all_proportions[mask].mean(axis=0)
        label_name = CLASS_LABELS.get(cls, f"Class {cls}")
        safe_name = label_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        out_file = f"concept_distribution_class_{cls}_{safe_name}.png"

        fig, ax = plt.subplots(figsize=(max(10, len(concept_names) * 0.9), 5))
        ax.bar(x, mean_props, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Proportion")
        ax.set_title(f"Concept Distribution — {label_name}  (n={mask.sum()})")
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"Saved {out_file}")
        saved.append(out_file)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by_class", action="store_true",
                        help="Also plot a per-anomaly-class breakdown")
    args = parser.parse_args()

    concepts = load_concepts(GC.CONCEPT_NAME_FILE)
    concept_names = [desc.split(":")[0].strip() for _, desc in concepts]

    print("Loading models...")
    controller, embed_projection, final_projector = _load_models()

    print("Computing concept proportions for all test states...")
    all_proportions, labels = compute_concept_proportions(
        controller, embed_projection, final_projector
    )

    plot_overall(all_proportions.mean(axis=0), concept_names)

    if args.by_class:
        plot_by_class(all_proportions, labels, concept_names)



if __name__ == "__main__":
    main()
