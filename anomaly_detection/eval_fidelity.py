"""Evaluate fidelity of the Agua concept-based explanations vs the controller.

Usage
-----
    python eval_fidelity.py

Loads the trained controller, embed_projection, and final_projection, then
runs all test states through both and prints a classification report comparing
controller predictions to Agua predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch as th
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

import global_constants as GC
from global_constants import AnomalyModel
from agua.linear_policy_model import ConceptPredictor


def main() -> None:
    """Run fidelity evaluation and print classification report."""
    # ------------------------------------------------------------------ load controller
    controller = AnomalyModel(
        window_size=10,
        n_features=GC.N_FEATURES,
        n_actions=GC.N_ACTIONS,
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
    )
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    controller.eval()

    # ------------------------------------------------------------------ load embed projection
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

    # ------------------------------------------------------------------ load final projection
    final_projection = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    state_dict = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projection.load_state_dict({"weight": state_dict["weight"], "bias": state_dict["bias"]})
    final_projection.eval()

    # ------------------------------------------------------------------ evaluate
    test_files = GC.load_test_states()
    true_actions, pred_actions = [], []

    ground_truth = []

    with th.no_grad():
        for file in test_files:
            data = np.load(file)
            gt_label = int(data["action"])
            state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
            controller_action = controller(state).item()
            features = controller.features_extractor(state)
            concept_emb = embed_projection(features)
            agua_logits = final_projection(concept_emb)
            agua_action = th.argmax(agua_logits, dim=1).item()
            ground_truth.append(gt_label)
            true_actions.append(controller_action)
            pred_actions.append(agua_action)

    classes = sorted(set(ground_truth))
    class_names = [str(c) for c in classes]

    print("=== Fidelity Report: Controller vs Agua Predictions ===")
    print(classification_report(true_actions, pred_actions, digits=4))

    print("\n--- Fidelity Confusion Matrix (rows=Controller, cols=Agua) ---")
    cm = confusion_matrix(true_actions, pred_actions, labels=classes)
    header = f"{'':>8}" + "".join(f"  Agua:{c:>2}" for c in classes)
    print(header)
    for i, row_cls in enumerate(classes):
        row = f"Ctrl:{row_cls:>2} |" + "".join(f"{cm[i,j]:>10}" for j in range(len(classes)))
        print(row)

    print("\n=== Ground Truth Accuracy: Controller ===")
    print(classification_report(ground_truth, true_actions, digits=4, target_names=[f"Class {c}" for c in classes]))

    print("--- Controller Confusion Matrix (rows=GT, cols=Predicted) ---")
    cm_ctrl = confusion_matrix(ground_truth, true_actions, labels=classes)
    print(header.replace("Agua", "Pred"))
    for i, row_cls in enumerate(classes):
        row = f"  GT:{row_cls:>2} |" + "".join(f"{cm_ctrl[i,j]:>10}" for j in range(len(classes)))
        print(row)

    print("\n=== Ground Truth Accuracy: Agua Surrogate ===")
    print(classification_report(ground_truth, pred_actions, digits=4, target_names=[f"Class {c}" for c in classes]))

    print("--- Agua Confusion Matrix (rows=GT, cols=Predicted) ---")
    cm_agua = confusion_matrix(ground_truth, pred_actions, labels=classes)
    print(header)
    for i, row_cls in enumerate(classes):
        row = f"  GT:{row_cls:>2} |" + "".join(f"{cm_agua[i,j]:>10}" for j in range(len(classes)))
        print(row)


if __name__ == "__main__":
    main()
