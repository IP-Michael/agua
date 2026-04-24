import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch as th
from sklearn.metrics import classification_report
import global_constants as GC
from global_constants import ABRModel
from torch import nn
from agua.linear_policy_model import ConceptPredictor


def main() -> None:
    controller = ABRModel()
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

    final_projection = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    state_dict = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projection.load_state_dict({"weight": state_dict["weight"], "bias": state_dict["bias"]})
    final_projection.eval()

    test_files = GC.load_test_states()
    true_actions, pred_actions = [], []

    with th.no_grad():
        for file in test_files:
            data = np.load(file)
            state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
            controller_action = controller(state).item()
            features = controller.features_extractor(state)
            concept_emb = embed_projection(features)
            agua_logits = final_projection(concept_emb)
            agua_action = th.argmax(agua_logits, dim=1).item()
            true_actions.append(controller_action)
            pred_actions.append(agua_action)

    print(classification_report(true_actions, pred_actions, digits=4))


if __name__ == "__main__":
    main()
