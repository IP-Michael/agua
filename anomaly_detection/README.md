# Anomaly Detection — Agua Domain

This directory applies the [Agua](../README.md) concept-based explainability framework to 5G RAN telemetry anomaly detection. A 1D CNN controller classifies anomaly types from sliding windows of network KPIs; Agua then provides human-readable concept-level explanations of the controller's decisions.

---

## Dataset

**`combined_timeseries_8000.csv`** — 8,000 timesteps × 455 raw features, sampled at 150 ms intervals.

| Class | Label | Timesteps |
|-------|-------|-----------|
| 0 | Normal (No Anomaly) | 5,365 |
| 1 | Radio Anomaly | 1,000 |
| 2 | Network Anomaly | 342 |
| 3 | PDCP Anomaly | 1,000 |
| 4 | MAC Anomaly | 293 |

**`mac_timeseries.csv`** — Separate MAC-layer test dataset (classes 0 and 4 only, 7,823 rows). Used for out-of-distribution evaluation.

Features span four layers of the 5G RAN stack: backhaul/midhaul transport (BH/MH), MAC/RLC scheduler KPIs, L1 PHY metrics, and platform CPU thread runtimes.

---

## Concepts

15 human-defined concepts are stored in `data/concepts.txt` and `data/concept_names.txt`:

| # | Concept |
|---|---------|
| 1 | Stable Radio Channel |
| 2 | Balanced Resource Utilization |
| 3 | Healthy Compute Execution |
| 4 | Clean Fronthaul Transport |
| 5 | External RF Interference Signature |
| 6 | Forced Modulation Downgrade |
| 7 | Reactive Throughput Collapse |
| 8 | Error Rate Spike Under Interference |
| 9 | PDCP Processing Overload |
| 10 | MAC Scheduler Delay Accumulation |
| 11 | Thread Affinity Misconfiguration |
| 12 | Platform-Radio Decoupling |
| 13 | Fronthaul Link Saturation |
| 14 | Fronthaul-Induced PHY Degradation |
| 15 | eCPRI Frame Loss Pattern |

---

## Pipeline

Run the steps below in order. Each step's output feeds the next.

### Step 1 — Preprocess training data

```bash
python prepare_data.py
```

- Reads `combined_timeseries_8000.csv`
- Removes low-variance features (keeps 73 of 455)
- Fits and saves a `StandardScaler` to `data/scaler.joblib`
- Saves selected feature names to `data/selected_features.txt`
- Creates stride-1 windows (size 10, no label-boundary crossing)
- Shuffles with seed 42, splits 80/20
- Saves training windows to `data/states_normalized/` (6,335 files)

### Step 2 — Train the controller

```bash
python train_controller.py
```

- Trains `AnomalyModel` (1D CNN → 128-dim embedding → 5-class output) on `data/states_normalized/`
- Saves weights to `data/anomaly_model.pt`

### Step 3 — Prepare the test set (leak-free temporal split)

```bash
python prepare_test_temporal.py
```

- Holds out the last 20% of timesteps from each contiguous label run, with a 9-row gap at the boundary to prevent any window overlap with training
- Applies saved `scaler.joblib` (no refit) and `selected_features.txt`
- Saves 1,441 test windows to `data/test_states_normalized/`

  | Class | Test windows |
  |-------|-------------|
  | 0 — Normal | 985 |
  | 1 — Radio | 182 |
  | 2 — Network | 51 |
  | 3 — PDCP | 182 |
  | 4 — MAC | 41 |

### Step 4 — Generate text descriptions

```bash
python input_to_text.py
```

- Iterates over `data/states_normalized/` training windows
- Converts each 10-timestep window to a structured text summary via LLM
- Saves one `.txt` per window to `data/input_descriptions/` (6,335 files)
- Requires `LLM_MODEL` configured in `.env` (default: `deepseek-r1:32b`)

### Step 5 — Embed concepts and descriptions

```bash
python embed_data.py --save_concept_embeddings --save_sample_embeddings
```

- Embeds `data/concepts.txt` concept descriptions using `BAAI/bge-m3` → `data/concept_embeddings/` (15 files, 1024-dim each)
- Embeds `data/input_descriptions/` text files → `data/input_embeddings/` (6,335 files)

### Step 6 — Train Agua mapping functions

```bash
python train_model.py --embedding_to_embedding
python train_model.py --linear_policy
```

**`--embedding_to_embedding`**: Trains `ConceptPredictor` — maps controller's 128-dim policy embeddings to a (n_bins × n_concepts) concept distribution. Supervised by cosine similarity between state text embeddings and concept embeddings. Saves to `data/embed_projection.pt`.

**`--linear_policy`**: Trains a linear output projection — maps concept-bin activations to action logits. Evaluates fidelity on test set. Saves to `data/final_projection.pt` and logs to `data/train_log.txt`.

---

## Evaluation

### Fidelity and ground-truth accuracy

```bash
python eval_fidelity.py
```

Prints three reports:
- **Fidelity**: Controller vs Agua surrogate agreement (how well the surrogate mimics the controller)
- **Controller accuracy**: Controller predictions vs ground-truth labels
- **Agua accuracy**: Agua surrogate predictions vs ground-truth labels

Current results on the temporal-split test set (1,441 samples): **100% accuracy and fidelity** across all five classes.

### Concept distribution plots

```bash
python plot_distribution.py                # overall mean proportion across all test samples
python plot_distribution.py --by_class     # one PNG per anomaly class
```

Saves per-class PNGs (e.g., `concept_distribution_class_4_mac.png`) showing the mean concept contribution proportion for each class.

To update class label names, edit the `CLASS_LABELS` dict at the top of `plot_distribution.py`.

### Per-sample explanation chart

```bash
python plot.py --idx 0 1 2
python plot.py --idx 5 --class 3
```

Generates a bar chart of top concept-bin contributions for specific test samples.

---

## Data directory layout

```
data/
├── anomaly_model.pt          # Trained 1D CNN controller weights
├── embed_projection.pt       # Trained ConceptPredictor weights
├── final_projection.pt       # Trained linear output projection weights
├── scaler.joblib             # StandardScaler fitted on training features
├── selected_features.txt     # 73 selected feature names
├── concepts.txt              # Full concept descriptions (15 concepts)
├── concept_names.txt         # Short concept names
├── prompt_skeleton.txt       # LLM prompt template for input_to_text.py
├── train_log.txt             # Training log from train_model.py
├── states_normalized/        # Training windows (6,335 × .npz)
├── test_states_normalized/   # Test windows — temporal split (1,441 × .npz)
├── input_descriptions/       # LLM text descriptions (6,335 × .txt)
├── input_embeddings/         # BAAI/bge-m3 embeddings of descriptions (6,335 × .npz)
└── concept_embeddings/       # BAAI/bge-m3 embeddings of concepts (15 × .npz)
```

Each `.npz` window file contains:
- `state`: float32 array of shape `(10, 73)` — 10-timestep window, 73 features, normalized
- `action`: int — ground-truth anomaly class label (0–4)

---

## Key design notes

- **No data leakage**: Training and test windows come from disjoint timestep regions. The temporal split ensures no test window shares any timestep with a boundary training window (9-row gap = `window_size − 1`).
- **Scaler applied once**: `scaler.joblib` is fitted on training data only and applied to test data without refitting.
- **Training data is shuffled**: `data/states_normalized/` files are in shuffled order (seed 42). Test data in `data/test_states_normalized/` preserves temporal order.
- **Feature selection**: 73 features retained from 455 raw columns based on variance threshold during `prepare_data.py`.
