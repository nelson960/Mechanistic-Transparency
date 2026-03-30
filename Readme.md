
# Prompt-Level Mechanistic Transparency (PLMT)

**From tracing token decisions in LLMs to toy-model circuits and the geometry of superposition.**

This project began as an infrastructure to explain *why* a model emits a specific token for a given prompt by tracing internal computation in `gpt2-small`. The active research object has now moved to a controlled toy-transformer KV-retrieval sandbox. The purpose of that pivot is to move beyond "which component mattered?" toward a human-readable internal algorithm, and then toward superposition and polysemanticity in a setting small enough to inspect end to end.

---

## The Research Hierarchy

I frame interpretability as a hierarchy of increasing difficulty. This repo has now made solid progress on the first level, partial progress on the second, and is using the toy-model sandbox to approach the third.

1.  **Tracing the Path (Solved):** Which tokens influenced this decision? (Our `gpt2-small` infrastructure solves this).
2.  **Naming the Features (The Barrier):** What does Neuron 456 conceptually mean? (In large models, this is obscured by **Polysemanticity**).
3.  **Understanding the Algorithm (The Goal):** How do features interact to perform logic? (This requires understanding **Superposition**).
4.  **Abstract Reasoning (The Frontier):** Is the model planning ahead? (We barely have tools for this yet).

---

## Research Evolution: Why The Pivot?

### Phase 1: The Tracing Infrastructure (Completed Foundation)
I built a reliable system to trace component contributions in `gpt2-small`.
*   **Capability:** I can identify *where* information flows (which heads, which layers).
*   **Limitation:** I cannot easily identify *what* the information means. I could tell you that Neuron 456 fired, but not whether it represents "financial bank" or "river bank" because that neuron likely represents both.

### Phase 2: The KV-Retrieve Toy Model (Current)
To move from tracing to algorithmic understanding, I shifted from large-model analysis to a deliberately small decoder-only transformer trained on a synthetic in-context KV-retrieval task.
*   **Strategy:** Train a tiny model on a task with known ground-truth structure and inspect the full retrieval circuit.
*   **Current goal:** Explain how the model uses heads, residual writes, and eventually features / neurons to map `query key -> matching pair -> value token`.
*   **Current status:** Circuit-level evidence is strong; feature-level and state-variable-level explanation are still incomplete.

### Phase 3: Feature Packing / Superposition (Next)
Once the retrieval algorithm is represented in human-readable internal variables, the next step is to study how features are packed, mixed, or disentangled in small models.
*   **Hypothesis:** Controlled toy settings will make superposition visible enough to inspect geometrically.
*   **Goal:** Understand how sparse features, neuron-level mechanisms, and compressed representations relate.

---

## Current Status (Active Research State)

![PLMT Overview](pics/image.png)

The active frontier is the `KV-Retrieve-3` toy model, not new GPT-2 tracing work. The GPT-2 tooling remains in the repo as infrastructure, but the current research state is:

**What is solid now:**
*   The dataset, model, checkpoint, and reload path are stable.
*   [`note.md`](note.md) is now the single canonical project summary.
*   [`notebook/kv_retrieve_algorithm_analysis.ipynb`](notebook/kv_retrieve_algorithm_analysis.ipynb) is now the single canonical toy-model notebook; the old split KV notebooks have been removed.
*   The best current circuit story is that `L1H0` writes a final-position control signal that changes `L2H0.Q`, while `L2H0` acts as the main downstream value-writing head and `L2H1` behaves more like a query/key-selection head.
*   [`scripts/run_single_prompt_program.py`](scripts/run_single_prompt_program.py) now produces an exact one-prompt stage/head/MLP report.
*   The live helper modules now sit in [`scripts/kv_retrieve_analysis.py`](scripts/kv_retrieve_analysis.py) and [`scripts/kv_retrieve_features.py`](scripts/kv_retrieve_features.py).

**What is only partial:**
*   The unified notebook shows that the narrowed `L1H0` site is much cleaner than the full residual stream, but the retrieval-control signal is still distributed across multiple features.
*   The repo does not yet contain a full human-readable prompt-to-response algorithm at the feature, neuron, or state-variable level.
*   Superposition is still a downstream target, not the current solved result.

---

## Active Toy-Model Entry Points

If you want the current research path, start here:

1.  [`note.md`](note.md)
2.  [`notebook/kv_retrieve_algorithm_analysis.ipynb`](notebook/kv_retrieve_algorithm_analysis.ipynb)
3.  [`scripts/run_single_prompt_program.py`](scripts/run_single_prompt_program.py)
4.  [`scripts/run_feature_basis_analysis.py`](scripts/run_feature_basis_analysis.py)

Support files for that path:

1.  [`scripts/kv_retrieve_analysis.py`](scripts/kv_retrieve_analysis.py)
2.  [`scripts/kv_retrieve_features.py`](scripts/kv_retrieve_features.py)

The following files still exist but are intentionally minimized and are no longer primary reading entrypoints:

1.  [`results_tracking.md`](results_tracking.md)
2.  [`PLMT_Methods_Draft.md`](PLMT_Methods_Draft.md)

---

## Historical GPT-2 Foundation

The sections below describe the original Phase 1 tracing infrastructure. They are still useful as background and tooling, but they are not the active research focus of the repo.

### How It Works (Phase 1)

For a prompt and next-token decision, the pipeline executes:

1.  Run a forward pass with activation hooks.
2.  Decompose logit margin into additive component contributions.
3.  Trace source tokens that drove key attention writes.
4.  Check faithfulness with internal ablations.
5.  Visualize structure + decision flow in the Web UI.

**In short:** `Prompt -> Internal Mechanism -> Token Decision`.

## Why GPT-2 Small

We use `gpt2-small` as the foundation for Phase 1 because:

1.  It is fast and cheap to run locally.
2.  It has the same core transformer blocks used in larger decoder-only LLMs.
3.  It is easy to instrument with `transformer-lens`.
4.  It is practical for debugging the full transparency pipeline before scaling to toy models.

---

## Historical GPT-2 Scripts

Historical GPT-2 tracing / viewer scripts:

1.  `scripts/build_interactive_model_viewer.py`
2.  `scripts/run_component_tracker.py`
3.  `scripts/visualize_3d_model_map.py`

## Quick Start (Historical GPT-2 Viewer Path)

### 1) Setup environment

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build single payload from one prompt

```bash
python scripts/build_interactive_model_viewer.py \
  --prompt "The secret code is 73914. Repeat the secret code exactly:"
```

Optional flags:

1.  `--out outputs/viewer_payload.json` (output path)
2.  `--prompt-id my_prompt_001` (stable run label)
3.  `--task copy` (task label metadata)
4.  `--model gpt2-small` (default is already `gpt2-small`)

Output:

1.  `outputs/viewer_payload.json`

Payload fields:

1.  `meta`: run metadata
2.  `graph`: 3D model nodes/edges
3.  `tracker`: prompt decision components/sources/paths
4.  `summary`: aggregate statistics

### 3) Start Web UI

```bash
cd webapp
python3 -m http.server 8000
```

Open:

1.  `http://localhost:8000`

Then load:

1.  `outputs/viewer_payload.json`
