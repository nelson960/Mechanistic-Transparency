
# Prompt-Level Mechanistic Transparency (PLMT)

**From tracing token decisions in LLMs to understanding the geometry of superposition.**

This project began as an infrastructure to explain *why* a model emits a specific token for a given prompt by tracing internal computation in `gpt2-small`. However, the research evolved when we hit the fundamental barrier of **Polysemanticity**.

We are now expanding the project to include **Controlled Toy Models**. By training small networks on deliberate, synthetic datasets, we aim to visually and mathematically understand how models compress concepts into limited dimensions (superposition).

---

## The Research Hierarchy

We frame interpretability as a hierarchy of increasing difficulty. This project addresses the first level and builds tools to attack the second.

1.  **Tracing the Path (Solved):** Which tokens influenced this decision? (Our `gpt2-small` infrastructure solves this).
2.  **Naming the Features (The Barrier):** What does Neuron 456 conceptually mean? (In large models, this is obscured by **Polysemanticity**).
3.  **Understanding the Algorithm (The Goal):** How do features interact to perform logic? (This requires understanding **Superposition**).
4.  **Abstract Reasoning (The Frontier):** Is the model planning ahead? (We barely have tools for this yet).

---

## Research Evolution: Why The Pivot?

### Phase 1: The Tracing Infrastructure (Current)
We built a reliable system to trace component contributions in `gpt2-small`.
*   **Capability:** We can identify *where* information flows (which heads, which layers).
*   **Limitation:** We cannot easily identify *what* the information means. We could tell you that Neuron 456 fired, but not whether it represents "financial bank" or "river bank" because that neuron likely represents both.

### Phase 2: The Toy Model Approach (Next)
To truly understand **Superposition**, we are moving from analyzing complex, opaque models to training simple, transparent ones.
*   **Strategy:** Train small architectures with controlled synthetic datasets where we define the "ground truth" features.
*   **Hypothesis:** By forcing a model to store more features than it has dimensions, we can observe **Superposition** directly.
*   **Goal:** Visualize how concepts are geometrically arranged in embedding space and disentangle them using Sparse Autoencoders (SAEs).

---

## Current Status (Phase 1 Infrastructure)

![PLMT Overview](pics/image.png)

The current codebase provides robust mechanistic tracing for `gpt2-small`. We can reliably track which internal components activate during inference for a given prompt and visualize residual-flow-related signals. We can also measure component contributions and run targeted ablations.

**What this system does:**
*   Traces the causal path from prompt to token.
*   Visualizes residual stream updates.
*   Performs ablation studies to verify feature importance.

**What it does not do (yet):**
*   Provide a complete causal explanation of *why* a specific choice was made over all alternatives in a human-interpretable way (due to the superposition problem).

---

## How It Works (Phase 1)

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

## Repo Flow

Current core scripts:

1.  `scripts/build_interactive_model_viewer.py`
2.  `scripts/run_component_tracker.py`
3.  `scripts/visualize_3d_model_map.py`

## Quick Start (End To End)

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