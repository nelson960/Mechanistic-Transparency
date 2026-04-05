---
layout: default
title: Retrieval Motif Emergence in Small Transformers
description: Empirical evidence for staged retrieval motif formation in small transformers trained on controlled in-context retrieval tasks.
permalink: /
---

# Mechanistic Transformer Circuits

<p class="lead">Controlled in-context retrieval benchmarks, checkpoint-by-checkpoint mechanistic measurements, and a full 150-epoch six-run matrix for studying circuit emergence in small transformers.</p>

## Abstract

This project asks a simple question in a setting small enough to inspect directly:

**can a small transformer learn a human-readable in-context retrieval mechanism, and can that mechanism be tracked as it emerges during training?**

The answer supported by the experiments here is yes, with an important qualification. The learned mechanism is not best described as one universal fixed head graph. It is better described as a **stable retrieval motif**:

- an upstream support or setup computation
- a downstream retrieval or write computation
- strong query-key and selected-value representations
- a matching step that is real but often more distributed than a single neat probe variable

This page tells that story from top to bottom. It starts with the smallest symbolic benchmark where a retrieval circuit can be reverse-engineered after training, checks the measurement stack on a benchmark that should fail if the model is only memorizing, and then moves to a larger textual retrieval benchmark where the same circuit family can be followed across checkpoints, seeds, and curricula.

## Main Result

The strongest claim supported by the full evidence is:

**small transformers repeatedly learn a staged retrieval motif, not one fixed universal head graph.**

That motif has four recurring properties:

- a downstream layer-2 retrieval or write role that is the most stable part of the mechanism
- an additional upstream support or setup computation
- very strong query-key and selected-value structure by the end of training
- a slot-matching computation that is real but often more distributed and later-settling than the other parts

## Question

The project studies two empirical questions:

1. What retrieval mechanism appears inside a small transformer on a controlled in-context retrieval task?
2. Which parts of that mechanism stay stable across training, seeds, and curricula?

The project does **not** yet claim a mathematical derivation of why gradient descent builds that motif. The current evidence is empirical: what forms, when it forms, what stays stable, and where the interpretation breaks down.

These experiments sit inside a broader mechanistic agenda: where useful computation is localized, why optimization discovers some circuits and not others, how internal structure shapes outputs, and when compact models become polysemantic by reusing the same representational directions for multiple functions.

## Shared Experimental Setup

Across the training-dynamics experiments, the repository uses a small decoder-only transformer with:

- causal self-attention
- RoPE position encoding
- RMSNorm
- SwiGLU MLP blocks
- tied embeddings and output logits

Core model implementation:

- [`scripts/tiny_transformer_core.py`]({{ site.repository_url }}/blob/main/scripts/tiny_transformer_core.py)

The standard workflow is:

1. train a model while logging optimization telemetry
2. save checkpoint `0`, dense early checkpoints, and the final checkpoint
3. run a mechanistic battery on each saved checkpoint
4. aggregate checkpoint artifacts into run-level and grouped summaries
5. render figures from those summaries

Core scripts:

- [`scripts/train_run.py`]({{ site.repository_url }}/blob/main/scripts/train_run.py)
- [`scripts/run_checkpoint_battery.py`]({{ site.repository_url }}/blob/main/scripts/run_checkpoint_battery.py)
- [`scripts/summarize_training_dynamics.py`]({{ site.repository_url }}/blob/main/scripts/summarize_training_dynamics.py)
- [`scripts/visualize_kv_circuit_dynamics.py`]({{ site.repository_url }}/blob/main/scripts/visualize_kv_circuit_dynamics.py)

## What Is Recorded

Per training step the runs record:

- batch loss
- gradient norm
- parameter norms
- update norms
- parameter-group and head-slice update metrics

Per checkpoint the battery records:

- train / val / test / OOD behavior
- variable probes
- variable faithfulness
- routing and copy scores
- head ablation / localization
- neuron summaries
- sparse feature summaries
- superposition metrics
- representation drift
- operator handoffs

These measurements answer different questions:

- behavior asks whether the model really solves the task on held-out and OOD data
- probes ask whether a representation contains decodable task information
- faithfulness asks whether that information is causally useful rather than merely readable
- operator scores ask whether a head looks routing-like or copy-like
- localization asks whether ablating a component actually hurts the behavior
- drift and handoffs ask whether the mechanism stabilizes or keeps reorganizing

## 1. Symbolic KV Retrieval

The first benchmark is the symbolic key-value task in [`dataset/phase2/kv_retrieve_3`]({{ site.repository_url }}/tree/main/dataset/phase2/kv_retrieve_3).

Prompt format:

```text
<bos> K3 V4 ; K1 V7 ; K5 V0 ; Q K5 ->
```

Target:

```text
V0
```

Dataset details:

- generator seed `7`
- `8` keys and `8` values
- splits: `5000` train / `500` val / `500` test / `500` OOD
- standard prompt length: `3` key-value pairs
- OOD split: `4` key-value pairs
- pair order shuffled inside each prompt

Analyzed checkpoint:

- [`models/kv_retrieve_3/selected_checkpoint.pt`]({{ site.repository_url }}/blob/main/models/kv_retrieve_3/selected_checkpoint.pt)
- `2` layers
- `2` heads per layer
- `d_model = 48`
- `d_ff = 96`
- `max_seq_len = 16`

Selected checkpoint performance:

| Split | Accuracy | Mean Margin |
| --- | ---: | ---: |
| Train | `0.9978` | `12.7831` |
| Val | `0.9520` | `11.7387` |
| Test | `0.9560` | `11.7673` |
| OOD 4-pair | `0.7280` | `4.8225` |

Canonical analysis:

- [`notebook/kv_retrieve_algorithm_analysis.ipynb`]({{ site.repository_url }}/blob/main/notebook/kv_retrieve_algorithm_analysis.ipynb)
- [`notebook/kv_retrieve_algorithm_discovery.ipynb`]({{ site.repository_url }}/blob/main/notebook/kv_retrieve_algorithm_discovery.ipynb)

The symbolic benchmark provides the first strong evidence for a retrieval circuit of the form:

`layer-1 support -> layer-2 query retargeting -> layer-2 value write`

This establishes that the project is studying a real learned in-context retrieval mechanism, not a fake toy where the model can solve the task without matching the current prompt.

## 2. Story Next-Token Negative Control

The second benchmark is a single-story next-token task. It is used as a negative control.

Setup:

- one fixed story token stream with `1168` tokens
- context length `24`
- longer-context OOD length `40`
- derived split sizes: `910` train / `117` val / `117` test / `117` OOD
- `2` layers
- `2` heads
- `d_model = 32`
- `d_ff = 64`
- `150` epochs
- batch size `32`
- learning rate `0.01`
- device `cpu`
- total saved checkpoints summarized: `47`

Key metrics:

| Checkpoint | Val Acc | Test Acc | OOD Acc |
| --- | ---: | ---: | ---: |
| Best held-out (`epoch 1`) | `0.0940` | `0.0769` | `0.0769` |
| Final | `0.0427` | `0.0256` | `0.0256` |

### Behavior and Variables

<figure class="paper-figure">
  <img src="figures/story_behavior_and_variables.png" alt="Story benchmark behavior and variable traces">
  <figcaption><strong>Figure 1.</strong> The story negative control overfits instead of forming a clean mechanism: train behavior improves, held-out behavior collapses, and the tracked variables never settle into a faithful generalizing circuit.</figcaption>
</figure>

### Operators and Localization

<figure class="paper-figure">
  <img src="figures/story_operator_and_localization.png" alt="Story benchmark operator and localization summaries">
  <figcaption><strong>Figure 2.</strong> Operator scores and localization in the story benchmark stay weak and inconsistent, which is what the harness should report when the task mainly invites memorization.</figcaption>
</figure>

This benchmark fails in exactly the way it should if the model mainly memorizes:

- train accuracy rises to `1.0`
- held-out accuracy collapses instead of improving together
- the summaries show no convincing behavior, routing, copy, or faithfulness births

That failure matters. It shows the measurement stack does not automatically invent a clean mechanism story when the task is bad.

## 3. Textual KV Retrieval

The main benchmark keeps retrieval algorithmically clean while making memorization much harder.

Prompt format:

```text
<bos> Mara amber ; Ivo cedar ; Sera linen ; Q Ivo ->
```

Target:

```text
cedar
```

Dataset design:

- generator seed `17`
- `16` keys and `16` values
- splits: `30000` train 2-pair / `30000` train 3-pair / `3000` val / `3000` test / `3000` OOD 4-pair
- balanced queried-slot coverage
- prompt deduplication across splits
- pair order shuffled
- distinct keys and values within each example
- OOD increases the context from `3` pairs to `4` pairs

Main model family:

- `2` layers
- `2` heads
- `d_model = 64`
- `d_ff = 128`
- batch size `64`
- learning rate `0.001`
- weight decay `0.0`
- device `cpu`

Birth thresholds used in the summaries:

- behavior val accuracy `0.9`
- operator score `0.85`
- variable score `0.85`
- faithfulness score `0.85`

### Pilot Run

The pilot is the first clear positive emergence result in the repository.

At epoch `30`:

- val accuracy: `0.9393`
- test accuracy: `0.9347`
- OOD accuracy: `0.8457`

Final pilot localization:

- ablating `block1_head0` drops accuracy by `0.6668`
- ablating `block2_head1` drops accuracy by `0.5348`

That already points to the same high-level decomposition seen in the symbolic benchmark:

- a strong layer-1 support or setup role
- a strong layer-2 retrieval or write role

### Pilot Circuit Emergence

<figure class="paper-figure">
  <img src="figures/textual_kv_circuit_emergence.png" alt="Pilot textual KV circuit emergence">
  <figcaption><strong>Figure 3.</strong> The pilot does not improve smoothly. Held-out behavior rises sharply once routing, copy, and variable structure coordinate over a narrow transition window late in training.</figcaption>
</figure>

### Pilot Network Change

<figure class="paper-figure">
  <img src="figures/textual_kv_network_change.png" alt="Pilot textual KV network change">
  <figcaption><strong>Figure 4.</strong> Optimization reorganizes the network before the strongest behavioral gains appear. The pilot spends a long time preparing partial structure before the retrieval mechanism locks in.</figcaption>
</figure>

### Pilot Operators and Localization

<figure class="paper-figure">
  <img src="figures/textual_kv_operator_and_localization.png" alt="Pilot textual KV operator and localization summaries">
  <figcaption><strong>Figure 5.</strong> By the end of the pilot, one layer-1 head and one layer-2 head dominate the mechanistic picture, supporting the interpretation of an upstream support role and a downstream retrieval-write role.</figcaption>
</figure>

### Pilot Representation Drift

<figure class="paper-figure">
  <img src="figures/textual_kv_representation_drift.png" alt="Pilot textual KV representation drift">
  <figcaption><strong>Figure 6.</strong> Representation drift remains substantial during assembly and then becomes more structured as the pilot settles into a stronger retrieval solution.</figcaption>
</figure>

The pilot is important because it shows a narrow transition window rather than vague gradual improvement.

#### Pilot Transition Window

| Epoch | Val Acc | Test Acc | OOD Acc | Routing Head | Routing Score | Copy Head | Copy Score |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| `23` | `0.3857` | `0.3657` | `0.2650` | `block2_head1` | `0.418` | `block1_head0` | `0.772` |
| `24` | `0.5993` | `0.5797` | `0.4483` | `block2_head1` | `0.595` | `block1_head0` | `0.772` |
| `25` | `0.7180` | `0.7027` | `0.5693` | `block2_head1` | `0.682` | `block1_head0` | `0.822` |
| `26` | `0.8040` | `0.8027` | `0.6727` | `block2_head1` | `0.751` | `block2_head1` | `0.914` |
| `27` | `0.8767` | `0.8643` | `0.7677` | `block2_head1` | `0.805` | `block2_head1` | `0.914` |
| `28` | `0.9193` | `0.9213` | `0.8330` | `block2_head1` | `0.848` | `block2_head1` | `0.951` |
| `29` | `0.9300` | `0.9343` | `0.8587` | `block2_head1` | `0.850` | `block2_head1` | `0.973` |
| `30` | `0.9393` | `0.9347` | `0.8457` | `block2_head1` | `0.905` | `block2_head1` | `0.991` |

This is one of the clearest pieces of evidence in the project. The run does not improve smoothly. The parts needed for retrieval become coordinated over a short window, and held-out behavior rises rapidly once that coordination locks in.

## Full 150-Epoch Six-Run Matrix

The main experiment is a `2 x 3` matrix:

- curriculum on, seeds `0`, `1`, `2`
- curriculum off, seeds `0`, `1`, `2`

`curriculum_on`:

- epochs `1` to `30`: `2`-pair only
- epochs `31` to `60`: mixed `2`-pair and `3`-pair
- epochs `61` to `150`: `3`-pair only

`curriculum_off`:

- epochs `1` to `150`: `3`-pair only

Matrix details:

- `6` runs
- `150` epochs per run
- same architecture as the pilot
- dense checkpoints through epoch `30`, then `24` log-spaced checkpoints, plus epoch `0` and final
- summarized checkpoints:
  - curriculum on: `216`
  - curriculum off: `190`

Run manifests:

- [`manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_on_d64_l2.json`]({{ site.repository_url }}/blob/main/manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_on_d64_l2.json)
- [`manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_on_d64_l2.json`]({{ site.repository_url }}/blob/main/manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_on_d64_l2.json)
- [`manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_on_d64_l2.json`]({{ site.repository_url }}/blob/main/manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_on_d64_l2.json)
- [`manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_off_d64_l2.json`]({{ site.repository_url }}/blob/main/manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_off_d64_l2.json)
- [`manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_off_d64_l2.json`]({{ site.repository_url }}/blob/main/manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_off_d64_l2.json)
- [`manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_off_d64_l2.json`]({{ site.repository_url }}/blob/main/manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_off_d64_l2.json)

### Grouped Outcome Table

| Condition | Val Acc | Test Acc | OOD Acc | Query Key | Matching Slot | Selected Value | Routing | Copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curriculum on | `1.0000` | `1.0000` | `0.9921` | `0.9997` | `0.4583` | `0.9997` | `0.9729` | `0.9293` |
| Curriculum off | `0.9999` | `1.0000` | `0.9993` | `1.0000` | `0.5339` | `0.9999` | `0.9999` | `0.9521` |

Both conditions solve the task almost perfectly. That rules out one easy story: curriculum is not required for success.

### Grouped Matrix Overview

<figure class="paper-figure">
  <img src="figures/textual_kv_full_matrix_overview.png" alt="Grouped full-matrix overview">
  <figcaption><strong>Figure 7.</strong> Both curricula solve the task almost perfectly. The grouped difference is not whether retrieval works, but how cleanly the matching computation settles compared with query and selected-value structure.</figcaption>
</figure>

The grouped comparison makes two points visible immediately:

- held-out behavior is near-perfect in both conditions
- the main internal difference is not query or selected-value quality, but the weaker and more variable matching step

### Per-Seed Final Checkpoints

| Condition | Seed | Val Acc | Test Acc | OOD Acc | Top Ablation Head | Routing Head | Copy Head |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| on | `0` | `1.0000` | `1.0000` | `1.0000` | `block1_head0` (`0.807`) | `block2_head1` (`1.000`) | `block2_head1` (`0.857`) |
| on | `1` | `1.0000` | `1.0000` | `0.9967` | `block1_head0` (`0.446`) | `block2_head1` (`1.000`) | `block2_head1` (`0.952`) |
| on | `2` | `1.0000` | `1.0000` | `0.9797` | `block2_head0` (`0.787`) | `block2_head0` (`0.919`) | `block2_head0` (`0.979`) |
| off | `0` | `1.0000` | `1.0000` | `0.9990` | `block1_head0` (`0.698`) | `block2_head0` (`1.000`) | `block2_head1` (`0.981`) |
| off | `1` | `0.9997` | `1.0000` | `0.9990` | `block1_head0` (`0.729`) | `block2_head0` (`1.000`) | `block2_head0` (`0.875`) |
| off | `2` | `1.0000` | `1.0000` | `1.0000` | `block1_head1` (`0.683`) | `block2_head0` (`1.000`) | `block2_head0` (`1.000`) |

This table is the concrete basis for the phrase **stable motif**.

What is stable:

- all six final runs end with a strong layer-2 retrieval role
- all six final runs end with at least one additional strongly causal component besides the dominant retrieval role
- query and selected-value representations are very strong by the end

What is not stable:

- the exact head identity of the strongest non-retrieval component
- the exact layer-2 head identity in every seed
- the exact timing at which routing, copy, and faithfulness cross threshold

### Final Localization Across All Six Runs

<figure class="paper-figure">
  <img src="figures/textual_kv_full_matrix_localization.png" alt="Final localization across all six runs">
  <figcaption><strong>Figure 8.</strong> Final head ablations across all six runs show a stable downstream layer-2 retrieval effect and a less stable upstream support implementation that moves across seeds.</figcaption>
</figure>

This heatmap is the clearest compact view of the role-level result:

- every run ends with one or two strongly causal heads
- the strongest downstream retrieval effect stays in layer 2
- the strongest upstream support effect moves across seeds

### Emergence Is Staged, Not Universal

| Condition | Seed | Behavior Birth | Copy Birth | Routing Birth | Matching-Slot Faithfulness Birth |
| --- | ---: | ---: | ---: | ---: | ---: |
| on | `0` | `28` | `31` | `114` | `99` |
| on | `1` | `15` | `36` | `31` | `122` |
| on | `2` | `37` | `62` | `—` | `40` |
| off | `0` | `12` | `93` | `122` | `57` |
| off | `1` | `17` | `1` | `17` | `—` |
| off | `2` | `20` | `17` | `131` | `—` |

There is no single epoch where "the circuit appears." The emergence tables show:

- behavior birth varies substantially across seeds
- copy can appear before routing
- faithfulness can appear after behavior
- some runs solve the task cleanly even when one strict internal birth never occurs

### Emergence Timing Across Seeds

<figure class="paper-figure">
  <img src="figures/textual_kv_full_matrix_emergence.png" alt="Emergence timing across all six runs">
  <figcaption><strong>Figure 9.</strong> Emergence timing varies substantially across seeds. There is no single universal epoch of circuit birth: behavior, copy, routing, and matching faithfulness cross their thresholds at different times.</figcaption>
</figure>

The emergence heatmap makes the staged character of formation hard to miss:

- behavior birth is early in some runs and late in others
- copy can become clean very early while routing stays below threshold for much longer
- matching-slot faithfulness is the least consistent internal milestone

### Additional Signals Hidden In The Summaries

| Signal | Curriculum On | Curriculum Off | Why It Matters |
| --- | --- | --- | --- |
| Final routing role identity | `block2_head1` in `2/3` seeds, `block2_head0` in `1/3` | `block2_head0` in `3/3` seeds | downstream routing is the most stable role in the project |
| Routing score-profile cosine across seeds | `> 0.97` in every pairwise comparison | `~0.9997` to `~0.9999` | role stability is stronger than raw head-name stability |
| Mean operator handoffs, copy | `11` | `3` | curriculum-on runs reorganize more while assembling the mechanism |
| Mean operator handoffs, routing | `12` | `9` | routing keeps sharpening and sometimes reassigning late |
| Matching-slot faithfulness births | `3/3` runs | `1/3` runs | curriculum mainly affects the cleanliness of the matching step |
| Hardest final behavior family | usually `longer_context_ood` | usually `longer_context_ood`, except one mild `value_permutation` weakness | the main remaining difficulty is context-length generalization |

### Stability, Turnover, And Birth Summary

<figure class="paper-figure">
  <img src="figures/textual_kv_full_matrix_stability.png" alt="Full-matrix stability, turnover, and grouped birth summary">
  <figcaption><strong>Figure 10.</strong> Routing is more stable at the score-profile level than raw head identities suggest. Curriculum-on runs turn over more while assembling the mechanism, but they also make matching-slot faithfulness more consistent.</figcaption>
</figure>

This summary figure compresses the grouped evidence behind the main interpretation:

- routing is more stable at the profile level than raw head names alone suggest
- curriculum-on runs reorganize more while assembling the mechanism
- curriculum mainly improves the consistency of matching-slot faithfulness, not the ability to solve the task

## Empirical Answer: How The Circuit Forms

The current evidence supports the following empirical formation story.

### 1. The circuit does not appear all at once

The model does not jump from "no mechanism" to "finished mechanism" in one clean step.

What the matrix shows instead:

- behavior can rise before every internal metric becomes clean
- copy-like structure often becomes clean before routing-like structure
- query and selected-value information can be strong even while matching-slot is still messy

### 2. The most stable part of formation is the downstream layer-2 retrieval role

Across all six final `150`-epoch runs:

- the final routing candidate is a layer-2 head in all six runs
- the final copy candidate is a layer-2 head in all six runs

So the most reliable answer to "what forms" in the matrix is the repeated emergence of a downstream layer-2 retrieval or write role.

### 3. The upstream part is real, but less head-stable

The matrix also shows a second important causal component besides the dominant layer-2 head, but not with one universal head identity:

- `on0`: strongest ablation is `block1_head0`
- `on1`: strongest ablation is `block1_head0`
- `on2`: strongest ablation is `block2_head0`
- `off0`: strongest ablation is `block1_head0`
- `off1`: strongest ablation is `block1_head0`
- `off2`: strongest ablation is `block1_head1`

So the strongest supported claim is:

- stable role-level motif
- unstable exact head-level implementation

### 4. The matching step is the hardest part to cleanly localize

`matching_slot` is the weakest part of the internal story.

What the matrix shows:

- `variable_matching_slot` never reaches birth threshold in any of the six full runs
- `faithfulness_matching_slot` emerges in `4/6` runs
  - `3/3` curriculum on
  - `1/3` curriculum off

That means the model is often using slot matching causally, but usually not as one neat high-scoring linearly probeable variable.

### 5. Behavior can become strong before routing looks clean by the strict operator metric

Representative examples:

- `curriculum_on seed0`
  - behavior birth: `28`
  - copy birth: `31`
  - routing birth: `114`
- `curriculum_off seed0`
  - behavior birth: `12`
  - copy birth: `93`
  - routing birth: `122`

The correct interpretation is not that routing is absent until late. It is that the model is already using a working retrieval process while the explicit routing metric keeps sharpening long after behavior is already good.

### 6. Curriculum mainly changes the cleanup stage, not the existence of the motif

Both `curriculum_on` and `curriculum_off` solve the task nearly perfectly.

What changes is not whether the retrieval motif exists, but how cleanly the last ambiguous part settles:

- query-key and selected-value signals become extremely strong in both conditions
- the downstream retrieval role becomes strong in both conditions
- matching-slot faithfulness is much more consistent with curriculum on

### 7. Two representative formation paths

#### Slower staged formation: `curriculum_on seed0`

- epochs `1` to `27`: held-out performance stays low while partial structure accumulates
- epoch `28`: behavior crosses the birth threshold
- epoch `31`: mixed `2`-pair / `3`-pair training begins, and query-key faithfulness, selected-value faithfulness, and copy birth appear
- epoch `99`: matching-slot faithfulness crosses threshold
- epoch `114`: selected-value variable and routing birth appear

#### Faster direct formation: `curriculum_off seed1`

- epoch `1`: copy birth already appears
- epoch `16`: query-key variable birth
- epoch `17`: behavior birth, query-key faithfulness birth, selected-value variable birth, and routing birth
- epoch `19`: selected-value faithfulness appears
- by epoch `19`, val and test are already `1.0` and OOD is `0.998`

The strongest empirical answer this project can currently give is:

**the circuit forms in stages, through partial retrieval pieces becoming coordinated, with a very stable downstream retrieval role and a much less stable upstream implementation.**

## What All Findings Add Up To

Taken together, the three benchmarks support a more specific conclusion than "small transformers have circuits."

The symbolic benchmark supports the claim that a tiny transformer can implement an in-context retrieval mechanism and that the mechanism can be decomposed into interpretable parts rather than behaving like an opaque lookup table.

The story benchmark supports the claim that the checkpoint-tracking harness does not automatically produce a success narrative. When the task mostly invites memorization, the harness reports weak local structure, no convincing births, and no stable faithful mechanism.

The textual benchmark supports the main result:

- a retrieval motif can be tracked across training
- that motif survives across a full `150`-epoch, `6`-run matrix
- the stable part of the motif is stronger at the role level than at the exact head-identity level
- the motif forms in stages rather than in one universal emergence event

The most evidence-backed conclusion is therefore:

**small transformers repeatedly learn a staged retrieval motif, not one fixed universal head graph.**

That staged motif looks like this:

- a strong downstream layer-2 retrieval or write role
- an additional upstream support or setup component
- very strong query-key and selected-value structure by the end
- a slot-matching computation that is real but often more distributed and later-settling than the other parts

## Limits

The project also makes several deliberate non-claims.

It does not yet show:

- that one exact head graph is universal
- that every decodable variable is a faithful causal variable
- why gradient descent prefers this retrieval motif over other possible solutions
- a closed-form mathematical derivation of circuit formation

Those are the current boundary of the evidence, not hidden exceptions.

## GPT-2 Activation Viewer

The repository also includes a lightweight GPT-2 activation viewer for prompt-level inspection.

<figure class="paper-figure">
  <img src="figures/gpt2_activation_viewer.png" alt="GPT-2 activation viewer">
  <figcaption><strong>Figure 11.</strong> Lightweight prompt-level activation viewer for GPT-2 inspection. This is separate from the small-transformer training-dynamics pipeline, but kept in the repository as a compact inspection tool.</figcaption>
</figure>

Build a viewer payload:

```bash
.venv/bin/python scripts/build_interactive_model_viewer.py \
  --prompt "The secret code is 73914. Repeat the secret code exactly:" \
  --prompt-id demo_prompt_001 \
  --task copy \
  --model gpt2-small \
  --out outputs/viewer_payload.json
```

Run the viewer:

```bash
cd webapp
.venv/bin/python -m http.server 8000
```

Then open `http://localhost:8000` and upload `outputs/viewer_payload.json`.

## Reproducibility

- [Reproducibility notes](reproducibility.md)

## Repository Sources For The Reported Experiments

The experiments reported on this page are repository artifacts, not external papers. The exact toy-model datasets, manifests, notebooks, scripts, and curated outputs live in this GitHub repository:

- [Repository root]({{ site.repository_url }})
- [Full repository report]({{ site.repository_url }}/blob/main/results.md)
- [Reproducibility commands]({{ site.repository_url }}/blob/main/docs/reproducibility.md)
- [Phase 2 datasets]({{ site.repository_url }}/tree/main/dataset/phase2)
- [Phase 2 manifests]({{ site.repository_url }}/tree/main/manifests/phase2)
- [Public evidence bundle]({{ site.repository_url }}/tree/main/artifacts/phase2)
- [Static-analysis notebooks]({{ site.repository_url }}/tree/main/notebook)

## References

The prior external work most relevant to the framing and methods used here is:

1. Anthropic, “Toy Models of Superposition,” September 14, 2022. [https://www.anthropic.com/research/toy-models-of-superposition](https://www.anthropic.com/research/toy-models-of-superposition)
2. Anthropic, “Towards Monosemanticity: Decomposing Language Models With Dictionary Learning,” October 5, 2023. [https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)
3. Anthropic, “Superposition, Memorization, and Double Descent,” January 5, 2023. [https://www.anthropic.com/news/superposition-memorization-and-double-descent](https://www.anthropic.com/news/superposition-memorization-and-double-descent)
4. Anthropic, “Tracing the thoughts of a large language model,” March 27, 2025. This release introduces the paper “Circuit tracing: Revealing computational graphs in language models.” [https://www.anthropic.com/research/tracing-thoughts-language-model](https://www.anthropic.com/research/tracing-thoughts-language-model)

This page is the canonical public report for the repository. Prior interpretability papers are cited above; the toy-model experiments described on this page are documented in the linked GitHub repository artifacts.
