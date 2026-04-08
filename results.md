# Mechanistic Transparency: Results

This report is the repository-facing long-form results document.

The public-facing paper-style version lives at [docs/index.md](docs/index.md). This file stays fuller and more repo-oriented so the complete technical story is still available in one place inside the repository.

The project asks a simple question in a setting small enough to inspect directly:

**can I recover a human-readable retrieval algorithm from a transformer, and can I watch that mechanism take shape during training?**

That question is part of a broader research agenda:

- where useful computation is localized inside a network
- why gradient descent repeatedly selects some circuit families rather than others
- how internal mechanisms shape outputs
- when neurons and features stay clean versus becoming polysemantic or superposed

The experiments answer that question in three steps:

1. start with the smallest symbolic retrieval task where a circuit can be reverse-engineered after training
2. run the checkpoint-tracking harness on a benchmark that should fail cleanly if the model only memorizes
3. move to a larger retrieval benchmark where the same circuit family can be tracked across checkpoints, seeds, and curricula

That order matters. Each benchmark answers a different part of the overall question.

The strongest conclusion is not that one exact head graph appears in every run. The strongest conclusion is that the experiments support a **retrieval motif** that appears repeatedly while taking variable formation paths:

- an upstream support or setup computation
- a downstream retrieval or write computation
- strong query-key and selected-value representations
- a matching step that is real but often more distributed than a single neat probe variable

That framing is closest to the induction-head and progress-measure view of emergence [(Olsson et al., 2022)](https://arxiv.org/abs/2209.11895), [(Nanda et al., 2023)](https://arxiv.org/abs/2301.05217).

## Related Work

This repository is in direct conversation with four nearby lines of interpretability work.

First, the closest predecessor is the induction-head literature. [Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) showed that in-context learning can be linked to a concrete transformer mechanism and that the mechanism emerges during training. The retrieval experiments here operate in a smaller and more controlled regime, but they ask the same family of questions: what mechanism appears, how stable is it, and how does it emerge?

Second, the project is closely related to end-to-end circuit explanations in larger or more natural tasks. [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) on indirect object identification is the clearest nearby example. That work established that a natural language behavior can be reverse-engineered into a causal circuit and evaluated with faithfulness-style criteria. The present repository uses much smaller models and tasks, but relies on the same distinction between readable structure and causally meaningful structure.

Third, the staged-emergence result here belongs in the same conversation as grokking and progress-measure work. [Power et al. (2022)](https://arxiv.org/abs/2201.02177) made small algorithmic datasets into a clean setting for delayed generalization, and [Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) argued that apparently sharp transitions can hide smoother internal progress measures. The six-run retrieval matrix shows a closely related pattern: behavior, copy, routing, and matching-related scores do not cross threshold in one fixed order, and threshold births often mark cleanup rather than literal first appearance.

Fourth, the feature-level interpretation in this repo is informed by Anthropic’s superposition and sparse-feature work. [Toy Models of Superposition](https://www.anthropic.com/research/toy-models-of-superposition), [Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning), and [Superposition, Memorization, and Double Descent](https://www.anthropic.com/news/superposition-memorization-and-double-descent) are directly relevant to the tension between compact reusable features, interference, memorization, and distributed internal structure. The matching-slot result in this repository fits that picture unusually well: slot matching looks causally real, but not neatly isolated.

## Shared Setup

Across the training-dynamics experiments, the repository uses a small decoder-only transformer with:

- causal self-attention
- RoPE position encoding
- RMSNorm
- SwiGLU MLP blocks
- tied embeddings and output logits

Core implementation:

- [scripts/tiny_transformer_core.py](scripts/tiny_transformer_core.py)

The standard execution loop is:

1. train a model while logging optimization telemetry
2. save checkpoint `0`, dense early checkpoints, and the final checkpoint
3. run a mechanistic battery on each saved checkpoint
4. aggregate checkpoint artifacts into run-level and grouped summaries
5. render figures from those summaries

Core scripts:

- [scripts/train_run.py](scripts/train_run.py)
- [scripts/run_checkpoint_battery.py](scripts/run_checkpoint_battery.py)
- [scripts/summarize_training_dynamics.py](scripts/summarize_training_dynamics.py)
- [scripts/visualize_kv_circuit_dynamics.py](scripts/visualize_kv_circuit_dynamics.py)

Command sequences are collected in:

- [docs/reproducibility.md](docs/reproducibility.md)

The tracked evidence pack in [artifacts/phase2](artifacts/phase2) is a curated copy of the main outputs used here. The raw `runs/` trees still contain the full checkpoint outputs; `artifacts/phase2` contains the subset needed to support the main claims.

### What Is Recorded

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
- operator scoring
- head ablation / localization
- neuron summaries
- SAE feature summaries
- superposition metrics
- representation drift
- operator handoffs

These recordings answer different questions:

- behavior asks whether the model really solves the task on held-out and OOD data
- probes ask whether a representation contains decodable task information
- faithfulness asks whether that information is causally useful rather than merely readable
- operator scores ask whether a head looks routing-like or copy-like
- localization asks whether ablating a component actually hurts the behavior
- neurons and SAE features ask whether the computation sharpens into smaller reusable parts
- drift and handoffs ask whether the mechanism stabilizes or keeps reorganizing

So this is an empirical training-dynamics study of **what emerges**, **when it emerges**, and **how stable that emergence is**.

## 1. Start With The Smallest Clean Retrieval Circuit

The first benchmark is the symbolic key-value task in [dataset/phase2/kv_retrieve_3](dataset/phase2/kv_retrieve_3), summarized in [artifacts/phase2/static_kv/metadata.json](artifacts/phase2/static_kv/metadata.json).

Dataset details:

- generator seed `7`
- `8` keys and `8` values
- splits: `5000` train / `500` val / `500` test / `500` OOD 4-pair
- the queried key is always sampled from the current context
- pair order is shuffled inside each prompt
- OOD increases the context from `3` pairs to `4` pairs

Prompt format:

```text
<bos> K3 V4 ; K1 V7 ; K5 V0 ; Q K5 ->
```

Target:

```text
V0
```

This benchmark is useful because each prompt defines a temporary mapping. There is no fixed global association like `K5 -> V0` stored in weights. The model has to use the current prompt, so the task behaves more like reading from a tiny in-context database than recalling a permanent fact.

The symbolic analysis uses the released [models/kv_retrieve_3/selected_checkpoint.pt](models/kv_retrieve_3/selected_checkpoint.pt), which stores `selected_epoch = 126` and benchmark metadata indicating that all symbolic evaluation checks pass. The checkpoint has:

- `2` layers
- `2` heads per layer
- `d_model = 48`
- `d_ff = 96`
- `max_seq_len = 16`

Two layers matter because they allow a natural decomposition:

- layer 1 can prepare a retrieval state at the final position
- layer 2 can use that state to select the correct source and write the answer

Canonical analysis lives in:

- [notebook/kv_retrieve_algorithm_analysis.ipynb](notebook/kv_retrieve_algorithm_analysis.ipynb)
- [notebook/kv_retrieve_algorithm_discovery.ipynb](notebook/kv_retrieve_algorithm_discovery.ipynb)

The static analysis stack includes:

- activation patching
- path patching
- causal ablations
- QK / OV analysis
- circuit tracing
- sparse feature analysis
- neuron-level analysis

Released checkpoint performance:

| Split | Accuracy | Mean Margin |
| --- | ---: | ---: |
| Train | `0.9978` | `12.7831` |
| Val | `0.9520` | `11.7387` |
| Test | `0.9560` | `11.7673` |
| OOD 4-pair | `0.7280` | `4.8225` |

This benchmark provides the first strong evidence in the repo for the following circuit story:

- a layer-1 support or control computation writes a useful retrieval state at the final position
- a layer-2 head uses that state to retarget attention toward the correct source
- the same layer-2 head writes the retrieved value into the final residual stream

Short form:

`layer-1 support -> layer-2 query retargeting -> layer-2 value write`

This is the circuit family that the later training-dynamics experiments track in richer settings.

It is also where the project first encounters a limit. Even in this smallest setting, it is much easier to identify a circuit than to compress that circuit into the simplest possible human-readable algorithm. Neurons and features are visible, but their semantics are not perfectly clean.

## 2. Test The Harness On A Benchmark That Should Fail Cleanly

The second benchmark is a story next-token task:

- [dataset/phase2/random_story_dataset_v1.txt](dataset/phase2/random_story_dataset_v1.txt)
- [artifacts/phase2/story_negative/config.json](artifacts/phase2/story_negative/config.json)

Dataset details:

- one fixed story token stream with `1168` tokens
- regex word-or-punctuation tokenization
- context length `24`
- longer-context OOD length `40`
- stride `1`
- derived split sizes: `910` train / `117` val / `117` test / `117` OOD

Model and training:

- `2` layers
- `2` heads
- `d_model = 32`
- `d_ff = 64`
- `150` epochs
- batch size `32`
- eval batch size `128`
- learning rate `0.01`
- weight decay `0.0`
- seed `0`
- device `cpu`
- checkpoint schedule: dense through epoch `20`, then `24` log-spaced checkpoints, plus epoch `0` and final
- total saved checkpoints summarized: `47`

Canonical evidence:

- [artifacts/phase2/story_negative/summaries/checkpoint_index.csv](artifacts/phase2/story_negative/summaries/checkpoint_index.csv)
- [artifacts/phase2/story_negative/summaries/run_summary.json](artifacts/phase2/story_negative/summaries/run_summary.json)
- [artifacts/phase2/story_negative/summaries/emergence.csv](artifacts/phase2/story_negative/summaries/emergence.csv)

Key metrics:

- best held-out checkpoint at epoch `1`
- best val accuracy: `0.0940`
- final train accuracy: `1.0`
- final val accuracy: `0.0427`
- final test accuracy: `0.0256`
- final OOD accuracy: `0.0256`

### Story Behavior and Variables

![Story behavior and variables](docs/figures/story_behavior_and_variables.png)

### Story Operator and Localization

![Story operator and localization](docs/figures/story_operator_and_localization.png)

This run fails, but it fails in an informative way.

What the model does:

- it memorizes the training windows
- it preserves some local internal structure, especially around previous-token information
- it never turns that local structure into a stable faithful generalizing circuit

Why it fails:

- the task is a single contiguous story stream, so memorizing local windows is cheap
- the training split is only `910` examples after windowing, which is too small for a strong generalization claim
- train accuracy rises to `1.0`, but val/test/OOD collapse instead of improving together
- [artifacts/phase2/story_negative/summaries/emergence.csv](artifacts/phase2/story_negative/summaries/emergence.csv) contains no birth epochs for behavior, routing, copy, or faithfulness
- [artifacts/phase2/story_negative/summaries/run_summary.json](artifacts/phase2/story_negative/summaries/run_summary.json) reports `0` behavior births, `0` routing births, and `0` copy births

This benchmark matters because it shows that the instrumentation does not automatically turn weak internal structure into a clean circuit story. The harness can see local signals without overclaiming that a stable mechanism exists.

## 3. Move To A Larger Retrieval Benchmark Where Emergence Can Be Tracked

The main benchmark is [dataset/phase2/kv_retrieve_textual_balanced_v1](dataset/phase2/kv_retrieve_textual_balanced_v1), summarized in [artifacts/phase2/textual_kv/metadata.json](artifacts/phase2/textual_kv/metadata.json).

This benchmark keeps the retrieval task clean while making memorization substantially harder through:

- many more prompts
- deduplication across splits
- balanced queried-slot coverage
- a textual surface form instead of raw `K*` / `V*` tokens

Dataset details:

- generator seed `17`
- `16` keys and `16` values
- splits: `30000` train 2-pair / `30000` train 3-pair / `3000` val / `3000` test / `3000` OOD 4-pair
- queried slot is balanced across the dataset
- pair order is shuffled
- prompts are deduplicated across splits
- keys and values are distinct within each example
- OOD uses `4` context pairs instead of `3`

Prompt format:

```text
<bos> Mara amber ; Ivo cedar ; Sera linen ; Q Ivo ->
```

Target:

```text
cedar
```

The main model family uses:

- `2` layers
- `2` heads
- `d_model = 64`
- `d_ff = 128`

Shared training details for the pilot and the full matrix:

- batch size `64`
- learning rate `0.001`
- weight decay `0.0`
- device `cpu`
- checkpoint `0` and final checkpoint saved in every run
- SAE tracking enabled on `12` sites
- birth thresholds:
  - behavior val accuracy `0.9`
  - operator score `0.85`
  - variable score `0.85`
  - faithfulness score `0.85`
  - operator / variable / faithfulness family-min gate `0.75`

Cross-seed role matching uses only final checkpoints. For each role, heads are ranked by operator score with family-min score as a tiebreaker; the top head is the role candidate; and cross-seed stability is summarized with candidate identity, top-`3` overlap, and cosine similarity of the full per-head score profile.

At this point the report stops asking whether a retrieval circuit exists at all. The symbolic benchmark already showed that. The question becomes:

**does a larger and harder retrieval benchmark let us watch that circuit family emerge during training?**

### The Pilot Run: First Clear Positive Training-Dynamics Result

The pilot run is defined in [artifacts/phase2/textual_kv/pilot/manifest.json](artifacts/phase2/textual_kv/pilot/manifest.json):

- `30` epochs
- curriculum `on`
- checkpoint saved every epoch
- best metric: validation accuracy

Final pilot metrics at epoch `30`:

- val accuracy: `0.9393`
- test accuracy: `0.9347`
- OOD 4-pair accuracy: `0.8457`

Pilot localization at the final checkpoint:

- `block1_head0` ablation drops accuracy by `0.6668`
- `block2_head1` ablation drops accuracy by `0.5348`

That already suggests the same high-level decomposition seen in the symbolic benchmark:

- a strong layer-1 support or setup role
- a strong layer-2 retrieval or write role

### Pilot Circuit Emergence

![Textual KV circuit emergence](docs/figures/textual_kv_circuit_emergence.png)

### Pilot Network Change

![Textual KV network change](docs/figures/textual_kv_network_change.png)

### Pilot Operator and Localization

![Textual KV operator and localization](docs/figures/textual_kv_operator_and_localization.png)

### Pilot Representation Drift

![Textual KV representation drift](docs/figures/textual_kv_representation_drift.png)

The pilot is important because it shows a narrow transition window instead of a vague gradual improvement.

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

This is one of the clearest pieces of evidence in the repository. The run does not improve smoothly. The parts needed for retrieval become coordinated over a short window, and held-out behavior rises rapidly once that coordination locks in.

The pilot therefore gives the first strong positive evidence for the central dynamics question:

- the retrieval mechanism is not only present at the end
- its pieces become visible before behavior peaks
- those pieces then lock together over a short interval

### The Full Matrix: Six Long Runs

The main experiment is a `2 x 3` matrix:

- curriculum on, seeds `0`, `1`, `2`
- curriculum off, seeds `0`, `1`, `2`

`curriculum_on` means:

- train on easy `2`-pair prompts first
- then a mix of `2`-pair and `3`-pair prompts
- then `3`-pair prompts only

In the actual manifests this means:

- epochs `1` to `30`: `2`-pair only
- epochs `31` to `60`: mixed `2`-pair and `3`-pair
- epochs `61` to `150`: `3`-pair only

`curriculum_off` means:

- train on `3`-pair prompts from the start

In the actual manifests this means:

- epochs `1` to `150`: `3`-pair only

Matrix details:

- `6` full runs total
- same architecture as the pilot
- `150` epochs per run
- batch size `64`
- learning rate `0.001`
- dense checkpoints through epoch `30`, then `24` log-spaced checkpoints, plus epoch `0` and final
- summarized checkpoints:
  - curriculum on: `216`
  - curriculum off: `190`

Run set:

- [manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_on_d64_l2.json](manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_on_d64_l2.json)
- [manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_on_d64_l2.json](manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_on_d64_l2.json)
- [manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_on_d64_l2.json](manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_on_d64_l2.json)
- [manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_off_d64_l2.json](manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_off_d64_l2.json)
- [manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_off_d64_l2.json](manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_off_d64_l2.json)
- [manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_off_d64_l2.json](manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_off_d64_l2.json)

Grouped evidence:

- [artifacts/phase2/textual_kv/full_matrix/curriculum_on](artifacts/phase2/textual_kv/full_matrix/curriculum_on)
- [artifacts/phase2/textual_kv/full_matrix/curriculum_off](artifacts/phase2/textual_kv/full_matrix/curriculum_off)

#### Grouped Outcome Table

| Condition | Val Acc | Test Acc | OOD Acc | Query Key | Matching Slot | Selected Value | Routing | Copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curriculum on | `1.0000` | `1.0000` | `0.9921` | `0.9997` | `0.4583` | `0.9997` | `0.9729` | `0.9293` |
| Curriculum off | `0.9999` | `1.0000` | `0.9993` | `1.0000` | `0.5339` | `0.9999` | `0.9999` | `0.9521` |

The first finding is simple: both conditions solve the task almost perfectly.

That already rules out one easy story, namely that curriculum is required for success. It is not. The task is learnable either way.

### Grouped Matrix Overview

![Textual KV full matrix overview](docs/figures/textual_kv_full_matrix_overview.png)

The grouped comparison makes two points visible immediately:

- held-out behavior is near-perfect in both conditions
- the main internal difference is not query or selected-value quality, but the weaker and more variable matching step

#### Per-Seed Final Checkpoints

The grouped means are clean, but the per-seed final rows are what show where the motif is stable and where it moves:

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

- all six full runs end with a strong layer-2 retrieval role
- all six full runs end with at least one additional strongly causal component besides the dominant retrieval role
- query and selected-value representations are very strong by the end

What is not stable:

- the exact head identity of the strongest non-retrieval component
- the exact layer-2 head identity in every seed
- the exact timing at which routing, copy, and faithfulness cross threshold

The important nuance is that most runs do look like a layer-1 support plus layer-2 retrieval decomposition, but not every seed localizes the strongest non-retrieval effect to the same layer-1 head. In fact, `5/6` runs end with the single largest ablation landing on layer 1, even though the most stable routing role lives in layer 2. So the data supports a stable role-level motif more strongly than one exact head-level template.

### Final Localization Across All Six Runs

![Textual KV full matrix localization](docs/figures/textual_kv_full_matrix_localization.png)

This heatmap is the clearest compact view of the role-level result:

- every run ends with one or two strongly causal heads
- the strongest downstream retrieval effect stays in layer 2
- the strongest upstream support effect moves across seeds

#### Emergence Is Staged, Not Universal

There is no single epoch where "the circuit appears."

Examples from the grouped emergence tables:

| Condition | Seed | Behavior Birth | Copy Birth | Routing Birth | Matching-Slot Faithfulness Birth |
| --- | ---: | ---: | ---: | ---: | ---: |
| on | `0` | `28` | `31` | `114` | `99` |
| on | `1` | `15` | `36` | `31` | `122` |
| on | `2` | `37` | `62` | `—` | `40` |
| off | `0` | `12` | `93` | `122` | `57` |
| off | `1` | `17` | `1` | `17` | `—` |
| off | `2` | `20` | `17` | `131` | `—` |

Several things are visible immediately:

- behavior birth varies substantially across seeds
- copy can appear before routing
- faithfulness can appear after behavior
- some runs solve the task cleanly even when one strict internal birth never occurs

### Emergence Timing Across Seeds

![Textual KV full matrix emergence](docs/figures/textual_kv_full_matrix_emergence.png)

The emergence heatmap makes the staged character of formation hard to miss:

- behavior birth is early in some runs and late in others
- copy can become clean very early while routing stays below threshold for much longer
- matching-slot faithfulness is the least consistent internal milestone

#### Threshold Robustness

The timing story is not a fragile artifact of one exact cutoff. Recomputing births with nearby primary thresholds leaves the qualitative result unchanged:

- behavior threshold `0.85` to `0.95`: birth epochs stay in the range `12` to `38`
- routing threshold `0.80` to `0.90` with family-min gate fixed at `0.75`: the same five runs still birth at `17`, `31`, `114`, `122`, and `131`, with one persistent non-birth
- copy threshold `0.80` to `0.90` with family-min gate fixed at `0.75`: births stay at `1`, `17`, `31`, `36`, `62`, and `93`
- matching-slot faithfulness threshold `0.80` to `0.90` with family-min gate fixed at `0.75`: births stay late and sparse at `40`, `57`, `99`, and `122`, with two persistent non-births

So the exact epoch numbers move slightly where expected, but the central conclusion does not: emergence remains widely spread across runs, and the ordering of behavior, copy, routing, and matching-related milestones still fails to collapse into one universal trajectory.

A particularly important finding is that `matching_slot` behaves differently from `query_key` and `selected_value`.

What becomes strong and stable:

- `query_key`
- `selected_value`

What stays weaker and more distributed:

- `matching_slot`

In fact, `matching_slot` never becomes a clean high-scoring linear variable in the same way across the matrix, even though matching-slot faithfulness sometimes does become strong. That suggests the model is using slot matching, but often not as a single neat probe-friendly variable.

#### Matching-Slot Is The Hardest Part To Localize Across Batteries

This is not just a weak-probe artifact. Several batteries agree on the same asymmetry.

Across the six final checkpoints:

- `variable_matching_slot` never reaches birth threshold in any run
- `faithfulness_matching_slot` does emerge in `4/6` runs
- final sparse-feature summaries are dominated by `query_key` and `selected_value`, with `matching_slot` appearing as the top feature at only `5/36` tracked sites with curriculum on and `4/36` with curriculum off
- final neuron summaries are dominated by `query_key` and `selected_value`; `matching_slot` is the top tracked neuron variable in `0` cases under either curriculum

So the strongest supported interpretation is that slot matching is causally real but unusually distributed. It is the least neatly packaged part of the motif in probes, sparse features, and neuron-level summaries.

This also fits naturally with superposition-style accounts of overlapping internal structure [(Anthropic, 2022)](https://www.anthropic.com/research/toy-models-of-superposition), [(Anthropic, 2023)](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning).

#### What Curriculum Changes

The most important curriculum effect is not final accuracy.

It is mechanism cleanliness.

- [artifacts/phase2/textual_kv/full_matrix/curriculum_on/run_summary.json](artifacts/phase2/textual_kv/full_matrix/curriculum_on/run_summary.json) reports `3/3` behavior births, `3/3` query-faithfulness births, `3/3` matching-slot-faithfulness births, and `3/3` copy births
- [artifacts/phase2/textual_kv/full_matrix/curriculum_off/run_summary.json](artifacts/phase2/textual_kv/full_matrix/curriculum_off/run_summary.json) also reports `3/3` behavior births and `3/3` copy births, but only `1/3` matching-slot-faithfulness births

So curriculum changes how cleanly the internal matching computation organizes, not whether the task is solvable at all.

That is the main full-matrix result:

- both conditions solve the task
- both conditions find the same broad retrieval motif
- curriculum mainly improves the consistency and interpretability of the internal matching step

#### Additional Signals Hidden In The Summaries

There are several smaller but still important results in the grouped artifacts that are easy to miss if one only looks at final accuracy and birth epochs.

First, the routing role is even more stable at the profile level than at the raw head-identity level.

- in [artifacts/phase2/textual_kv/full_matrix/curriculum_on/role_matching.csv](artifacts/phase2/textual_kv/full_matrix/curriculum_on/role_matching.csv), routing score-profile cosine stays above `0.97` in every pairwise comparison, even when one seed uses `block2_head1` and another uses `block2_head0`
- in [artifacts/phase2/textual_kv/full_matrix/curriculum_off/role_matching.csv](artifacts/phase2/textual_kv/full_matrix/curriculum_off/role_matching.csv), routing head identity is exactly the same in all `3/3` seeds and routing score-profile cosine is about `0.9997` to `0.9999`

So the routing role is more stable than the raw head names alone suggest.

Second, curriculum changes operator turnover in an interesting way.

From [artifacts/phase2/textual_kv/full_matrix/curriculum_on/operator_handoffs.csv](artifacts/phase2/textual_kv/full_matrix/curriculum_on/operator_handoffs.csv) and [artifacts/phase2/textual_kv/full_matrix/curriculum_off/operator_handoffs.csv](artifacts/phase2/textual_kv/full_matrix/curriculum_off/operator_handoffs.csv):

- curriculum on mean candidate changes:
  - copy: `11`
  - routing: `12`
- curriculum off mean candidate changes:
  - copy: `3`
  - routing: `9`

So curriculum-on runs are not simply "more stable" in every sense. They have more turnover while the mechanism is being assembled, especially on the copy role, but they end with cleaner matching-slot faithfulness.

Third, the hardest behavior family is usually the longer-context OOD family, not the in-distribution perturbation families.

Representative final worst families:

- `on0`: `longer_context_ood`, accuracy `1.0`, lowest margin
- `on1`: `longer_context_ood`, accuracy `0.9922`
- `on2`: `longer_context_ood`, accuracy `0.9844`
- `off0`: `longer_context_ood`, accuracy `1.0`, lowest margin
- `off1`: slight weakness on `value_permutation`, accuracy `0.9987`
- `off2`: all families essentially solved, with `same_answer_different_slot` having the lowest margin

So the main remaining difficulty in the benchmark is still context-length generalization.

Fourth, strict birth epochs often mark threshold crossing, not literal first appearance.

Two examples:

- in [artifacts/phase2/textual_kv/full_matrix/curriculum_off/operator_handoffs.csv](artifacts/phase2/textual_kv/full_matrix/curriculum_off/operator_handoffs.csv), `curriculum_off seed0` keeps `block2_head1` as the routing candidate from epoch `11` onward and only flips to final routing head `block2_head0` at epoch `122`
- in [artifacts/phase2/textual_kv/full_matrix/curriculum_on/operator_handoffs.csv](artifacts/phase2/textual_kv/full_matrix/curriculum_on/operator_handoffs.csv), `curriculum_on seed0` shows a cleaner operator handoff: copy is led by `block1_head0` through epoch `25`, then `block2_head1` takes over at epoch `26` and remains dominant

So a late routing birth does not mean routing is absent until late. It often means the operator metric does not cross a strict threshold until a long-running cleanup or purification stage.

Fifth, the feature, neuron, and superposition batteries all point at the same internal asymmetry.

- at final checkpoints, top sparse features are dominated by `query_key` and `selected_value`
- at final checkpoints, top neuron labels are also dominated by `query_key` and `selected_value`
- in [runs/kv_textual_balanced_v1/curriculum_on/summaries/superposition_dynamics.csv](runs/kv_textual_balanced_v1/curriculum_on/summaries/superposition_dynamics.csv) and [runs/kv_textual_balanced_v1/curriculum_off/summaries/superposition_dynamics.csv](runs/kv_textual_balanced_v1/curriculum_off/summaries/superposition_dynamics.csv), `block2_final_mlp_out` has the weakest top-feature selectivity among the main final sites

So the late writeout region stays more entangled than the cleaner upstream representations, and the internal batteries agree that matching/write organization is where the mechanism remains messiest.

#### Compact Summary Of The Extra Signals

| Signal | Curriculum On | Curriculum Off | Why It Matters |
| --- | --- | --- | --- |
| Final routing role identity | `block2_head1` in `2/3` seeds, `block2_head0` in `1/3` | `block2_head0` in `3/3` seeds | downstream routing is the most stable role in the project |
| Routing score-profile cosine across seeds | `> 0.97` in every pairwise comparison | `~0.9997` to `~0.9999` | role stability is stronger than raw head-name stability |
| Mean operator handoffs, copy | `11` | `3` | curriculum-on runs reorganize more while assembling the mechanism |
| Mean operator handoffs, routing | `12` | `9` | routing keeps sharpening and sometimes reassigning late |
| Matching-slot faithfulness births | `3/3` runs | `1/3` runs | curriculum mainly affects the cleanliness of the matching step |
| Hardest final behavior family | usually `longer_context_ood` | usually `longer_context_ood`, except one mild `value_permutation` weakness | the main remaining difficulty is context-length generalization |
| Strongest final ablation head in layer 1 | `2/3` runs | `3/3` runs | the stable downstream role still depends on an upstream bottleneck in most seeds |
| Top final feature / neuron labels | dominated by `query_key` and `selected_value`; `matching_slot` is rare | dominated by `query_key` and `selected_value`; `matching_slot` is rare | the feature and neuron batteries agree that matching is the least neatly packaged part |
| Most weakly stabilized late representation | `block2_final_mlp_out` | `block2_final_mlp_out` | the late writeout region keeps reorganizing longer than the cleaner upstream representations |

### Stability, Turnover, And Birth Summary

![Textual KV full matrix stability](docs/figures/textual_kv_full_matrix_stability.png)

This summary figure compresses the grouped evidence behind the main interpretation:

- routing is more stable at the profile level than raw head names alone suggest
- curriculum-on runs reorganize more while assembling the mechanism
- curriculum mainly improves the consistency of matching-slot faithfulness, not the ability to solve the task

## Empirical Answer: How The Circuit Forms

This repository does not yet answer the mathematical question "why gradient descent builds this motif." But it **does** answer the empirical formation question:

**what actually forms first, what forms later, and which parts of the circuit are stable enough to trust?**

The data supports the following formation story.

### 1. The circuit does not appear all at once

The model does not jump from "no mechanism" to "finished mechanism" in a single clean step.

What the full matrix shows instead is:

- behavior can rise before every internal metric becomes clean
- copy-like structure often becomes clean before routing-like structure
- query and selected-value information can be strong even while matching-slot is still messy

This is visible directly in the emergence tables. In the grouped summaries:

- [artifacts/phase2/textual_kv/full_matrix/curriculum_on/run_summary.json](artifacts/phase2/textual_kv/full_matrix/curriculum_on/run_summary.json) reports `routing_before_copy_runs = 1`
- [artifacts/phase2/textual_kv/full_matrix/curriculum_off/run_summary.json](artifacts/phase2/textual_kv/full_matrix/curriculum_off/run_summary.json) reports `routing_before_copy_runs = 0`

So in most runs, copy is as early or earlier than routing under the strict thresholds.

### 2. The most stable part of formation is the downstream layer-2 retrieval role

Across all six final `150`-epoch runs:

- the final routing candidate is a layer-2 head in all six runs
- the final copy candidate is a layer-2 head in all six runs

So the strongest stable object in formation is not the entire circuit graph. It is the repeated emergence of a downstream layer-2 retrieval or write role.

That is the most reliable answer to "what forms" in the matrix.

### 3. The upstream part is real, but less head-stable

The matrix also shows a second important causal component besides the dominant layer-2 head:

- `on0`: strongest ablation is `block1_head0`
- `on1`: strongest ablation is `block1_head0`
- `on2`: strongest ablation is `block2_head0`
- `off0`: strongest ablation is `block1_head0`
- `off1`: strongest ablation is `block1_head0`
- `off2`: strongest ablation is `block1_head1`

So the repo supports an upstream support or setup part of the motif, but not a single universal head identity for that role.

That is why the right empirical claim is:

- stable role-level motif
- unstable exact head-level implementation

not:

- one fixed circuit template in every seed

### 4. The matching step is the hardest part to cleanly localize

`matching_slot` is the weakest part of the internal story.

What the matrix shows:

- `variable_matching_slot` never reaches birth threshold in any of the six full runs
- `faithfulness_matching_slot` does emerge in `4/6` runs
  - `3/3` curriculum on
  - `1/3` curriculum off

This is one of the most important findings in the repository, and the section above shows that it survives cross-checking against feature and neuron summaries too.

It means:

- the model is often using slot matching causally
- but that matching computation is not usually represented as one neat, high-scoring, linearly probeable variable

So the circuit forms with a matching computation that is frequently distributed or implicit, not cleanly packaged into one probe-friendly state.

### 5. Behavior can become strong before routing looks clean by the strict operator metric

This is another place where the matrix is more subtle than a simple story.

Representative examples:

- `curriculum_on seed0`
  - behavior birth: `28`
  - copy birth: `31`
  - routing birth: `114`
- `curriculum_off seed0`
  - behavior birth: `12`
  - copy birth: `93`
  - routing birth: `122`

So the model can already behave well on the task before the routing operator is clean enough to cross the strict birth threshold.

The correct interpretation is not that routing is absent until late. It is that:

- the model is already using a working retrieval process
- the explicit routing metric continues sharpening long after behavior is already good

That is a very different claim from "routing appears at one exact epoch and then behavior starts."

### 6. Curriculum mainly changes the cleanup stage, not the existence of the motif

Both `curriculum_on` and `curriculum_off` solve the task nearly perfectly.

What changes is not whether the retrieval motif exists, but how cleanly the last ambiguous part settles:

- query-key and selected-value signals become extremely strong in both conditions
- the downstream retrieval role becomes strong in both conditions
- matching-slot faithfulness is much more consistent with curriculum on

So curriculum is best understood as shaping the organization of the mechanism, especially the slot-matching stage, rather than deciding whether the model can solve retrieval at all.

### 7. Two representative formation paths

The matrix is easiest to understand by looking at one slower run and one faster run.

#### Slower staged formation: `curriculum_on seed0`

This run continues the pilot out to `150` epochs.

What happens:

- epochs `1` to `27`: held-out performance stays low while partial structure accumulates
- epoch `28`: behavior crosses the birth threshold
- epoch `31`: mixed `2`-pair / `3`-pair training begins, and query-key faithfulness, selected-value faithfulness, and copy birth all appear
- epoch `99`: matching-slot faithfulness finally crosses threshold
- epoch `114`: selected-value variable and routing birth appear

So this run forms in layers:

- rough working mechanism
- then explicit copy / faithful variable structure
- then much later a cleaner routing operator and cleaner selected-value variable

#### Faster direct formation: `curriculum_off seed1`

This run solves the harder `3`-pair task from the start.

What happens:

- epoch `1`: copy birth already appears
- epoch `16`: query-key variable birth
- epoch `17`: behavior birth, query-key faithfulness birth, selected-value variable birth, and routing birth
- epoch `19`: selected-value faithfulness appears
- by epoch `19`, val and test are already `1.0` and OOD is `0.998`

So this run forms much faster and more directly, but it still lands on the same broad retrieval motif.

### What this section actually answers

The strongest empirical answer the current repository can give is:

- the circuit forms by **coordination of partial retrieval subfunctions**
- the most stable early anchor is the emergence of a **layer-2 retrieval or write role**
- a second upstream support part is usually present but less head-stable
- explicit slot matching is the least clean and latest-settling part
- curriculum mainly improves how cleanly that last part settles

So the data permits an empirical answer to "how does the circuit form?":

**it forms in stages, through partial retrieval pieces becoming coordinated, with a very stable downstream retrieval role and a much less stable upstream implementation.**

## What All Findings Add Up To

Taken together, the three benchmarks support a much more specific conclusion than "small transformers have circuits."

The symbolic benchmark supports the claim that a tiny transformer can implement an in-context retrieval mechanism, and that the mechanism can be decomposed into interpretable parts rather than behaving like an opaque lookup table.

The story benchmark supports the claim that the checkpoint-tracking harness does not automatically produce a success narrative. When the task mostly invites memorization, the harness reports weak local structure, no convincing births, and no stable faithful mechanism.

The textual benchmark then supports the main result of the repository:

- a retrieval motif can be tracked across training
- that motif survives across a full `150`-epoch, `6`-run matrix
- the stable part of the motif is stronger at the role level than at the exact head-identity level
- the motif forms in stages rather than in one universal emergence event

The most evidence-backed conclusion is therefore:

**small transformers repeatedly learn a staged retrieval motif, not one fixed universal head graph.**

That staged motif looks like this:

- a strong downstream layer-2 retrieval or write role
- an additional upstream support or setup component
- very strong query-key and selected-value structure by the end, even though strict sitewise faithfulness is not perfectly uniform in every seed
- a slot-matching computation that is real but often more distributed and later-settling than the other parts

The strongest claims the data supports are:

1. The clean symbolic benchmark is consistent with a real in-context retrieval mechanism.
2. The negative-control story benchmark fails in a way consistent with memorization without a stable generalizing circuit.
3. The larger textual retrieval benchmark makes checkpoint-by-checkpoint circuit emergence visible.
4. The full `curriculum_on/off x seeds 0/1/2` matrix shows that the mechanism is best described as a stable motif rather than a fixed head template.
5. The circuit forms through staged coordination of partial subfunctions, not through one single emergence point.
6. Curriculum changes the cleanliness and consistency of the internal matching computation more than whether the model can solve retrieval at all.

The most important nuance is that the model often behaves correctly before every internal metric becomes clean. That means the report should not be read as saying "nothing exists until a threshold is crossed." Instead, the evidence supports a slower story of assembly, sharpening, handoff, and cleanup.

If the report is read conservatively, the central takeaway is:

**the repository recovers a retrieval mechanism, shows that the mechanism is unlikely to be just an artifact of the tooling, and shows that its emergence is staged, seed-sensitive, and still stable enough to describe at the level of roles rather than fixed head identities.**

## Where The Project Fails, And Why That Matters

The repository contains one strong failure mode and several deliberate non-claims.

The strong failure mode is the story benchmark:

- it fails to generalize
- it never reaches a convincing birth threshold
- it preserves some weak local structure anyway
- it therefore shows that the battery can separate overfitting from stable circuit emergence

The deliberate non-claims are just as important:

- the project does not show that one exact head graph is universal
- the project does not show that every decodable variable is a faithful causal variable
- the project does not yet show why gradient descent prefers this retrieval motif over other possible solutions
- the project does not give a closed-form mathematical derivation of circuit formation

Those are not hidden problems. They are the current boundary of the evidence.

## What The Repo Supports, And What It Leaves Open

This repository supports the following claims:

- what mechanism exists in the clean symbolic benchmark
- what a clean failure looks like when the task invites memorization
- how a retrieval motif appears over training in a larger benchmark
- which parts of that motif are stable across seeds
- which parts vary across seeds and curricula

It does not yet provide a mathematical derivation of why gradient descent builds that motif.

The next narrower question in the ongoing research program is:

**why optimization builds this motif rather than another solution**

That work lives under [research/phase3](research/phase3).
