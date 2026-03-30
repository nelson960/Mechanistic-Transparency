# KV Retrieval Project Note

## What This Project Is

This repo is a small mechanistic-interpretability sandbox.

At the highest level, the research question is:

- where does useful model intelligence come from internally?

In simple terms, that means:

- what internal features are being formed
- how those features interact
- how those interactions become a working algorithm
- how that eventually produces a correct answer

So the real aim is not just "which head fired?" or "which neuron mattered?"
The real aim is to move toward a human-readable explanation of how a model turns a prompt into a piece of reasoning or a correct decision.

The practical path in this repo is:

- first, trace where information flows
- then, identify what the important internal states mean
- then, understand the algorithm those states implement
- later, study how those states are packed together in superposition or polysemantic representations

This is why the repo moved away from only large-model tracing and into a tiny controlled toy model. The toy model is small enough that we can try to inspect the full computation end to end.

The immediate goal here is simple:

- train a tiny transformer on a task where the correct algorithm is known
- inspect how the model solves that task internally
- move from component tracking toward a real explanation of the model's internal algorithm

This is also the current concrete sandbox for the broader Prompt-Level Mechanistic Transparency direction.

## The Bigger Research Aim

The README frames the work as a hierarchy:

1. trace the path of information
2. name the internal features
3. understand the internal algorithm
4. later study more abstract reasoning

This toy project sits mainly between steps 2 and 3.

What has already been built well:

- tracing infrastructure
- causal patching and ablation tools
- head-level circuit analysis

What is still hard:

- saying what a feature really means
- saying what a neuron really means
- showing the internal algorithm in simple human terms

So when this note says "where intelligence comes from", it means:

- not magic
- not just one neuron
- but learned internal circuits, features, and state updates that together implement a useful computation

In this sandbox, the useful computation is retrieval:

- read the query
- find the right slot in context
- output the linked value

If we can explain that computation cleanly in a tiny model, that gives a much better foundation for asking the same question in larger models later.

## What We Are Trying To Explain

The model sees a short prompt with key-value pairs and a query key at the end.
It must output the value linked to that query key.

Example:

```text
<bos> K3 V4 ; K1 V7 ; K5 V0 ; Q K5 ->
```

Correct next token:

```text
V0
```

The intended algorithm is:

1. read the query key
2. find the matching key-value pair in the prompt
3. output the corresponding value

That makes this task useful for interpretability: we know what the model should be doing, so we can test whether the internal circuit actually matches that algorithm.

Another way to say it:

- the dataset gives us a known external task
- the model gives us a learned internal computation
- interpretability is the work of connecting those two in a faithful way

So the target is not only correct prediction.
The target is an explanation like:

- this part of the model reads the query
- this part identifies the matching source
- this part writes the correct value

That is the first small step toward answering where useful internal intelligence comes from.

## Dataset

Dataset name:

- `KV-Retrieve-3`

Files:

- `dataset/kv_retrieve_3/train.jsonl`
- `dataset/kv_retrieve_3/val.jsonl`
- `dataset/kv_retrieve_3/test.jsonl`
- `dataset/kv_retrieve_3/test_ood_4_pairs.jsonl`
- `dataset/kv_retrieve_3/metadata.json`

Simple summary:

- vocabulary size: `20`
- standard prompt length: `13` tokens
- OOD prompt length: `16` tokens
- train / val / test / OOD sizes: `5000 / 500 / 500 / 500`

Why this dataset is useful:

- every prompt defines a new temporary key-value mapping
- there is no global fixed mapping like `K4 -> V0`
- the model has to use the current prompt, not memorized weights

So this task is closer to "read from a tiny in-context database" than to memorization.

## Model

Checkpoint:

- `models/kv_retrieve_3/selected_checkpoint.pt`

Model summary:

- decoder-only transformer
- `2` layers
- `2` heads per layer
- `d_model = 48`
- `head_dim = 24`
- `d_ff = 96`
- `max_seq_len = 16`
- total parameters: `47,280`

Why `2` layers:

- layer 1 can build a contextualized retrieval state
- layer 2 can use that state to select the right source and write the answer

## Canonical Analysis File

Use this notebook as the main artifact:

- `notebook/kv_retrieve_algorithm_analysis.ipynb`

This notebook now combines:

- activation patching
- path patching
- causal ablations
- QK / OV analysis
- circuit tracing
- sparse autoencoder analysis
- feature dashboards
- feature lens / cross-layer feature checks
- neuron-level analysis

Supporting scripts:

- `scripts/generate_kv_retrieve_algorithm_notebook.py`
- `scripts/run_single_prompt_program.py`
- `scripts/run_feature_basis_analysis.py`
- `scripts/kv_retrieve_analysis.py`
- `scripts/kv_retrieve_features.py`

## Where We Are Now

The project is past the setup stage.

Completed:

- dataset built
- model trained
- checkpoint selected and reloadable
- unified analysis notebook built and executed
- circuit-level story is materially advanced

Current frontier:

- feature meaning
- neuron meaning
- extracting the internal algorithm in a way humans can read

So the current problem is no longer "can the model do the task?"
The current problem is "what exact internal computation is the model using?"

## Main Results So Far

### Performance

Selected checkpoint:

- epoch: `126`

Split results:

| Split | Loss | Accuracy | Mean Margin |
| --- | ---: | ---: | ---: |
| Train | `0.005163` | `0.9978` | `12.7831` |
| Val | `0.224087` | `0.9520` | `11.7387` |
| Test | `0.292146` | `0.9560` | `11.7673` |
| Test OOD 4 Pairs | `3.210265` | `0.7280` | `4.8225` |

Simple interpretation:

- the model solves the standard task strongly
- it still generalizes somewhat to longer prompts
- the checkpoint is good enough for mechanism analysis

### Circuit-Level Result

This is the strongest result so far.

Best current circuit story:

- `L1H0` writes an upstream retrieval-control signal at the final position
- that signal matters mainly by changing `L2H0.Q`
- `L2H0` then retargets attention toward the correct source value
- `L2H0` writes the retrieved value strongly into the final residual stream
- `L2H1` still matters, but looks more like a query/key-selection head than the main answer-writing head

Short version:

`L1H0 -> L2H0.Q -> L2H0 value write`

This story is supported by:

- clean/corrupt query swaps
- batch head ablations
- activation patching
- path patching
- QK analysis
- OV analysis

### Feature-Level Result

Feature analysis is useful, but not finished.

Best current feature site:

- `block1_final_l1h0`

Why:

- it is much cleaner than the full block-1 residual site
- it gives a better sparse basis for studying the upstream retrieval signal

Current feature findings:

- support features: `40`, `30`
- control feature: `370`
- grouped activations are sharper by `query_key` than by target value
- this suggests the upstream features look more like query / routing features than direct answer-value features

Important limitation:

- single-feature interventions are weak
- no single SAE feature explains retrieval by itself
- the signal is still distributed across multiple features

### Neuron-Level Result

Neuron analysis has started, mainly in the layer-1 MLP.

Current findings:

- `L1 MLP` matters a lot on the anchor prompt
- top prompt-local neurons included `2`, `60`, and `30`
- some of these neurons clearly respond to query-dependent upstream changes
- `L1H0` is the strongest upstream head cause for those selected neurons on the anchor prompt

Simple interpretation:

- these neurons look more like query / routing / state-construction neurons than direct answer-token neurons

Important limitation:

- we do not yet have stable human-readable semantic labels for these neurons
- the computation still looks distributed, not like one clean "retrieval neuron"

## What We Understand Best

The strongest current understanding is at the head / circuit level.

What seems solid:

- the model uses the query token in context
- `L2H0` is the main downstream value-writing head
- `L1H0` is an important upstream sender
- the most important routed effect identified so far is through `L2H0.Q`

## What Is Still Not Solved

We do **not** yet have a full human-readable internal algorithm.

What is still missing:

- a clean symbolic register view such as `query_key -> selected_slot -> selected_value`
- stable semantic labels for important features
- stable semantic labels for important neurons
- a closed prompt-to-response explanation at feature / neuron / state-variable level

So the honest current state is:

- tracking tools: built
- circuit story: strong
- feature / neuron semantics: partial
- full internal algorithm: not solved yet

## Current Best One-Sentence Summary

This tiny transformer really does in-context retrieval, and the best current mechanistic story is that `L1H0` prepares a retrieval-control signal that changes `L2H0`'s query, which lets `L2H0` attend to the right source and write the correct value into the final residual stream.
