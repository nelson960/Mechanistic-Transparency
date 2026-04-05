# Synthetic Datasets

This folder is organized by internal research stage:

- `phase2/` contains the benchmarks used in the public circuit-emergence results
- `phase3/` contains the reduced datasets used for the formation-causal follow-up work

## `phase2/kv_retrieve_3`

Task:
- The model sees 3 key-value pairs plus a query key.
- It must predict the value paired with the query key.

Example prompt:
```text
<bos> K2 V5 ; K7 V1 ; K0 V4 ; Q K7 ->
```

Target:
```text
V1
```

Why this task exists:
- The algorithm is simple and known in advance.
- A 1-layer decoder can plausibly solve it.
- Attention patterns, source-token tracing, and ablations are easy to interpret.

Files:
- `phase2/kv_retrieve_3/train.jsonl`
- `phase2/kv_retrieve_3/val.jsonl`
- `phase2/kv_retrieve_3/test.jsonl`
- `phase2/kv_retrieve_3/test_ood_4_pairs.jsonl`
- `phase2/kv_retrieve_3/metadata.json`

Each row contains:
- `id`
- `task`
- `split`
- `num_pairs`
- `prompt`
- `target`
- `query_key`
- `context_pairs`

## `phase2/kv_retrieve_textual_balanced_v1`

Purpose:
- keep the clean retrieval task while making memorization much harder
- support the main training-dynamics matrix

Prompt format:
```text
<bos> Mara amber ; Ivo cedar ; Sera linen ; Q Ivo ->
```

Target:
```text
cedar
```

Why this dataset matters:
- the random-story benchmark overfit almost immediately
- the original tiny KV dataset was too small to be a reliable origin study
- this version keeps the same oracle and circuit analysis path, but gives the model enough unique examples to actually learn the retrieval rule

Recommended generation command:
```bash
python -m scripts.generate_kv_retrieval_dataset \
  --outdir dataset/phase2/kv_retrieve_textual_balanced_v1 \
  --dataset-name kv_retrieve_textual_balanced_v1 \
  --train-size 30000 \
  --val-size 3000 \
  --test-size 3000 \
  --ood-size 3000 \
  --num-keys 16 \
  --num-values 16 \
  --keys-file dataset/phase2/textual_retrieval_keys_v1.txt \
  --values-file dataset/phase2/textual_retrieval_values_v1.txt \
  --context-pairs 3 \
  --train-context-pairs 2,3 \
  --ood-context-pairs 4 \
  --query-slot-policy balanced \
  --seed 17
```

Properties of the improved generator:
- global prompt deduplication across splits by default
- balanced queried-slot coverage by default
- optional textual vocab files
- explicit error if the requested split size is too large for the available unique prompt space

## `phase2/random_story_dataset_v1.txt`

Purpose:

- simple next-token negative control for the training-dynamics harness

Status:

- included for completeness
- not recommended as the main mechanism benchmark

## `phase3/kv_symbolic_balanced_v1`

Purpose:

- reduced symbolic retrieval benchmark for the formation-causal follow-up work
- smaller and cleaner than the textual benchmark
- used to study how gradient descent builds retrieval motifs during training
