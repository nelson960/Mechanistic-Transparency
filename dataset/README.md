# Synthetic Datasets

This folder contains synthetic datasets for mechanistic interpretability experiments on small toy transformer models.

## Available Dataset

### `kv_retrieve_3`

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
- `kv_retrieve_3/train.jsonl`
- `kv_retrieve_3/val.jsonl`
- `kv_retrieve_3/test.jsonl`
- `kv_retrieve_3/test_ood_4_pairs.jsonl`
- `kv_retrieve_3/metadata.json`

Each row contains:
- `id`
- `task`
- `split`
- `num_pairs`
- `prompt`
- `target`
- `query_key`
- `context_pairs`
