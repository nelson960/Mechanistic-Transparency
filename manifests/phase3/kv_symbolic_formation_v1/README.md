# KV Symbolic Formation V1

This manifest set is for the formation-causal study on the reduced symbolic KV benchmark.

Core properties:

- dataset: symbolic KV retrieval
- training regime: `2_pairs_only` via a single `train` split
- OOD split: `3` pairs
- model: `2` layers, `2` heads, `d_model=64`
- curriculum: `off`
- formation mode: enabled, candidate discovery on, family-gradient logging on

Recommended workflow:

1. Generate the dataset:

```bash
python -m scripts.generate_kv_retrieval_dataset \
  --outdir dataset/phase3/kv_symbolic_balanced_v1 \
  --dataset-name kv_symbolic_balanced_v1 \
  --train-size 30000 \
  --val-size 3000 \
  --test-size 3000 \
  --ood-size 3000 \
  --num-keys 16 \
  --num-values 16 \
  --context-pairs 2 \
  --train-context-pairs 2 \
  --ood-context-pairs 3 \
  --query-slot-policy balanced \
  --seed 17
```

2. Run the discovered-candidate baselines:

```bash
python -m scripts.train_run --manifest manifests/phase3/kv_symbolic_formation_v1/baseline_seed0_off_d64_l2_discovered.json
python -m scripts.train_run --manifest manifests/phase3/kv_symbolic_formation_v1/baseline_seed1_off_d64_l2_discovered.json
python -m scripts.train_run --manifest manifests/phase3/kv_symbolic_formation_v1/baseline_seed2_off_d64_l2_discovered.json
```

3. Summarize the formation traces:

```bash
python -m scripts.summarize_training_dynamics --target-dir research/phase3/runs/kv_symbolic_formation_v1/baseline
```

4. Derive fixed-head intervention manifests from a completed baseline:

```bash
python -m research.phase3.scripts.build_kv_formation_intervention_manifest \
  --baseline-run-dir research/phase3/runs/kv_symbolic_formation_v1/baseline/baseline_seed0_off_d64_l2_discovered \
  --intervention-role support \
  --epoch-start 1 \
  --epoch-end 20 \
  --scale 0.0 \
  --output-manifest manifests/phase3/kv_symbolic_formation_v1/interventions/support_damp_seed0.json
```

Then run the derived manifest with `scripts.train_run`.
