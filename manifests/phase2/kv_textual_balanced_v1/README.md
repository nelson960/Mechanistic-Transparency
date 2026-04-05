# KV Textual Balanced V1

Run these commands from the repo root.

## 1. Generate the dataset

```bash
cd .
conda activate ml

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

## 2. Run the pilot

```bash
python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2.json

python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2

python -m scripts.summarize_training_dynamics \
  --target-dir runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2
```

## 3. Run the full baseline matrix

The full matrix is:

- `curriculum=on`, seeds `0`, `1`, `2`
- `curriculum=off`, seeds `0`, `1`, `2`

The runs are grouped by condition so you can summarize across seeds cleanly.

### Curriculum On

```bash
python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_on_d64_l2.json

python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_on_d64_l2.json

python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_on_d64_l2.json
```

```bash
python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed0_curriculum_on_d64_l2 \
  --skip-complete

python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed1_curriculum_on_d64_l2 \
  --skip-complete

python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed2_curriculum_on_d64_l2 \
  --skip-complete
```

```bash
python -m scripts.summarize_training_dynamics \
  --target-dir runs/kv_textual_balanced_v1/curriculum_on
```

### Curriculum Off

```bash
python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_off_d64_l2.json

python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_off_d64_l2.json

python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_off_d64_l2.json
```

```bash
python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_off/baseline_seed0_curriculum_off_d64_l2 \
  --skip-complete

python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_off/baseline_seed1_curriculum_off_d64_l2 \
  --skip-complete

python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_off/baseline_seed2_curriculum_off_d64_l2 \
  --skip-complete
```

```bash
python -m scripts.summarize_training_dynamics \
  --target-dir runs/kv_textual_balanced_v1/curriculum_off
```

### Optional visualization for one completed run

```bash
python -m scripts.visualize_kv_circuit_dynamics \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed0_curriculum_on_d64_l2 \
  --out-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed0_curriculum_on_d64_l2_visuals \
  --checkpoint-kind scheduled
```

## Why this setup

- much larger deduplicated train/val/test splits
- balanced queried-slot coverage
- same retrieval task and same mechanistic harness
- `d_model=64`, `2` layers, `2` heads
- lower learning rate than the failed story run
- explicit curriculum-on versus curriculum-off comparison
- grouped output directories so cross-seed summaries work without manual file moves
