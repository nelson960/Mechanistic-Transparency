---
layout: default
title: Reproducibility
---

# Reproducibility

This page is supporting material for the paper. Run commands from the repository root. The folder names shown below reflect the repository layout.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Static KV Analysis

Static analysis uses the checked-in symbolic KV dataset and checkpoint:

- `dataset/phase2/kv_retrieve_3`
- `models/kv_retrieve_3/selected_checkpoint.pt`
- `notebook/kv_retrieve_algorithm_analysis.ipynb`

The symbolic reverse-engineering walkthrough is notebook-based rather than exposed as one single CLI reproduction script.

## Story Negative Control

```bash
.venv/bin/python -m scripts.train_story_text_circuit_loop \
  --text-path dataset/phase2/random_story_dataset_v1.txt \
  --run-dir runs/story_text_circuit_run_001 \
  --context-length 24 \
  --ood-context-length 40 \
  --stride 1 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --epochs 150 \
  --batch-size 32 \
  --eval-batch-size 128 \
  --learning-rate 0.01 \
  --weight-decay 0.0 \
  --seed 0 \
  --device cpu \
  --d-model 32 \
  --n-heads 2 \
  --d-ff 64 \
  --n-layers 2 \
  --max-seq-len 64 \
  --dense-through-epoch 20 \
  --log-spaced-epoch-count 24 \
  --save-epoch-zero \
  --sweep-base-limit 32 \
  --train-probe-limit 128 \
  --sae-train-limit 128 \
  --sae-val-limit 64 \
  --sae-hidden-multiplier 2 \
  --sae-l1-coeff 0.001 \
  --sae-learning-rate 0.01 \
  --sae-batch-size 64 \
  --sae-epochs 5 \
  --top-features-per-site 5 \
  --superposition-cosine-threshold 0.2 \
  --role-top-k 3
```

## Textual KV Dataset

```bash
.venv/bin/python -m scripts.generate_kv_retrieval_dataset \
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

## Textual KV Pilot

```bash
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2.json

.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2 \
  --skip-complete

.venv/bin/python -m scripts.summarize_training_dynamics \
  --target-dir runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2
```

## Textual KV Full Matrix

### Curriculum On

```bash
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_on_d64_l2.json
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_on_d64_l2.json
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_on_d64_l2.json
```

```bash
.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed0_curriculum_on_d64_l2 \
  --skip-complete
.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed1_curriculum_on_d64_l2 \
  --skip-complete
.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_on/baseline_seed2_curriculum_on_d64_l2 \
  --skip-complete
```

```bash
.venv/bin/python -m scripts.summarize_training_dynamics \
  --target-dir runs/kv_textual_balanced_v1/curriculum_on
```

### Curriculum Off

```bash
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed0_curriculum_off_d64_l2.json
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed1_curriculum_off_d64_l2.json
.venv/bin/python -m scripts.train_run \
  --manifest manifests/phase2/kv_textual_balanced_v1/baseline_seed2_curriculum_off_d64_l2.json
```

```bash
.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_off/baseline_seed0_curriculum_off_d64_l2 \
  --skip-complete
.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_off/baseline_seed1_curriculum_off_d64_l2 \
  --skip-complete
.venv/bin/python -m scripts.run_checkpoint_battery \
  --run-dir runs/kv_textual_balanced_v1/curriculum_off/baseline_seed2_curriculum_off_d64_l2 \
  --skip-complete
```

```bash
.venv/bin/python -m scripts.summarize_training_dynamics \
  --target-dir runs/kv_textual_balanced_v1/curriculum_off
```

## Visuals

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp .venv/bin/python -m scripts.visualize_kv_circuit_dynamics \
  --run-dir runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2 \
  --out-dir runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2_visuals_v4 \
  --checkpoint-kind scheduled
```

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp .venv/bin/python -m scripts.render_public_matrix_figures \
  --matrix-dir artifacts/phase2/textual_kv/full_matrix \
  --out-dir docs/figures
```

## GPT-2 Activation Viewer

```bash
.venv/bin/python scripts/build_interactive_model_viewer.py \
  --prompt "The secret code is 73914. Repeat the secret code exactly:" \
  --prompt-id demo_prompt_001 \
  --task copy \
  --model gpt2-small \
  --out outputs/viewer_payload.json
```

```bash
cd webapp
.venv/bin/python -m http.server 8000
```
