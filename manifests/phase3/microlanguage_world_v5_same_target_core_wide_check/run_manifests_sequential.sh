#!/bin/bash
set -euo pipefail

python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_v5_same_target_core_wide_check/lr0p001_l2_ep80_ms64/lr0p001_l2_ep80_ms64_seed0.json
