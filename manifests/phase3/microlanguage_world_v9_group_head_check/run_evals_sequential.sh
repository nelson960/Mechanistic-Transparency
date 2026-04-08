#!/bin/bash
set -euo pipefail

python -m scripts.run_microlanguage_checkpoint_eval --run-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_v9_group_head_check/lr0p001_l2_ep80_ms64/lr0p001_l2_ep80_ms64_seed0 --skip-complete
