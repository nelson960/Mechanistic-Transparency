#!/bin/bash
set -euo pipefail

python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_v4_twohop_core_fresh_check/lr0p001_l2_ep60_ms48/lr0p001_l2_ep60_ms48_seed0.json
python -m scripts.run_microlanguage_checkpoint_eval --run-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_v4_twohop_core_fresh_check/lr0p001_l2_ep60_ms48/lr0p001_l2_ep60_ms48_seed0 --skip-complete
python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_v4_twohop_core_fresh_check/lr0p001_l2_ep60_ms48
