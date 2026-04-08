#!/bin/bash
set -euo pipefail

python -m scripts.run_microlanguage_checkpoint_eval --run-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_v8_scaffolded_room_chain_check/lr0p0003_l2_ep80_ms64/lr0p0003_l2_ep80_ms64_seed0 --skip-complete
