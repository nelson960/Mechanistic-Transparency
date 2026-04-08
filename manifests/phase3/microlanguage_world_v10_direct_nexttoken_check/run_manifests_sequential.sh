#!/bin/bash
set -euo pipefail

python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_v10_direct_nexttoken_check/micro_world_seed0_d64_l3_lr0p001_wd0_init1.json
