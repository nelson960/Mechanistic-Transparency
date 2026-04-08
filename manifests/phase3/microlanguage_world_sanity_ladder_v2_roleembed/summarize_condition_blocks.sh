#!/bin/bash
set -euo pipefail

python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v2_roleembed/overfit32_l2
python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v2_roleembed/iid_l2
python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v2_roleembed/iid_l3
