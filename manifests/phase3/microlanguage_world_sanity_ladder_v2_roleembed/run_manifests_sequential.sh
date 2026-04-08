#!/bin/bash
set -euo pipefail

python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_sanity_ladder_v2_roleembed/overfit32_l2/overfit32_l2_seed0.json
python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_sanity_ladder_v2_roleembed/iid_l2/iid_l2_seed0.json
python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_sanity_ladder_v2_roleembed/iid_l3/iid_l3_seed0.json
