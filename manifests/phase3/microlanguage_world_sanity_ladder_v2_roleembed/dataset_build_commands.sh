#!/bin/bash
set -euo pipefail

python scripts/generate_microlanguage_world_dataset.py --preset v10_person_room_direct --outdir /Users/nelson/py/mechanistic_inter/dataset/phase3/microlanguage_world_v10_person_room_direct_overfit32 --train-size 32 --val-size 256 --test-size 256 --ood-size 256 --seed 0
python scripts/generate_microlanguage_world_dataset.py --preset v10_person_room_direct --outdir /Users/nelson/py/mechanistic_inter/dataset/phase3/microlanguage_world_v10_person_room_direct_iid --train-size 4096 --val-size 512 --test-size 512 --ood-size 512 --seed 0
