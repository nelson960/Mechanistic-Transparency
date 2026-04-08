#!/bin/bash
set -euo pipefail

python scripts/generate_microlanguage_world_dataset.py --preset v10_person_room_direct --outdir /Users/nelson/py/mechanistic_inter/dataset/phase3/microlanguage_world_v10_person_room_direct_overfit32 --train-size 32 --val-size 256 --test-size 256 --ood-size 256 --seed 0
python scripts/generate_microlanguage_world_dataset.py --preset v10_person_room_direct --outdir /Users/nelson/py/mechanistic_inter/dataset/phase3/microlanguage_world_v10_person_room_direct_iid --train-size 4096 --val-size 512 --test-size 512 --ood-size 512 --seed 0
python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_sanity_ladder_v1/overfit32_l2/overfit32_l2_seed0.json
python -m scripts.run_microlanguage_checkpoint_eval --run-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v1/overfit32_l2/overfit32_l2_seed0 --skip-complete
python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v1/overfit32_l2
python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_sanity_ladder_v1/iid_l2/iid_l2_seed0.json
python -m scripts.run_microlanguage_checkpoint_eval --run-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v1/iid_l2/iid_l2_seed0 --skip-complete
python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v1/iid_l2
python -m scripts.train_microlanguage_world_run --manifest /Users/nelson/py/mechanistic_inter/manifests/phase3/microlanguage_world_sanity_ladder_v1/iid_l3/iid_l3_seed0.json
python -m scripts.run_microlanguage_checkpoint_eval --run-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v1/iid_l3/iid_l3_seed0 --skip-complete
python -m scripts.summarize_microlanguage_world_training --target-dir /Users/nelson/py/mechanistic_inter/runs/phase3/microlanguage_world_sanity_ladder_v1/iid_l3
