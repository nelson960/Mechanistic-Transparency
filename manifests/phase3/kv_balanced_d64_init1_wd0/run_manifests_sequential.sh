#!/usr/bin/env bash
set -euo pipefail

/opt/miniconda3/envs/ml/bin/python -m research.phase3.scripts.run_kv_factor_screen --manifest-dir /Users/nelson/py/mechanistic_inter/manifests/phase3/kv_balanced_d64_init1_wd0 --python-bin /opt/miniconda3/envs/ml/bin/python
