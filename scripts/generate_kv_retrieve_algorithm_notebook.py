#!/usr/bin/env python3
"""Generate the artifact-first KV analysis notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.generate_kv_retrieve_algorithm_discovery_notebook import build_notebook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_PATH = ROOT / "notebook" / "kv_retrieve_algorithm_analysis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the artifact-first KV analysis notebook.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory that contains manifest.json.")
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        required=True,
        help="Checkpoint stem to open from the run battery directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Output notebook path.",
    )
    parser.add_argument(
        "--visuals-dir",
        type=Path,
        default=None,
        help="Optional visualization directory produced by visualize_kv_circuit_dynamics.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = build_notebook(run_dir=args.run_dir, checkpoint_id=args.checkpoint_id, visuals_dir=args.visuals_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
    print(f"Wrote notebook to {args.out}")


if __name__ == "__main__":
    main()
