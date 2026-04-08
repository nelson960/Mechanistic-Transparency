#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts.device_utils import resolve_training_device
from scripts.microlanguage_world_eval import run_microlanguage_checkpoint_eval
from scripts.training_dynamics import discover_checkpoints, load_run_manifest


def _checkpoint_eval_is_complete(run_dir: Path, checkpoint_id: str) -> bool:
    battery_dir = run_dir / "battery" / checkpoint_id
    required_paths = [
        battery_dir / "behavior.json",
        battery_dir / "slice_metrics.csv",
        battery_dir / "scored_rows.csv",
    ]
    return all(path.exists() for path in required_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run microlanguage checkpoint evaluation over a manifest-backed run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing manifest.json and checkpoints/.")
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Optional checkpoint stem. If omitted, evaluation runs over every checkpoint in the run.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip checkpoints whose evaluation artifacts are already complete.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional evaluation device override. Defaults to manifest.training.device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = load_run_manifest(run_dir / "manifest.json")
    if manifest.benchmark.name != "microlanguage_world_next_token":
        raise ValueError(
            "run_microlanguage_checkpoint_eval.py requires benchmark.name='microlanguage_world_next_token'"
        )
    requested_device = args.device if args.device is not None else manifest.training.device
    device = torch.device(resolve_training_device(requested_device))
    checkpoint_paths = discover_checkpoints(run_dir)
    if args.checkpoint_id is not None:
        checkpoint_paths = [path for path in checkpoint_paths if path.stem == args.checkpoint_id]
        if not checkpoint_paths:
            raise ValueError(f"Checkpoint id {args.checkpoint_id!r} was not found under {run_dir / 'checkpoints'}")

    progress = tqdm(checkpoint_paths, desc="micro_checkpoint_eval", unit="ckpt")
    for checkpoint_path in progress:
        checkpoint_id = checkpoint_path.stem
        progress.set_postfix_str(checkpoint_id)
        if args.skip_complete and _checkpoint_eval_is_complete(run_dir, checkpoint_id):
            progress.write(f"[micro-eval:{checkpoint_id}] skip complete")
            continue
        run_microlanguage_checkpoint_eval(
            manifest=manifest,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            announce=True,
        )


if __name__ == "__main__":
    main()
