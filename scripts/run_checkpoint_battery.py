#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts.kv_benchmark import run_kv_checkpoint_battery
from scripts.training_dynamics import discover_checkpoints, load_run_manifest


def _checkpoint_battery_is_complete(run_dir: Path, checkpoint_id: str, *, sae_enabled: bool) -> bool:
    battery_dir = run_dir / "battery" / checkpoint_id
    required_paths = [
        battery_dir / "behavior.json",
        battery_dir / "variable_scores.csv",
        battery_dir / "variable_faithfulness.csv",
        battery_dir / "operator_scores.csv",
        battery_dir / "localization.csv",
        battery_dir / "weight_metrics.json",
        battery_dir / "neuron_scores.csv",
        battery_dir / "canonical_site_vectors.pt",
        battery_dir / "tensors.pt",
    ]
    if sae_enabled:
        required_paths.extend(
            [
                battery_dir / "feature_scores.csv",
                battery_dir / "superposition_metrics.json",
            ]
        )
    return all(path.exists() for path in required_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KV checkpoint battery over a manifest-backed run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing manifest.json and checkpoints/.")
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Optional checkpoint stem. If omitted, the battery runs over every checkpoint in the run.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip checkpoints whose battery artifacts are already complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = load_run_manifest(run_dir / "manifest.json")
    device = torch.device(manifest.training.device)
    checkpoint_paths = discover_checkpoints(run_dir)
    if args.checkpoint_id is not None:
        checkpoint_paths = [path for path in checkpoint_paths if path.stem == args.checkpoint_id]
        if not checkpoint_paths:
            raise ValueError(f"Checkpoint id {args.checkpoint_id!r} was not found under {run_dir / 'checkpoints'}")

    progress = tqdm(checkpoint_paths, desc="checkpoint_battery", unit="ckpt")
    for checkpoint_path in progress:
        checkpoint_id = checkpoint_path.stem
        progress.set_postfix_str(checkpoint_id)
        if args.skip_complete and _checkpoint_battery_is_complete(
            run_dir,
            checkpoint_id,
            sae_enabled=manifest.sae_tracking.enabled,
        ):
            progress.write(f"[battery:{checkpoint_id}] skip complete")
            continue
        run_kv_checkpoint_battery(
            manifest=manifest,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            announce=True,
        )


if __name__ == "__main__":
    main()
