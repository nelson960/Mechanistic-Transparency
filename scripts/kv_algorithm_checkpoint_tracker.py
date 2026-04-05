from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def discover_checkpoint_series(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")
    checkpoints = sorted(
        path for path in model_dir.glob("*.pt") if path.is_file() and "sae_" not in path.name
    )
    return checkpoints


def build_checkpoint_availability_table(model_dir: Path) -> pd.DataFrame:
    checkpoints = discover_checkpoint_series(model_dir)
    rows = [
        {
            "checkpoint_path": str(path),
            "checkpoint_name": path.name,
        }
        for path in checkpoints
    ]
    return pd.DataFrame(rows)


def build_checkpoint_metadata_table(checkpoint_paths: list[Path]) -> pd.DataFrame:
    if not checkpoint_paths:
        raise ValueError("Expected at least one checkpoint path for checkpoint metadata")

    rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metrics = checkpoint.get("selected_metrics") or {}
        rows.append(
            {
                "checkpoint_name": checkpoint_path.name,
                "selected_epoch": checkpoint.get("selected_epoch"),
                "test_accuracy": metrics.get("test_accuracy"),
                "ood_accuracy": metrics.get("ood_accuracy"),
                "all_checks_pass": metrics.get("all_checks_pass"),
            }
        )
    return pd.DataFrame(rows)


def summarize_checkpoint_tracker(model_dir: Path) -> pd.DataFrame:
    checkpoints = discover_checkpoint_series(model_dir)
    return pd.DataFrame(
        [
            {
                "checkpoint_count": len(checkpoints),
                "has_checkpoint_series": len(checkpoints) >= 2,
                "message": (
                    "Checkpoint-series analysis is available."
                    if len(checkpoints) >= 2
                    else "Only one non-SAE checkpoint is available, so training-time emergence cannot yet be evaluated."
                ),
            }
        ]
    )
