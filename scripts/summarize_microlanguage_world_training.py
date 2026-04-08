#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scripts.training_dynamics import build_training_intervention_signature, load_run_manifest


def discover_run_directories(target_dir: Path) -> list[Path]:
    target_dir = target_dir.expanduser().resolve()
    if (target_dir / "manifest.json").exists():
        return [target_dir]
    child_run_dirs = sorted(
        child for child in target_dir.iterdir()
        if child.is_dir() and (child / "manifest.json").exists()
    )
    if not child_run_dirs:
        raise ValueError(f"No run directories found under {target_dir}")
    return child_run_dirs


def _manifest_signature(manifest) -> str:
    payload = manifest.to_dict()
    payload["training"].pop("seed", None)
    payload["sae_tracking"].pop("seed", None)
    payload.pop("output_dir", None)
    return json.dumps(payload, sort_keys=True)


def _assert_compatible_manifests(manifests: list[Any]) -> None:
    if not manifests:
        raise ValueError("Expected at least one manifest to summarize")
    first_signature = _manifest_signature(manifests[0])
    for manifest in manifests[1:]:
        if _manifest_signature(manifest) != first_signature:
            raise ValueError("All runs under a summarized target directory must share the same non-seed manifest fields")


def _load_selected_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("selected_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Checkpoint payload is missing selected_metrics")
    required = [
        "train_loss",
        "train_accuracy",
        "train_margin",
        "val_loss",
        "val_accuracy",
        "val_margin",
        "test_loss",
        "test_accuracy",
        "test_margin",
        "ood_loss",
        "ood_accuracy",
        "ood_margin",
    ]
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"Checkpoint selected_metrics is missing required keys: {missing}")
    return {name: float(metrics[name]) for name in required}


def collect_run_checkpoint_rows(run_dir: Path, manifest) -> list[dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        metrics = _load_selected_metrics(payload)
        rows.append(
            {
                "run_dir": str(run_dir.resolve()),
                "run_id": str(payload.get("run_id") or run_dir.name),
                "seed": int(payload.get("seed", manifest.training.seed)),
                "intervention_signature": build_training_intervention_signature(manifest.training_interventions),
                "intervention_count": int(len(manifest.training_interventions)),
                "checkpoint_id": checkpoint_id,
                "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                "global_step": int(payload.get("global_step", 0)),
                "save_reason": str(payload.get("save_reason")),
                **metrics,
            }
        )
    if not rows:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    return rows


def collect_run_slice_rows(run_dir: Path, manifest) -> list[dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        slice_path = run_dir / "battery" / checkpoint_id / "slice_metrics.csv"
        if not slice_path.exists():
            raise FileNotFoundError(f"Missing microlanguage slice metrics: {slice_path}")
        slice_table = pd.read_csv(slice_path)
        required_columns = {"split_name", "slice_kind", "slice_value", "slice_order", "rows", "accuracy", "margin"}
        missing = sorted(required_columns - set(slice_table.columns))
        if missing:
            raise ValueError(f"Slice metrics {slice_path} is missing required columns: {missing}")
        for _, row in slice_table.iterrows():
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_id": str(payload.get("run_id") or run_dir.name),
                    "seed": int(payload.get("seed", manifest.training.seed)),
                    "checkpoint_id": checkpoint_id,
                    "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                    "save_reason": str(payload.get("save_reason")),
                    "split_name": str(row["split_name"]),
                    "slice_kind": str(row["slice_kind"]),
                    "slice_value": str(row["slice_value"]),
                    "slice_order": int(row["slice_order"]),
                    "rows": int(row["rows"]),
                    "accuracy": float(row["accuracy"]),
                    "margin": float(row["margin"]),
                }
            )
    return rows


def _final_checkpoint_table(checkpoint_table: pd.DataFrame) -> pd.DataFrame:
    save_reason_priority = {
        "best_val": 0,
        "scheduled": 1,
        "final": 2,
        "early_stop": 3,
    }
    ordered = checkpoint_table.copy()
    ordered["_final_checkpoint_priority"] = ordered["save_reason"].map(save_reason_priority).fillna(-1).astype(int)
    final_rows = (
        ordered.sort_values(
            ["run_id", "epoch", "_final_checkpoint_priority", "checkpoint_id"],
            ascending=[True, True, True, True],
        )
        .groupby("run_id", as_index=False)
        .tail(1)
        .drop(columns="_final_checkpoint_priority")
    )
    return final_rows.reset_index(drop=True)


def _final_slice_table(checkpoint_table: pd.DataFrame, slice_table: pd.DataFrame) -> pd.DataFrame:
    final_checkpoints = _final_checkpoint_table(checkpoint_table)[["run_id", "checkpoint_id"]].copy()
    return slice_table.merge(final_checkpoints, on=["run_id", "checkpoint_id"], how="inner")


def _summarize_overall_seed_stability(final_checkpoint_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_name in [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "ood_accuracy",
        "train_margin",
        "val_margin",
        "test_margin",
        "ood_margin",
    ]:
        series = final_checkpoint_table[metric_name].astype(float)
        rows.append(
            {
                "metric_name": metric_name,
                "num_runs": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_slice_seed_stability(final_slice_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = final_slice_table.groupby(["split_name", "slice_kind", "slice_value", "slice_order"], dropna=False)
    for (split_name, slice_kind, slice_value, slice_order), group in grouped:
        for metric_name in ["accuracy", "margin"]:
            series = group[metric_name].astype(float)
            rows.append(
                {
                    "split_name": split_name,
                    "slice_kind": slice_kind,
                    "slice_value": slice_value,
                    "slice_order": int(slice_order),
                    "metric_name": metric_name,
                    "num_runs": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["split_name", "slice_kind", "metric_name", "slice_order", "slice_value"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def _lookup_slice_accuracy(
    final_slice_table: pd.DataFrame,
    *,
    run_id: str,
    split_name: str,
    slice_kind: str,
    slice_value: str,
) -> float | None:
    rows = final_slice_table[
        (final_slice_table["run_id"] == run_id)
        & (final_slice_table["split_name"] == split_name)
        & (final_slice_table["slice_kind"] == slice_kind)
        & (final_slice_table["slice_value"] == slice_value)
    ]
    if rows.empty:
        return None
    return float(rows.iloc[0]["accuracy"])


def _lookup_hardest_bucket(
    final_slice_table: pd.DataFrame,
    *,
    run_id: str,
    split_name: str,
    slice_kind: str,
) -> tuple[str | None, float | None]:
    rows = final_slice_table[
        (final_slice_table["run_id"] == run_id)
        & (final_slice_table["split_name"] == split_name)
        & (final_slice_table["slice_kind"] == slice_kind)
    ].sort_values(["slice_order", "slice_value"])
    if rows.empty:
        return None, None
    hardest = rows.iloc[-1]
    return str(hardest["slice_value"]), float(hardest["accuracy"])


def _build_stability_diagnostics(
    final_checkpoint_table: pd.DataFrame,
    final_slice_table: pd.DataFrame,
    *,
    test_split_name: str,
    ood_split_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, checkpoint_row in final_checkpoint_table.iterrows():
        run_id = str(checkpoint_row["run_id"])
        seed = int(checkpoint_row["seed"])
        for split_name, overall_metric_name in [
            (test_split_name, "test_accuracy"),
            (ood_split_name, "ood_accuracy"),
        ]:
            family_rows = final_slice_table[
                (final_slice_table["run_id"] == run_id)
                & (final_slice_table["split_name"] == split_name)
                & (final_slice_table["slice_kind"] == "query_family")
            ]
            if family_rows.empty:
                raise ValueError(
                    f"Final slice table is missing query_family rows for run_id={run_id!r}, split={split_name!r}"
                )
            hardest_overwrite_label, hardest_overwrite_accuracy = _lookup_hardest_bucket(
                final_slice_table,
                run_id=run_id,
                split_name=split_name,
                slice_kind="trace_overwrite_bucket",
            )
            longest_prompt_label, longest_prompt_accuracy = _lookup_hardest_bucket(
                final_slice_table,
                run_id=run_id,
                split_name=split_name,
                slice_kind="prompt_length_bucket",
            )
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "split_name": split_name,
                    "overall_accuracy": float(checkpoint_row[overall_metric_name]),
                    "depth_1_accuracy": _lookup_slice_accuracy(
                        final_slice_table,
                        run_id=run_id,
                        split_name=split_name,
                        slice_kind="query_depth",
                        slice_value="1",
                    ),
                    "depth_2_accuracy": _lookup_slice_accuracy(
                        final_slice_table,
                        run_id=run_id,
                        split_name=split_name,
                        slice_kind="query_depth",
                        slice_value="2",
                    ),
                    "depth_3_accuracy": _lookup_slice_accuracy(
                        final_slice_table,
                        run_id=run_id,
                        split_name=split_name,
                        slice_kind="query_depth",
                        slice_value="3",
                    ),
                    "family_min_accuracy": float(family_rows["accuracy"].min()),
                    "family_max_accuracy": float(family_rows["accuracy"].max()),
                    "family_gap": float(family_rows["accuracy"].max() - family_rows["accuracy"].min()),
                    "family_count": int(len(family_rows)),
                    "hardest_overwrite_bucket": hardest_overwrite_label,
                    "hardest_overwrite_accuracy": hardest_overwrite_accuracy,
                    "longest_prompt_bucket": longest_prompt_label,
                    "longest_prompt_accuracy": longest_prompt_accuracy,
                }
            )
    diagnostics = pd.DataFrame(rows)
    diagnostics["depth_1_minus_depth_3_gap"] = diagnostics["depth_1_accuracy"] - diagnostics["depth_3_accuracy"]
    return diagnostics.sort_values(["split_name", "seed", "run_id"]).reset_index(drop=True)


def _metric_summary(series: pd.Series) -> dict[str, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _build_run_summary(
    checkpoint_table: pd.DataFrame,
    stability_diagnostics: pd.DataFrame,
    *,
    test_split_name: str,
    ood_split_name: str,
) -> dict[str, Any]:
    final_checkpoints = _final_checkpoint_table(checkpoint_table)
    ood_rows = stability_diagnostics[stability_diagnostics["split_name"] == ood_split_name]
    test_rows = stability_diagnostics[stability_diagnostics["split_name"] == test_split_name]
    return {
        "num_runs": int(final_checkpoints["run_id"].nunique()),
        "num_checkpoints": int(len(checkpoint_table)),
        "final_val_accuracy": _metric_summary(final_checkpoints["val_accuracy"]),
        "final_test_accuracy": _metric_summary(final_checkpoints["test_accuracy"]),
        "final_ood_accuracy": _metric_summary(final_checkpoints["ood_accuracy"]),
        "ood_depth_2_accuracy": _metric_summary(ood_rows["depth_2_accuracy"]),
        "ood_depth_3_accuracy": _metric_summary(ood_rows["depth_3_accuracy"]),
        "ood_family_min_accuracy": _metric_summary(ood_rows["family_min_accuracy"]),
        "ood_family_gap": _metric_summary(ood_rows["family_gap"]),
        "ood_depth_1_minus_depth_3_gap": _metric_summary(ood_rows["depth_1_minus_depth_3_gap"]),
        "test_depth_2_accuracy": _metric_summary(test_rows["depth_2_accuracy"]),
        "test_depth_3_accuracy": _metric_summary(test_rows["depth_3_accuracy"]),
        "test_family_min_accuracy": _metric_summary(test_rows["family_min_accuracy"]),
        "test_family_gap": _metric_summary(test_rows["family_gap"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize checkpoint-level microlanguage world training dynamics for one run or a directory of runs."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Either a single run directory or a parent directory that contains homogeneous seed runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    run_dirs = discover_run_directories(target_dir)
    manifests = [load_run_manifest(run_dir / "manifest.json") for run_dir in run_dirs]
    if any(manifest.benchmark.name != "microlanguage_world_next_token" for manifest in manifests):
        raise ValueError("summarize_microlanguage_world_training.py only supports microlanguage world runs")
    _assert_compatible_manifests(manifests)
    manifest = manifests[0]

    checkpoint_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for run_dir, manifest in zip(run_dirs, manifests, strict=True):
        checkpoint_rows.extend(collect_run_checkpoint_rows(run_dir, manifest))
        slice_rows.extend(collect_run_slice_rows(run_dir, manifest))

    checkpoint_table = pd.DataFrame(checkpoint_rows).sort_values(
        ["run_id", "epoch", "checkpoint_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    slice_table = pd.DataFrame(slice_rows).sort_values(
        ["run_id", "epoch", "split_name", "slice_kind", "slice_order", "slice_value"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)
    final_checkpoint_table = _final_checkpoint_table(checkpoint_table)
    final_slice_table = _final_slice_table(checkpoint_table, slice_table)
    overall_seed_stability = _summarize_overall_seed_stability(final_checkpoint_table)
    slice_seed_stability = _summarize_slice_seed_stability(final_slice_table)
    stability_diagnostics = _build_stability_diagnostics(
        final_checkpoint_table,
        final_slice_table,
        test_split_name=manifest.dataset.eval_splits["test"],
        ood_split_name=manifest.dataset.eval_splits["ood"],
    )
    run_summary = _build_run_summary(
        checkpoint_table,
        stability_diagnostics,
        test_split_name=manifest.dataset.eval_splits["test"],
        ood_split_name=manifest.dataset.eval_splits["ood"],
    )

    summary_dir = (target_dir if len(run_dirs) > 1 else run_dirs[0]) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_table.to_csv(summary_dir / "checkpoint_index.csv", index=False)
    slice_table.to_csv(summary_dir / "slice_dynamics.csv", index=False)
    final_slice_table[final_slice_table["slice_kind"] == "query_family"].to_csv(
        summary_dir / "final_query_family_metrics.csv",
        index=False,
    )
    final_slice_table[final_slice_table["slice_kind"] == "query_depth"].to_csv(
        summary_dir / "final_depth_metrics.csv",
        index=False,
    )
    final_slice_table[
        final_slice_table["slice_kind"].isin(["prompt_length_bucket", "trace_overwrite_bucket", "trace_recent_gap_bucket"])
    ].to_csv(
        summary_dir / "final_difficulty_metrics.csv",
        index=False,
    )
    overall_seed_stability.to_csv(summary_dir / "seed_stability.csv", index=False)
    slice_seed_stability.to_csv(summary_dir / "slice_seed_stability.csv", index=False)
    stability_diagnostics.to_csv(summary_dir / "stability_diagnostics.csv", index=False)
    (summary_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
