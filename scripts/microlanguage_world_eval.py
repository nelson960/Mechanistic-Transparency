from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from scripts.generate_microlanguage_world_dataset import parse_prompt, resolve_prompt_target
from scripts.microlanguage_world_benchmark import (
    evaluate_microlanguage_world_rows,
    load_microlanguage_world_bundle,
)
from scripts.tiny_transformer_core import load_decoder_checkpoint
from scripts.training_dynamics import RunManifest


def _row_prompt_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("prompt_id") or row["prompt"])


def _derive_row_features(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"])
    parsed = parse_prompt(prompt)
    target, trace = resolve_prompt_target(prompt)
    if target != str(row["target"]):
        raise ValueError(
            f"Stored target {row['target']!r} does not match oracle target {target!r} for row {_row_prompt_id(row)!r}"
        )
    query_depth = int(row.get("query_depth", len(trace)))
    if query_depth != len(trace):
        raise ValueError(
            f"Row {_row_prompt_id(row)!r} has query_depth={query_depth}, but oracle trace depth is {len(trace)}"
        )
    event_counts = Counter((event["relation"], event["subject"]) for event in parsed["events"])
    overwrite_counts = [
        max(0, event_counts[(str(step["relation"]), str(step["subject"]))] - 1)
        for step in trace
    ]
    update_count = len(parsed["events"])
    latest_event_index = max(int(step["event_index"]) for step in trace)
    earliest_event_index = min(int(step["event_index"]) for step in trace)
    return {
        "prompt_id": _row_prompt_id(row),
        "query_family": str(row.get("query_family") or parsed["query_family"]),
        "query_depth": query_depth,
        "query_depth_group": "direct" if query_depth == 1 else "multihop",
        "prompt_length": len(prompt.split()),
        "update_count": int(row.get("update_count", update_count)),
        "trace_overwrite_total": int(sum(overwrite_counts)),
        "trace_overwrite_max": int(max(overwrite_counts) if overwrite_counts else 0),
        "trace_recent_gap": int(update_count - 1 - latest_event_index),
        "trace_span": int(latest_event_index - earliest_event_index),
    }


def _build_bucket_upper_bounds(values: Sequence[int], *, bucket_count: int) -> list[int]:
    if bucket_count <= 0:
        raise ValueError(f"bucket_count must be positive, got {bucket_count}")
    if not values:
        raise ValueError("Expected at least one value when building bucket boundaries")
    ordered = sorted(int(value) for value in values)
    upper_bounds: list[int] = []
    for bucket_index in range(1, bucket_count + 1):
        target_index = math.ceil(len(ordered) * bucket_index / bucket_count) - 1
        upper_bound = ordered[target_index]
        if not upper_bounds or upper_bound > upper_bounds[-1]:
            upper_bounds.append(upper_bound)
    if not upper_bounds:
        upper_bounds.append(ordered[-1])
    return upper_bounds


def _assign_bucket(value: int, upper_bounds: Sequence[int]) -> tuple[str, int]:
    for bucket_index, upper_bound in enumerate(upper_bounds):
        if value <= upper_bound:
            if bucket_index == 0:
                return f"<= {upper_bound}", bucket_index
            lower_bound = upper_bounds[bucket_index - 1] + 1
            return f"{lower_bound}..{upper_bound}", bucket_index
    raise ValueError(f"Value {value} exceeded computed bucket bounds {list(upper_bounds)}")


def _annotate_rows_with_buckets(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_bucket_count: int,
    overwrite_bucket_count: int,
    recency_bucket_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_rows = [_derive_row_features(row) for row in rows]
    table = pd.DataFrame(feature_rows)
    prompt_bounds = _build_bucket_upper_bounds(table["prompt_length"].astype(int).tolist(), bucket_count=prompt_bucket_count)
    overwrite_bounds = _build_bucket_upper_bounds(
        table["trace_overwrite_total"].astype(int).tolist(),
        bucket_count=overwrite_bucket_count,
    )
    recency_bounds = _build_bucket_upper_bounds(
        table["trace_recent_gap"].astype(int).tolist(),
        bucket_count=recency_bucket_count,
    )
    prompt_buckets = table["prompt_length"].astype(int).apply(lambda value: _assign_bucket(int(value), prompt_bounds))
    overwrite_buckets = table["trace_overwrite_total"].astype(int).apply(
        lambda value: _assign_bucket(int(value), overwrite_bounds)
    )
    recency_buckets = table["trace_recent_gap"].astype(int).apply(
        lambda value: _assign_bucket(int(value), recency_bounds)
    )
    table["prompt_length_bucket"] = prompt_buckets.apply(lambda item: item[0])
    table["prompt_length_bucket_order"] = prompt_buckets.apply(lambda item: item[1])
    table["trace_overwrite_bucket"] = overwrite_buckets.apply(lambda item: item[0])
    table["trace_overwrite_bucket_order"] = overwrite_buckets.apply(lambda item: item[1])
    table["trace_recent_gap_bucket"] = recency_buckets.apply(lambda item: item[0])
    table["trace_recent_gap_bucket_order"] = recency_buckets.apply(lambda item: item[1])
    return table, {
        "prompt_length_bucket_upper_bounds": prompt_bounds,
        "trace_overwrite_bucket_upper_bounds": overwrite_bounds,
        "trace_recent_gap_bucket_upper_bounds": recency_bounds,
    }


def _build_scored_rows_for_split(
    *,
    split_name: str,
    rows: Sequence[dict[str, Any]],
    predictions: dict[str, Any],
    annotation_table: pd.DataFrame,
) -> pd.DataFrame:
    prediction_lookup = {
        str(prediction["prompt_id"]): prediction
        for prediction in predictions["predictions"]
    }
    annotation_lookup = annotation_table.set_index("prompt_id").to_dict(orient="index")
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt_id = _row_prompt_id(row)
        prediction = prediction_lookup.get(prompt_id)
        if prediction is None:
            raise ValueError(f"Missing prediction row for prompt_id={prompt_id!r} in split {split_name!r}")
        annotation = annotation_lookup.get(prompt_id)
        if annotation is None:
            raise ValueError(f"Missing annotation row for prompt_id={prompt_id!r}")
        scored_rows.append(
            {
                "split_name": split_name,
                "prompt_id": prompt_id,
                "target_token": str(row["target"]),
                "predicted_token": str(prediction["predicted_token"]),
                "foil_token": str(prediction["foil_token"]),
                "correct": bool(prediction["correct"]),
                "margin": float(prediction["margin"]),
                **annotation,
            }
        )
    return pd.DataFrame(scored_rows)


def _aggregate_slice_metrics(scored_rows: pd.DataFrame) -> pd.DataFrame:
    if scored_rows.empty:
        raise ValueError("Expected non-empty scored rows when aggregating slice metrics")
    metric_frames: list[pd.DataFrame] = []
    slice_specs = [
        ("query_family", "query_family", None),
        ("query_depth", "query_depth", "query_depth"),
        ("query_depth_group", "query_depth_group", None),
        ("prompt_length_bucket", "prompt_length_bucket", "prompt_length_bucket_order"),
        ("trace_overwrite_bucket", "trace_overwrite_bucket", "trace_overwrite_bucket_order"),
        ("trace_recent_gap_bucket", "trace_recent_gap_bucket", "trace_recent_gap_bucket_order"),
    ]
    for slice_kind, value_column, order_column in slice_specs:
        group_columns = ["split_name", value_column]
        if order_column is not None and order_column != value_column:
            group_columns.append(order_column)
        grouped = (
            scored_rows.groupby(group_columns, dropna=False)
            .agg(
                rows=("prompt_id", "size"),
                accuracy=("correct", "mean"),
                margin=("margin", "mean"),
            )
            .reset_index()
        )
        grouped["slice_kind"] = slice_kind
        grouped["slice_value"] = grouped[value_column].astype(str)
        if order_column is None:
            grouped["slice_order"] = 0
        else:
            grouped["slice_order"] = grouped[order_column].astype(int)
        metric_frames.append(grouped[["split_name", "slice_kind", "slice_value", "slice_order", "rows", "accuracy", "margin"]])
    combined = pd.concat(metric_frames, ignore_index=True, sort=False)
    return combined.sort_values(["split_name", "slice_kind", "slice_order", "slice_value"]).reset_index(drop=True)


def run_microlanguage_checkpoint_eval(
    *,
    manifest: RunManifest,
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    announce: bool = False,
    prompt_bucket_count: int = 4,
    overwrite_bucket_count: int = 3,
    recency_bucket_count: int = 4,
) -> dict[str, Any]:
    bundle = load_microlanguage_world_bundle(manifest)
    checkpoint_payload, model = load_decoder_checkpoint(checkpoint_path, device)
    checkpoint_id = checkpoint_path.stem
    battery_dir = run_dir / "battery" / checkpoint_id
    battery_dir.mkdir(parents=True, exist_ok=True)

    eval_split_names = [
        manifest.dataset.eval_splits["val"],
        manifest.dataset.eval_splits["test"],
        manifest.dataset.eval_splits["ood"],
    ]
    all_eval_rows = [
        row
        for split_name in eval_split_names
        for row in bundle.raw_splits[split_name]
    ]
    annotation_table, bucket_specs = _annotate_rows_with_buckets(
        all_eval_rows,
        prompt_bucket_count=prompt_bucket_count,
        overwrite_bucket_count=overwrite_bucket_count,
        recency_bucket_count=recency_bucket_count,
    )
    split_metrics: dict[str, Any] = {}
    scored_frames: list[pd.DataFrame] = []
    for split_name in eval_split_names:
        if announce:
            print(f"[micro-eval:{checkpoint_id}] {split_name}", flush=True)
        rows = bundle.raw_splits[split_name]
        metrics = evaluate_microlanguage_world_rows(
            model,
            rows,
            manifest=manifest,
            bundle=bundle,
            device=device,
            batch_size=manifest.battery.eval_batch_size,
        )
        split_metrics[split_name] = {
            "rows": int(metrics["rows"]),
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
            "margin": float(metrics["margin"]),
        }
        scored_frames.append(
            _build_scored_rows_for_split(
                split_name=split_name,
                rows=rows,
                predictions=metrics,
                annotation_table=annotation_table,
            )
        )

    selected_metrics = checkpoint_payload.get("selected_metrics")
    if not isinstance(selected_metrics, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is missing selected_metrics")
    if announce:
        print(f"[micro-eval:{checkpoint_id}] write_artifacts", flush=True)
    split_metrics["train_selected"] = {
        "rows": int(len(bundle.raw_splits[manifest.dataset.train_split_by_pairs["default"]])),
        "loss": float(selected_metrics["train_loss"]),
        "accuracy": float(selected_metrics["train_accuracy"]),
        "margin": float(selected_metrics["train_margin"]),
    }
    behavior_artifact = {
        "run_id": str(checkpoint_payload.get("run_id") or run_dir.name),
        "checkpoint_id": checkpoint_id,
        "epoch": int(checkpoint_payload.get("epoch", checkpoint_payload.get("selected_epoch"))),
        "save_reason": str(checkpoint_payload.get("save_reason")),
        "split_metrics": split_metrics,
        "bucket_specs": bucket_specs,
    }
    scored_rows = pd.concat(scored_frames, ignore_index=True, sort=False).sort_values(
        ["split_name", "query_depth", "query_family", "prompt_id"]
    ).reset_index(drop=True)
    slice_metrics = _aggregate_slice_metrics(scored_rows)
    (battery_dir / "behavior.json").write_text(json.dumps(behavior_artifact, indent=2), encoding="utf-8")
    scored_rows.to_csv(battery_dir / "scored_rows.csv", index=False)
    slice_metrics.to_csv(battery_dir / "slice_metrics.csv", index=False)
    return {
        "behavior": behavior_artifact,
        "scored_rows": scored_rows,
        "slice_metrics": slice_metrics,
    }
