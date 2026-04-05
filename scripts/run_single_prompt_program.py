#!/usr/bin/env python3
"""Build a single-prompt prompt-program report with exact head/MLP ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kv_retrieve_analysis import (
    build_head_source_contribution_table,
    build_layer_feature_readout_table,
    build_mlp_neuron_contribution_table,
    build_qk_table,
    load_checkpoint_model,
    load_dataset_bundle,
    run_prompt,
    score_mlp_neuron_ablation_prompt,
    score_rows_with_optional_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-prompt prompt-program report."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/kv_retrieve_3/selected_checkpoint.pt"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/phase2/kv_retrieve_3"),
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("notebook/outputs/kv_retrieve_single_prompt_program.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _load_prompt_row(args: argparse.Namespace, bundle: Any) -> dict[str, Any]:
    using_dataset_row = args.split is not None or args.row_index is not None
    using_prompt = args.prompt is not None or args.target is not None

    if using_dataset_row and using_prompt:
        raise ValueError("Use either --split/--row-index or --prompt/--target, not both.")
    if using_dataset_row:
        if args.split is None or args.row_index is None:
            raise ValueError("Both --split and --row-index are required when selecting a dataset row.")
        if args.split not in bundle.raw_splits:
            raise ValueError(f"Unknown split: {args.split}")
        rows = bundle.raw_splits[args.split]
        if args.row_index < 0 or args.row_index >= len(rows):
            raise ValueError(
                f"row-index {args.row_index} is out of range for split {args.split} with {len(rows)} rows"
            )
        row = dict(rows[args.row_index])
        row["analysis_source"] = "dataset"
        row["analysis_split"] = args.split
        row["analysis_row_index"] = args.row_index
        return row
    if using_prompt:
        if args.prompt is None or args.target is None:
            raise ValueError("Both --prompt and --target are required for manual prompt analysis.")
        return {
            "id": "manual_prompt",
            "task": "manual",
            "split": "manual",
            "num_pairs": None,
            "prompt": args.prompt,
            "target": args.target,
            "query_key": "",
            "context_pairs": [],
            "analysis_source": "manual",
            "analysis_split": "manual",
            "analysis_row_index": -1,
        }
    raise ValueError("Provide either --split/--row-index or --prompt/--target.")


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    bundle = load_dataset_bundle(args.dataset_dir)
    device = torch.device("cpu")
    checkpoint, model = load_checkpoint_model(args.checkpoint, device=device)

    row = _load_prompt_row(args, bundle)
    prompt = row["prompt"]
    target_token = row["target"]

    baseline_result, cache = run_prompt(
        model,
        bundle,
        prompt,
        device=device,
        expected_target=target_token,
        return_cache=True,
    )
    if cache is None:
        raise ValueError("Expected cache for single-prompt program report")
    foil_token = baseline_result["foil_token"]

    stage_readout_df = build_layer_feature_readout_table(
        model=model,
        bundle=bundle,
        cache=cache,
        target_token=target_token,
        foil_token=foil_token,
        top_k=args.top_k,
    )

    row_payload = {
        "prompt": prompt,
        "target": target_token,
        "query_key": row.get("query_key", ""),
    }

    head_summary_rows: list[dict[str, Any]] = []
    head_details: list[dict[str, Any]] = []
    for layer_index in range(checkpoint["config"]["n_layers"]):
        for head_index in range(checkpoint["config"]["n_heads"]):
            qk_df = build_qk_table(
                prompt=prompt,
                cache=cache,
                layer_index=layer_index,
                head_index=head_index,
            )
            source_df = build_head_source_contribution_table(
                model=model,
                bundle=bundle,
                prompt=prompt,
                cache=cache,
                layer_index=layer_index,
                head_index=head_index,
                target_token=target_token,
                foil_token=foil_token,
            )
            ablated_df = score_rows_with_optional_ablation(
                model=model,
                bundle=bundle,
                rows=[row_payload],
                device=device,
                ablation={"layer_index": layer_index, "head_index": head_index},
            )
            ablated_row = ablated_df.iloc[0]
            top_attention_row = qk_df.sort_values("attention_weight", ascending=False).iloc[0]
            top_weighted_row = source_df.sort_values(
                "weighted_target_minus_foil", ascending=False
            ).iloc[0]
            head_summary_rows.append(
                {
                    "component": f"L{layer_index + 1}H{head_index}",
                    "exact_ablation_predicted_token": ablated_row["predicted_token"],
                    "exact_ablation_margin": float(ablated_row["margin"]),
                    "exact_ablation_margin_drop": float(
                        baseline_result["margin"] - ablated_row["margin"]
                    ),
                    "top_attention_source_position": int(top_attention_row["position"]),
                    "top_attention_source_token": top_attention_row["token"],
                    "top_attention_weight": float(top_attention_row["attention_weight"]),
                    "top_weighted_source_position": int(top_weighted_row["source_position"]),
                    "top_weighted_source_token": top_weighted_row["source_token"],
                    "top_weighted_target_minus_foil": float(
                        top_weighted_row["weighted_target_minus_foil"]
                    ),
                }
            )
            head_details.append(
                {
                    "component": f"L{layer_index + 1}H{head_index}",
                    "qk_table": _frame_to_records(qk_df),
                    "source_contribution_table": _frame_to_records(source_df),
                }
            )

    mlp_block_rows: list[dict[str, Any]] = []
    mlp_layers: list[dict[str, Any]] = []
    final_position_index = len(prompt.split()) - 1
    for layer_index in range(checkpoint["config"]["n_layers"]):
        block_ablation_result = score_mlp_neuron_ablation_prompt(
            model=model,
            bundle=bundle,
            prompt=prompt,
            target_token=target_token,
            device=device,
            layer_index=layer_index,
            neuron_index=None,
            position_index=final_position_index,
        )
        neuron_df = build_mlp_neuron_contribution_table(
            model=model,
            bundle=bundle,
            prompt=prompt,
            cache=cache,
            layer_index=layer_index,
            target_token=target_token,
            foil_token=foil_token,
            device=device,
            position=final_position_index,
            top_k=args.top_k,
            include_exact_ablation=True,
        )
        top_neuron_row = neuron_df.sort_values(
            "exact_ablation_margin_drop",
            ascending=False,
        ).iloc[0]
        mlp_block_rows.append(
            {
                "component": f"L{layer_index + 1} MLP",
                "exact_block_ablation_predicted_token": block_ablation_result["predicted_token"],
                "exact_block_ablation_margin": block_ablation_result["margin"],
                "exact_block_ablation_margin_drop": (
                    baseline_result["margin"] - block_ablation_result["margin"]
                ),
                "top_neuron_index_by_margin_drop": int(top_neuron_row["neuron_index"]),
                "top_neuron_margin_drop": float(top_neuron_row["exact_ablation_margin_drop"]),
            }
        )
        mlp_layers.append(
            {
                "component": f"L{layer_index + 1} MLP",
                "block_ablation": {
                    "predicted_token": block_ablation_result["predicted_token"],
                    "margin": block_ablation_result["margin"],
                    "margin_drop": baseline_result["margin"] - block_ablation_result["margin"],
                },
                "top_neurons_by_margin_drop": _frame_to_records(
                    neuron_df.sort_values(
                        "exact_ablation_margin_drop",
                        ascending=False,
                    ).head(10)
                ),
                "top_neurons_by_standalone_margin": _frame_to_records(
                    neuron_df.sort_values(
                        "standalone_target_minus_foil",
                        ascending=False,
                    ).head(10)
                ),
                "neuron_table": _frame_to_records(neuron_df),
            }
        )

    payload = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "dataset_dir": str(args.dataset_dir),
            "analysis_source": row["analysis_source"],
            "analysis_split": row["analysis_split"],
            "analysis_row_index": row["analysis_row_index"],
        },
        "prompt_record": {
            key: row.get(key)
            for key in [
                "id",
                "task",
                "split",
                "num_pairs",
                "prompt",
                "target",
                "query_key",
                "context_pairs",
            ]
        },
        "baseline_result": baseline_result,
        "stage_readout_table": _frame_to_records(stage_readout_df),
        "head_summary_table": _frame_to_records(
            pd.DataFrame(head_summary_rows).sort_values(
                "exact_ablation_margin_drop",
                ascending=False,
            )
        ),
        "head_details": head_details,
        "mlp_block_summary_table": _frame_to_records(
            pd.DataFrame(mlp_block_rows).sort_values(
                "exact_block_ablation_margin_drop",
                ascending=False,
            )
        ),
        "mlp_layers": mlp_layers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"Saved prompt-program report to {args.out}")
    print()
    print("Baseline:")
    print(
        pd.DataFrame(
            [
                {
                    "predicted_token": baseline_result["predicted_token"],
                    "target_token": baseline_result["target_token"],
                    "foil_token": baseline_result["foil_token"],
                    "margin": baseline_result["margin"],
                    "correct": baseline_result["correct"],
                }
            ]
        ).to_string(index=False)
    )
    print()
    print("Top head ablations by exact margin drop:")
    print(
        pd.DataFrame(head_summary_rows)
        .sort_values("exact_ablation_margin_drop", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print()
    print("Top MLP blocks / neurons by exact margin drop:")
    print(
        pd.DataFrame(mlp_block_rows)
        .sort_values("exact_block_ablation_margin_drop", ascending=False)
        .to_string(index=False)
    )
    for layer_payload in mlp_layers:
        print()
        print(layer_payload["component"])
        print(
            pd.DataFrame(layer_payload["top_neurons_by_margin_drop"])
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
