from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scripts.kv_algorithm_causal_judge import (
    build_family_query_replacement_summary_table,
    build_family_query_replacement_table,
)
from scripts.kv_algorithm_faithfulness import build_variable_faithfulness_table
from scripts.kv_algorithm_feature_tracker import (
    build_feature_score_table,
    build_feature_site_summary_table,
    build_superposition_metrics,
)
from scripts.kv_algorithm_neuron_tracker import (
    build_layer_top_neuron_table,
    build_mlp_neuron_score_table,
)
from scripts.kv_algorithm_operator_finder import (
    build_head_attention_family_stability_table,
    build_head_attention_operator_summary_table,
    build_head_attention_operator_table,
    build_head_copy_rule_family_stability_table,
    build_head_copy_rule_summary_table,
    build_head_copy_rule_table,
)
from scripts.kv_algorithm_oracle import annotate_row
from scripts.kv_algorithm_record import (
    build_final_position_site_list,
    build_recording_summary_table,
    build_site_dataset,
    record_prompt_rows,
)
from scripts.kv_algorithm_sweeps import generate_controlled_sweeps
from scripts.kv_algorithm_variable_finder import (
    build_family_variable_stability_summary_table,
    build_family_variable_stability_table,
    build_site_variable_ranking_table,
    build_variable_recovery_table,
)
from scripts.kv_retrieve_analysis import (
    build_path_patched_attention_table,
    decode_token,
    load_dataset_bundle,
    score_path_patched_prompt,
    score_rows_with_optional_ablation,
)
from scripts.kv_retrieve_features import save_sae_checkpoint, train_sae
from scripts.tiny_transformer_core import TinyDecoderTransformer, load_tiny_decoder_checkpoint
from scripts.training_dynamics import (
    RunManifest,
    build_training_intervention_signature,
    evaluate_next_token_rows,
)


VARIABLE_NAMES = ["query_key", "matching_slot", "selected_value"]


def load_kv_bundle(manifest: RunManifest):
    bundle = load_dataset_bundle(Path(manifest.dataset.dataset_dir).expanduser().resolve())
    validate_kv_manifest_dataset(manifest, bundle)
    return bundle


def validate_kv_manifest_dataset(manifest: RunManifest, bundle: Any) -> None:
    required_splits = set(manifest.dataset.train_split_by_pairs.values()) | set(manifest.dataset.eval_splits.values())
    missing = sorted(required_splits - set(bundle.raw_splits))
    if missing:
        raise ValueError(f"KV dataset is missing required splits: {missing}")

    if manifest.dataset.sweep_base_split not in bundle.raw_splits:
        raise ValueError(
            f"KV dataset is missing the configured sweep base split {manifest.dataset.sweep_base_split!r}"
        )

    if manifest.training.curriculum == "on":
        split2 = bundle.raw_splits[manifest.dataset.train_split_by_pairs["2"]]
        split3 = bundle.raw_splits[manifest.dataset.train_split_by_pairs["3"]]
        if len(split2) != len(split3):
            raise ValueError(
                "Curriculum 'on' requires train 2-pair and train 3-pair splits with equal sizes"
            )

    max_prompt_length = max(
        len(str(row["prompt"]).split())
        for split_rows in bundle.raw_splits.values()
        for row in split_rows
    )
    if max_prompt_length > manifest.model.max_seq_len:
        raise ValueError(
            "Manifest model.max_seq_len is too small for the dataset: "
            f"required {max_prompt_length}, got {manifest.model.max_seq_len}"
        )


def build_kv_model_config(manifest: RunManifest, bundle: Any) -> dict[str, Any]:
    return {
        "vocab_size": len(bundle.vocab),
        "d_model": manifest.model.d_model,
        "n_heads": manifest.model.n_heads,
        "d_ff": manifest.model.d_ff,
        "n_layers": manifest.model.n_layers,
        "max_seq_len": manifest.model.max_seq_len,
    }


def instantiate_kv_model(manifest: RunManifest, bundle: Any, device: torch.device) -> TinyDecoderTransformer:
    return TinyDecoderTransformer(**build_kv_model_config(manifest, bundle)).to(device)


def _shuffle_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(rows)
    rng = torch.Generator()
    rng.manual_seed(seed)
    order = torch.randperm(len(shuffled), generator=rng).tolist()
    return [shuffled[index] for index in order]


def _interleave_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if len(left_rows) != len(right_rows):
        raise ValueError("Expected equal-length row lists for a 50/50 curriculum interleave")
    shuffled_left = _shuffle_rows(left_rows, seed)
    shuffled_right = _shuffle_rows(right_rows, seed + 1)
    combined: list[dict[str, Any]] = []
    for left_row, right_row in zip(shuffled_left, shuffled_right, strict=True):
        combined.append(left_row)
        combined.append(right_row)
    return combined


def select_kv_training_rows(manifest: RunManifest, bundle: Any, epoch: int) -> tuple[list[dict[str, Any]], str]:
    def _stage_label(rows: list[dict[str, Any]]) -> str:
        if not rows:
            raise ValueError("Expected non-empty training rows when deriving a curriculum stage label")
        num_pairs = rows[0].get("num_pairs")
        if not isinstance(num_pairs, int) or num_pairs <= 0:
            raise ValueError(f"Training row is missing a valid num_pairs field: {rows[0]!r}")
        return f"{num_pairs}_pairs_only"

    split2_name = manifest.dataset.train_split_by_pairs["2"]
    split3_name = manifest.dataset.train_split_by_pairs["3"]
    split2_rows = bundle.raw_splits[split2_name]
    split3_rows = bundle.raw_splits[split3_name]

    if manifest.training.curriculum == "off":
        return split3_rows, _stage_label(split3_rows)
    if 1 <= epoch <= 30:
        return split2_rows, _stage_label(split2_rows)
    if 31 <= epoch <= 60:
        return _interleave_rows(split2_rows, split3_rows, seed=manifest.training.seed + epoch), "mixed_2_and_3_pairs"
    if 61 <= epoch <= manifest.training.epochs:
        return split3_rows, _stage_label(split3_rows)
    raise ValueError(f"Epoch {epoch} is outside the configured training range 1..{manifest.training.epochs}")


def build_kv_controlled_sweeps(manifest: RunManifest, bundle: Any) -> list[dict[str, Any]]:
    base_rows = bundle.raw_splits[manifest.dataset.sweep_base_split][: manifest.battery.sweep_base_limit]
    if len(base_rows) < manifest.battery.sweep_base_limit:
        raise ValueError(
            f"Sweep base split {manifest.dataset.sweep_base_split!r} contains only {len(base_rows)} rows; "
            f"expected at least {manifest.battery.sweep_base_limit}"
        )

    ood_split = manifest.dataset.eval_splits.get("ood")
    if ood_split is None:
        raise ValueError("Manifest dataset.eval_splits must define an 'ood' split for longer-context sweeps")
    ood_rows = bundle.raw_splits[ood_split][: manifest.battery.sweep_base_limit]
    if len(ood_rows) < manifest.battery.sweep_base_limit:
        raise ValueError(
            f"OOD split {ood_split!r} contains only {len(ood_rows)} rows; "
            f"expected at least {manifest.battery.sweep_base_limit}"
        )
    return generate_controlled_sweeps(base_rows, longer_context_rows=ood_rows)


def _summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Expected at least one prediction row to summarize")
    margins = [float(row["margin"]) for row in rows]
    correct = [bool(row["correct"]) for row in rows]
    return {
        "rows": len(rows),
        "accuracy": sum(correct) / len(correct),
        "margin": sum(margins) / len(margins),
    }


def build_behavior_artifact(
    manifest: RunManifest,
    model: torch.nn.Module,
    bundle: Any,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_metrics: dict[str, Any] = {}
    for split_name in sorted(set(manifest.dataset.train_split_by_pairs.values()) | set(manifest.dataset.eval_splits.values())):
        split_metrics[split_name] = evaluate_next_token_rows(
            model,
            bundle.raw_splits[split_name],
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=manifest.battery.eval_batch_size,
        )

    sweep_rows = build_kv_controlled_sweeps(manifest, bundle)
    sweep_metrics = evaluate_next_token_rows(
        model,
        sweep_rows,
        token_to_id=bundle.token_to_id,
        id_to_token=bundle.id_to_token,
        device=device,
        batch_size=manifest.battery.eval_batch_size,
    )
    prediction_lookup = {row["prompt_id"]: row for row in sweep_metrics["predictions"]}
    family_breakdown: list[dict[str, Any]] = []
    for family_name in sorted({str(row["family_name"]) for row in sweep_rows}):
        family_prediction_rows = [
            prediction_lookup[str(row["id"])]
            for row in sweep_rows
            if str(row["family_name"]) == family_name
        ]
        family_breakdown.append(
            {
                "family_name": family_name,
                **_summarize_prediction_rows(family_prediction_rows),
            }
        )

    behavior_artifact = {
        "split_metrics": {
            split_name: {
                "rows": metrics["rows"],
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "margin": metrics["margin"],
            }
            for split_name, metrics in split_metrics.items()
        },
        "family_breakdown": family_breakdown,
    }
    return behavior_artifact, sweep_rows


def build_checkpoint_metrics(
    manifest: RunManifest,
    model: torch.nn.Module,
    bundle: Any,
    *,
    train_rows: list[dict[str, Any]],
    device: torch.device,
    train_batch_loss: float | None,
) -> dict[str, Any]:
    split_metrics = {
        "train_reference": evaluate_next_token_rows(
            model,
            train_rows,
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=manifest.battery.eval_batch_size,
        )
    }
    for split_name in manifest.dataset.eval_splits.values():
        split_metrics[split_name] = evaluate_next_token_rows(
            model,
            bundle.raw_splits[split_name],
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=manifest.battery.eval_batch_size,
        )

    val_split = manifest.dataset.eval_splits["val"]
    test_split = manifest.dataset.eval_splits["test"]
    ood_split = manifest.dataset.eval_splits["ood"]
    all_metrics = {
        "train_batch_loss": train_batch_loss,
        "train_loss": split_metrics["train_reference"]["loss"],
        "train_accuracy": split_metrics["train_reference"]["accuracy"],
        "train_margin": split_metrics["train_reference"]["margin"],
        "val_loss": split_metrics[val_split]["loss"],
        "val_accuracy": split_metrics[val_split]["accuracy"],
        "val_margin": split_metrics[val_split]["margin"],
        "test_loss": split_metrics[test_split]["loss"],
        "test_accuracy": split_metrics[test_split]["accuracy"],
        "test_margin": split_metrics[test_split]["margin"],
        "ood_loss": split_metrics[ood_split]["loss"],
        "ood_accuracy": split_metrics[ood_split]["accuracy"],
        "ood_margin": split_metrics[ood_split]["margin"],
    }
    all_metrics["all_checks_pass"] = all(
        isinstance(value, (int, float)) and not pd.isna(value)
        for key, value in all_metrics.items()
        if key != "train_batch_loss" or value is not None
    )
    return all_metrics


def _score_all_head_operators(
    model: torch.nn.Module,
    bundle: Any,
    sweep_recorded: list[Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    head_rows: list[dict[str, Any]] = []
    routing_candidate: dict[str, Any] | None = None
    copy_candidate: dict[str, Any] | None = None

    for layer_index, block in enumerate(model.blocks):
        for head_index in range(block.attn.n_heads):
            attention_table = build_head_attention_operator_table(
                sweep_recorded,
                layer_index=layer_index,
                head_index=head_index,
            )
            attention_summary = build_head_attention_operator_summary_table(
                attention_table,
                label=f"block{layer_index + 1}_head{head_index}_routing",
            ).iloc[0]
            attention_family = build_head_attention_family_stability_table(
                attention_table,
                label=f"block{layer_index + 1}_head{head_index}_routing",
            )
            copy_table = build_head_copy_rule_table(
                model,
                bundle,
                sweep_recorded,
                layer_index=layer_index,
                head_index=head_index,
            )
            copy_summary = build_head_copy_rule_summary_table(copy_table).iloc[0]
            copy_family = build_head_copy_rule_family_stability_table(copy_table)
            row = {
                "row_kind": "head_score",
                "layer_index": layer_index,
                "head_index": head_index,
                "head_name": f"block{layer_index + 1}_head{head_index}",
                "routing_score": float(attention_summary["top_total_slot_matches"]),
                "routing_key_score": float(attention_summary["top_key_slot_matches"]),
                "routing_value_score": float(attention_summary["top_value_slot_matches"]),
                "routing_family_min_score": float(attention_family["top_total_slot_matches"].min()),
                "copy_score": float(copy_summary["selected_value_top_written_rate"]),
                "copy_family_min_score": float(copy_family["selected_value_top_written_rate"].min()),
                "copy_rank_mean": float(copy_summary["selected_value_rank_mean"]),
            }
            head_rows.append(row)

            if routing_candidate is None or (
                row["routing_score"],
                row["routing_family_min_score"],
                -row["layer_index"],
                -row["head_index"],
            ) > (
                routing_candidate["routing_score"],
                routing_candidate["routing_family_min_score"],
                -routing_candidate["layer_index"],
                -routing_candidate["head_index"],
            ):
                routing_candidate = dict(row)

            if copy_candidate is None or (
                row["copy_score"],
                row["copy_family_min_score"],
                -row["layer_index"],
                -row["head_index"],
            ) > (
                copy_candidate["copy_score"],
                copy_candidate["copy_family_min_score"],
                -copy_candidate["layer_index"],
                -copy_candidate["head_index"],
            ):
                copy_candidate = dict(row)

    if routing_candidate is None or copy_candidate is None:
        raise ValueError("Operator scoring failed to produce routing and copy candidates")

    return pd.DataFrame(head_rows), routing_candidate, copy_candidate


def _build_variable_scores(
    manifest: RunManifest,
    model: torch.nn.Module,
    bundle: Any,
    sweep_rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_split_name = manifest.dataset.train_split_by_pairs["3"]
    train_rows = bundle.raw_splits[train_split_name][: manifest.battery.train_probe_limit]
    if len(train_rows) < manifest.battery.train_probe_limit:
        raise ValueError(
            f"Train probe split {train_split_name!r} contains only {len(train_rows)} rows; "
            f"expected at least {manifest.battery.train_probe_limit}"
        )

    train_recorded = record_prompt_rows(model, bundle, train_rows, device)
    sweep_recorded = record_prompt_rows(model, bundle, sweep_rows, device)
    combined_recorded = train_recorded + sweep_recorded
    candidate_sites = build_final_position_site_list(model)
    recorded_site_dataset = build_site_dataset(model, combined_recorded, candidate_sites)
    train_mask = (
        (recorded_site_dataset.metadata["source_kind"] == "dataset")
        & (recorded_site_dataset.metadata["split"] == train_split_name)
    )
    eval_mask = recorded_site_dataset.metadata["source_kind"] == "sweep"

    variable_recovery_table = build_variable_recovery_table(
        recorded_site_dataset,
        sites=candidate_sites,
        variables=VARIABLE_NAMES,
        train_mask=train_mask,
        eval_mask=eval_mask,
    )
    variable_recovery_rows = variable_recovery_table.copy()
    variable_recovery_rows["row_kind"] = "probe"

    ranking_table = build_site_variable_ranking_table(variable_recovery_table)
    summary_rows: list[dict[str, Any]] = []
    for variable in VARIABLE_NAMES:
        best_row = ranking_table.query("variable == @variable").iloc[0]
        family_table = build_family_variable_stability_table(
            recorded_site_dataset,
            site=str(best_row["site"]),
            variable=variable,
            train_mask=train_mask,
            eval_mask=eval_mask,
        )
        family_summary = build_family_variable_stability_summary_table(family_table).iloc[0]
        summary_rows.append(
            {
                "row_kind": "summary",
                "variable": variable,
                "best_site": str(best_row["site"]),
                "pooled_score": float(best_row["eval_accuracy"]),
                "family_min_score": float(family_table["family_accuracy"].min()),
                "family_mean_score": float(family_summary["family_accuracy_mean"]),
                "family_max_score": float(family_summary["family_accuracy_max"]),
            }
        )
    variable_scores = pd.concat(
        [
            variable_recovery_rows,
            pd.DataFrame(summary_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    return variable_scores, {
        "recording_summary": build_recording_summary_table(combined_recorded).to_dict(orient="records"),
        "candidate_sites": candidate_sites,
        "sweep_recorded": sweep_recorded,
        "sweep_site_dataset": build_site_dataset(model, sweep_recorded, candidate_sites),
    }


def _build_variable_faithfulness(
    model: torch.nn.Module,
    bundle: Any,
    variable_scores: pd.DataFrame,
    sweep_recorded: list[Any],
    *,
    device: torch.device,
) -> pd.DataFrame:
    variable_summary = variable_scores[variable_scores["row_kind"] == "summary"].copy()
    variable_best_sites = {
        str(row["variable"]): str(row["best_site"])
        for _, row in variable_summary.iterrows()
    }
    return build_variable_faithfulness_table(
        model,
        bundle,
        sweep_recorded,
        variable_best_sites,
        device=device,
    )


def _build_query_replacement_and_path_rows(
    model: torch.nn.Module,
    bundle: Any,
    sweep_recorded: list[Any],
    routing_candidate: dict[str, Any],
    copy_candidate: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if routing_candidate["layer_index"] + 1 != copy_candidate["layer_index"]:
        unsupported = pd.DataFrame(
            [
                {
                    "row_kind": "status",
                    "supported": False,
                    "reason": "Selected routing and copy candidates are not adjacent layers",
                    "routing_candidate": routing_candidate["head_name"],
                    "copy_candidate": copy_candidate["head_name"],
                }
            ]
        )
        return unsupported, unsupported.copy()

    query_sweep_recorded = [
        recorded
        for recorded in sweep_recorded
        if recorded.annotation.family_name == "query_key_sweep"
    ]
    if not query_sweep_recorded:
        raise ValueError("Expected at least one query_key_sweep recorded prompt for causal tests")

    source_patch = {
        "layer_index": int(routing_candidate["layer_index"]),
        "kind": "head_resid",
        "head_index": int(routing_candidate["head_index"]),
        "source_positions": [query_sweep_recorded[0].annotation.arrow_position],
    }
    family_tables: list[pd.DataFrame] = []
    path_rows: list[dict[str, Any]] = []
    for base_prompt_id in sorted({recorded.annotation.base_prompt_id for recorded in query_sweep_recorded}):
        family_prompt_group = [
            recorded
            for recorded in query_sweep_recorded
            if recorded.annotation.base_prompt_id == base_prompt_id
        ]
        family_tables.append(
            build_family_query_replacement_table(
                model,
                bundle,
                family_prompt_group,
                source_patch=source_patch,
                destination_layer_index=int(copy_candidate["layer_index"]),
                device=device,
                analysis_head_index=int(copy_candidate["head_index"]),
            )
        )
        base_recorded = family_prompt_group[0]
        for source_recorded in family_prompt_group[1:]:
            result = score_path_patched_prompt(
                model,
                bundle,
                clean_prompt=source_recorded.annotation.prompt,
                corrupt_prompt=base_recorded.annotation.prompt,
                clean_target=source_recorded.annotation.selected_value,
                device=device,
                source_patch=source_patch,
                destination={
                    "layer_index": int(copy_candidate["layer_index"]),
                    "head_index": int(copy_candidate["head_index"]),
                },
                clean_cache=source_recorded.cache,
                corrupt_cache=base_recorded.cache,
            )
            attention_table = build_path_patched_attention_table(
                model,
                base_recorded.annotation.prompt,
                source_recorded.cache,
                base_recorded.cache,
                source_patch=source_patch,
                destination={
                    "layer_index": int(copy_candidate["layer_index"]),
                    "head_index": int(copy_candidate["head_index"]),
                },
            )
            slot_attention = [
                float(attention_table.loc[attention_table["source_position"] == position, "attention_weight"].iloc[0])
                for position in base_recorded.annotation.value_positions
            ]
            predicted_slot = int(max(range(len(slot_attention)), key=lambda index: slot_attention[index]))
            path_rows.append(
                {
                    "row_kind": "path_patch_detail",
                    "base_prompt_id": base_prompt_id,
                    "base_query_key": base_recorded.annotation.query_key,
                    "source_query_key": source_recorded.annotation.query_key,
                    "expected_slot": source_recorded.annotation.matching_slot,
                    "predicted_slot": predicted_slot,
                    "slot_correct": predicted_slot == source_recorded.annotation.matching_slot,
                    "predicted_token": result["predicted_token"],
                    "expected_token": source_recorded.annotation.selected_value,
                    "correct": bool(result["correct"]),
                    "margin": float(result["margin"]),
                }
            )

    family_query_replacement_table = pd.concat(family_tables, ignore_index=True)
    family_query_replacement_summary = build_family_query_replacement_summary_table(
        family_query_replacement_table
    )
    replacement_rows = family_query_replacement_summary.copy()
    replacement_rows["row_kind"] = "query_replacement_summary"

    path_patch_table = pd.DataFrame(path_rows)
    path_patch_summary = pd.DataFrame(
        [
            {
                "row_kind": "path_patch_summary",
                "rows": int(len(path_patch_table)),
                "answer_switch_accuracy": float(path_patch_table["correct"].mean()),
                "slot_switch_accuracy": float(path_patch_table["slot_correct"].mean()),
                "margin_mean": float(path_patch_table["margin"].mean()),
            }
        ]
    )
    return replacement_rows, pd.concat([path_patch_table, path_patch_summary], ignore_index=True)


def _build_operator_scores(
    manifest: RunManifest,
    model: torch.nn.Module,
    bundle: Any,
    sweep_rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sweep_recorded = record_prompt_rows(model, bundle, sweep_rows, device)
    head_scores, routing_candidate, copy_candidate = _score_all_head_operators(model, bundle, sweep_recorded)
    candidate_rows = pd.DataFrame(
        [
            {
                "candidate_type": "routing",
                **routing_candidate,
                "row_kind": "candidate",
            },
            {
                "candidate_type": "copy",
                **copy_candidate,
                "row_kind": "candidate",
            },
        ]
    )
    replacement_rows, path_patch_rows = _build_query_replacement_and_path_rows(
        model,
        bundle,
        sweep_recorded,
        routing_candidate,
        copy_candidate,
        device=device,
    )
    operator_scores = pd.concat([head_scores, candidate_rows, replacement_rows], ignore_index=True, sort=False)
    return operator_scores, path_patch_rows, {
        "routing_candidate": routing_candidate,
        "copy_candidate": copy_candidate,
    }


def _build_localization_scores(
    model: torch.nn.Module,
    bundle: Any,
    sweep_rows: list[dict[str, Any]],
    path_patch_rows: pd.DataFrame,
    *,
    device: torch.device,
) -> pd.DataFrame:
    def score_mixed_length_rows(ablation: dict[str, Any] | None) -> pd.DataFrame:
        grouped_rows: dict[int, list[dict[str, Any]]] = {}
        for row in sweep_rows:
            grouped_rows.setdefault(len(str(row["prompt"]).split()), []).append(row)
        tables = [
            score_rows_with_optional_ablation(model, bundle, rows, device, ablation=ablation)
            for _, rows in sorted(grouped_rows.items())
        ]
        return pd.concat(tables, ignore_index=True)

    baseline_table = score_mixed_length_rows(ablation=None)
    baseline_accuracy = float(baseline_table["correct"].mean())
    baseline_margin = float(baseline_table["margin"].mean())

    ablation_rows: list[dict[str, Any]] = []
    for layer_index, block in enumerate(model.blocks):
        for head_index in range(block.attn.n_heads):
            ablated_table = score_mixed_length_rows(
                ablation={"layer_index": layer_index, "head_index": head_index},
            )
            ablation_rows.append(
                {
                    "row_kind": "head_ablation",
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "head_name": f"block{layer_index + 1}_head{head_index}",
                    "baseline_accuracy": baseline_accuracy,
                    "ablated_accuracy": float(ablated_table["correct"].mean()),
                    "accuracy_drop": baseline_accuracy - float(ablated_table["correct"].mean()),
                    "baseline_margin": baseline_margin,
                    "ablated_margin": float(ablated_table["margin"].mean()),
                    "margin_drop": baseline_margin - float(ablated_table["margin"].mean()),
                }
            )
    localization_rows = pd.DataFrame(ablation_rows)
    if path_patch_rows.empty:
        return localization_rows
    return pd.concat([localization_rows, path_patch_rows], ignore_index=True, sort=False)


def _matrix_metrics(name: str, matrix: torch.Tensor, *, layer_index: int, head_index: int | None = None) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(matrix.float()).detach().cpu()
    return {
        "name": name,
        "layer_index": layer_index,
        "head_index": head_index,
        "shape": list(matrix.shape),
        "fro_norm": float(matrix.float().norm().item()),
        "spectral_norm": float(singular_values[0].item()),
        "top_singular_values": [float(value) for value in singular_values[:5].tolist()],
    }


def build_weight_metrics(model: torch.nn.Module) -> dict[str, Any]:
    matrix_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for layer_index, block in enumerate(model.blocks):
        attn = block.attn
        matrices = {
            "q_proj": attn.q_proj.weight.detach().cpu(),
            "k_proj": attn.k_proj.weight.detach().cpu(),
            "v_proj": attn.v_proj.weight.detach().cpu(),
            "o_proj": attn.o_proj.weight.detach().cpu(),
            "gate_proj": block.mlp.gate_proj.weight.detach().cpu(),
            "up_proj": block.mlp.up_proj.weight.detach().cpu(),
            "down_proj": block.mlp.down_proj.weight.detach().cpu(),
        }
        for name, matrix in matrices.items():
            matrix_rows.append(_matrix_metrics(name, matrix, layer_index=layer_index))
        head_dim = attn.head_dim
        for head_index in range(attn.n_heads):
            row_start = head_index * head_dim
            row_stop = row_start + head_dim
            head_rows.append(
                _matrix_metrics(
                    "q_proj_head_slice",
                    attn.q_proj.weight.detach().cpu()[row_start:row_stop, :],
                    layer_index=layer_index,
                    head_index=head_index,
                )
            )
            head_rows.append(
                _matrix_metrics(
                    "k_proj_head_slice",
                    attn.k_proj.weight.detach().cpu()[row_start:row_stop, :],
                    layer_index=layer_index,
                    head_index=head_index,
                )
            )
            head_rows.append(
                _matrix_metrics(
                    "v_proj_head_slice",
                    attn.v_proj.weight.detach().cpu()[row_start:row_stop, :],
                    layer_index=layer_index,
                    head_index=head_index,
                )
            )
            head_rows.append(
                _matrix_metrics(
                    "o_proj_head_slice",
                    attn.o_proj.weight.detach().cpu()[:, row_start:row_stop],
                    layer_index=layer_index,
                    head_index=head_index,
                )
            )
    return {
        "matrices": matrix_rows,
        "head_slices": head_rows,
    }


def _build_neuron_scores(
    model: torch.nn.Module,
    bundle: Any,
    sweep_rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_recorded = record_prompt_rows(model, bundle, sweep_rows, device)
    neuron_scores = build_mlp_neuron_score_table(model, sweep_recorded)
    neuron_top = build_layer_top_neuron_table(neuron_scores, top_k=10)
    neuron_top = neuron_top.copy()
    neuron_top["row_kind"] = "layer_top"
    neuron_scores = neuron_scores.copy()
    neuron_scores["row_kind"] = "neuron"
    return neuron_scores, neuron_top


def _build_feature_tracking(
    manifest: RunManifest,
    model: torch.nn.Module,
    bundle: Any,
    sweep_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    battery_dir: Path,
    checkpoint_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not manifest.sae_tracking.enabled:
        return pd.DataFrame(), []

    available_sites = set(build_final_position_site_list(model))
    missing_sites = sorted(set(manifest.sae_tracking.sites) - available_sites)
    if missing_sites:
        raise ValueError(
            f"Manifest sae_tracking.sites includes unknown sites for the current model: {missing_sites}"
        )

    train_split_name = manifest.dataset.train_split_by_pairs["3"]
    val_split_name = manifest.dataset.eval_splits["val"]
    train_rows = bundle.raw_splits[train_split_name][: manifest.sae_tracking.train_limit]
    val_rows = bundle.raw_splits[val_split_name][: manifest.sae_tracking.val_limit]
    if len(train_rows) < manifest.sae_tracking.train_limit:
        raise ValueError(
            f"SAE train split {train_split_name!r} contains only {len(train_rows)} rows; "
            f"expected at least {manifest.sae_tracking.train_limit}"
        )
    if len(val_rows) < manifest.sae_tracking.val_limit:
        raise ValueError(
            f"SAE val split {val_split_name!r} contains only {len(val_rows)} rows; "
            f"expected at least {manifest.sae_tracking.val_limit}"
        )

    train_recorded = record_prompt_rows(model, bundle, train_rows, device)
    val_recorded = record_prompt_rows(model, bundle, val_rows, device)
    sweep_recorded = record_prompt_rows(model, bundle, sweep_rows, device)
    recorded_dataset = build_site_dataset(
        model,
        train_recorded + val_recorded + sweep_recorded,
        manifest.sae_tracking.sites,
    )

    train_mask = (
        (recorded_dataset.metadata["source_kind"] == "dataset")
        & (recorded_dataset.metadata["split"] == train_split_name)
    )
    val_mask = (
        (recorded_dataset.metadata["source_kind"] == "dataset")
        & (recorded_dataset.metadata["split"] == val_split_name)
    )
    sweep_mask = recorded_dataset.metadata["source_kind"] == "sweep"
    train_mask_tensor = torch.tensor(train_mask.to_list(), dtype=torch.bool)
    val_mask_tensor = torch.tensor(val_mask.to_list(), dtype=torch.bool)
    sweep_mask_tensor = torch.tensor(sweep_mask.to_list(), dtype=torch.bool)

    feature_tables: list[pd.DataFrame] = []
    superposition_rows: list[dict[str, Any]] = []
    sae_dir = battery_dir / "sae"
    sae_history_dir = battery_dir / "sae_history"
    sae_dir.mkdir(parents=True, exist_ok=True)
    sae_history_dir.mkdir(parents=True, exist_ok=True)

    for site in manifest.sae_tracking.sites:
        train_activations = recorded_dataset.site_vectors[site][train_mask_tensor].float()
        val_activations = recorded_dataset.site_vectors[site][val_mask_tensor].float()
        sweep_activations = recorded_dataset.site_vectors[site][sweep_mask_tensor].float()
        hidden_dim = int(train_activations.shape[1] * manifest.sae_tracking.hidden_multiplier)
        sae, history_table = train_sae(
            train_activations=train_activations,
            val_activations=val_activations,
            hidden_dim=hidden_dim,
            l1_coeff=manifest.sae_tracking.l1_coeff,
            learning_rate=manifest.sae_tracking.learning_rate,
            batch_size=manifest.sae_tracking.batch_size,
            epochs=manifest.sae_tracking.epochs,
            seed=manifest.sae_tracking.seed,
        )
        save_sae_checkpoint(
            sae_dir / f"{site}.pt",
            sae,
            metadata={
                "checkpoint_id": checkpoint_id,
                "site": site,
                "train_split": train_split_name,
                "val_split": val_split_name,
                "train_rows": int(train_activations.shape[0]),
                "val_rows": int(val_activations.shape[0]),
                "sweep_rows": int(sweep_activations.shape[0]),
            },
        )
        history_table.to_csv(sae_history_dir / f"{site}.csv", index=False)

        feature_table = build_feature_score_table(
            recorded_dataset,
            site,
            sae,
            eval_mask=sweep_mask,
        )
        site_summary = build_feature_site_summary_table(
            feature_table,
            top_features_per_site=manifest.sae_tracking.top_features_per_site,
        )
        feature_tables.extend([feature_table, site_summary])
        superposition_rows.append(
            build_superposition_metrics(
                site,
                sae,
                sweep_activations,
                feature_table,
                history_table,
                cosine_threshold=manifest.sae_tracking.superposition_cosine_threshold,
            )
        )

    return pd.concat(feature_tables, ignore_index=True, sort=False), superposition_rows


def run_kv_checkpoint_battery(
    *,
    manifest: RunManifest,
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    announce: bool = False,
) -> dict[str, Any]:
    bundle = load_kv_bundle(manifest)
    checkpoint_payload, model = load_tiny_decoder_checkpoint(checkpoint_path, device)
    checkpoint_id = checkpoint_path.stem
    battery_dir = run_dir / "battery" / checkpoint_id
    battery_dir.mkdir(parents=True, exist_ok=True)
    if announce:
        print(f"[battery:{checkpoint_id}] behavior", flush=True)
    behavior_artifact, sweep_rows = build_behavior_artifact(manifest, model, bundle, device=device)
    if announce:
        print(f"[battery:{checkpoint_id}] variables", flush=True)
    variable_scores, variable_artifacts = _build_variable_scores(
        manifest,
        model,
        bundle,
        sweep_rows,
        device=device,
    )
    if announce:
        print(f"[battery:{checkpoint_id}] faithfulness", flush=True)
    variable_faithfulness = _build_variable_faithfulness(
        model,
        bundle,
        variable_scores,
        variable_artifacts["sweep_recorded"],
        device=device,
    )
    if announce:
        print(f"[battery:{checkpoint_id}] operators", flush=True)
    operator_scores, path_patch_rows, operator_artifacts = _build_operator_scores(
        manifest,
        model,
        bundle,
        sweep_rows,
        device=device,
    )
    if announce:
        print(f"[battery:{checkpoint_id}] localization", flush=True)
    localization_scores = _build_localization_scores(
        model,
        bundle,
        sweep_rows,
        path_patch_rows,
        device=device,
    )
    if announce:
        print(f"[battery:{checkpoint_id}] weights", flush=True)
    weight_metrics = build_weight_metrics(model)
    if announce:
        print(f"[battery:{checkpoint_id}] neurons", flush=True)
    neuron_scores, neuron_top = _build_neuron_scores(
        model,
        bundle,
        sweep_rows,
        device=device,
    )
    if announce:
        print(f"[battery:{checkpoint_id}] features", flush=True)
    feature_scores, superposition_metrics = _build_feature_tracking(
        manifest,
        model,
        bundle,
        sweep_rows,
        device=device,
        battery_dir=battery_dir,
        checkpoint_id=checkpoint_id,
    )
    if announce:
        print(f"[battery:{checkpoint_id}] write_artifacts", flush=True)
    (battery_dir / "behavior.json").write_text(json.dumps(behavior_artifact, indent=2), encoding="utf-8")
    variable_scores.to_csv(battery_dir / "variable_scores.csv", index=False)
    variable_faithfulness.to_csv(battery_dir / "variable_faithfulness.csv", index=False)
    operator_scores.to_csv(battery_dir / "operator_scores.csv", index=False)
    localization_scores.to_csv(battery_dir / "localization.csv", index=False)
    (battery_dir / "weight_metrics.json").write_text(json.dumps(weight_metrics, indent=2), encoding="utf-8")
    pd.concat([neuron_scores, neuron_top], ignore_index=True, sort=False).to_csv(
        battery_dir / "neuron_scores.csv",
        index=False,
    )
    if manifest.sae_tracking.enabled:
        feature_scores.to_csv(battery_dir / "feature_scores.csv", index=False)
        (battery_dir / "superposition_metrics.json").write_text(
            json.dumps(superposition_metrics, indent=2),
            encoding="utf-8",
        )
    torch.save(
        {
            "checkpoint_id": checkpoint_id,
            "candidate_sites": variable_artifacts["candidate_sites"],
            "metadata": variable_artifacts["sweep_site_dataset"].metadata.to_dict(orient="records"),
            "site_vectors": variable_artifacts["sweep_site_dataset"].site_vectors,
        },
        battery_dir / "canonical_site_vectors.pt",
    )
    torch.save(
        {
            "checkpoint_id": checkpoint_id,
            "checkpoint_payload_summary": {
                "epoch": checkpoint_payload.get("epoch"),
                "save_reason": checkpoint_payload.get("save_reason"),
            },
            "recording_summary": variable_artifacts["recording_summary"],
            "candidate_sites": variable_artifacts["candidate_sites"],
            "routing_candidate": operator_artifacts["routing_candidate"],
            "copy_candidate": operator_artifacts["copy_candidate"],
            "sae_sites": list(manifest.sae_tracking.sites) if manifest.sae_tracking.enabled else [],
        },
        battery_dir / "tensors.pt",
    )
    if announce:
        print(f"[battery:{checkpoint_id}] done", flush=True)
    return {
        "checkpoint_id": checkpoint_id,
        "behavior": behavior_artifact,
        "variable_scores_path": str(battery_dir / "variable_scores.csv"),
        "variable_faithfulness_path": str(battery_dir / "variable_faithfulness.csv"),
        "operator_scores_path": str(battery_dir / "operator_scores.csv"),
        "localization_path": str(battery_dir / "localization.csv"),
        "weight_metrics_path": str(battery_dir / "weight_metrics.json"),
        "neuron_scores_path": str(battery_dir / "neuron_scores.csv"),
        "feature_scores_path": str(battery_dir / "feature_scores.csv") if manifest.sae_tracking.enabled else None,
        "superposition_metrics_path": (
            str(battery_dir / "superposition_metrics.json") if manifest.sae_tracking.enabled else None
        ),
    }


def load_behavior_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing behavior artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing score artifact: {path}")
    return pd.read_csv(path)


def collect_run_checkpoint_rows(run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        battery_dir = run_dir / "battery" / checkpoint_id
        behavior = load_behavior_artifact(battery_dir / "behavior.json")
        variable_scores = load_scores_csv(battery_dir / "variable_scores.csv")
        variable_faithfulness = load_scores_csv(battery_dir / "variable_faithfulness.csv")
        operator_scores = load_scores_csv(battery_dir / "operator_scores.csv")

        variable_summary = variable_scores[variable_scores["row_kind"] == "summary"].copy()
        faithfulness_summary = variable_faithfulness[variable_faithfulness["row_kind"] == "summary"].copy()
        operator_candidates = operator_scores[operator_scores["row_kind"] == "candidate"].copy()
        row = {
            "run_dir": str(run_dir.resolve()),
            "run_id": str(payload.get("run_id") or run_dir.name),
            "seed": int(payload.get("seed", manifest.training.seed)),
            "intervention_signature": build_training_intervention_signature(manifest.training_interventions),
            "intervention_count": int(len(manifest.training_interventions)),
            "checkpoint_id": checkpoint_id,
            "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
            "save_reason": str(payload.get("save_reason")),
            "val_accuracy": float(behavior["split_metrics"][manifest.dataset.eval_splits["val"]]["accuracy"]),
            "test_accuracy": float(behavior["split_metrics"][manifest.dataset.eval_splits["test"]]["accuracy"]),
            "ood_accuracy": float(behavior["split_metrics"][manifest.dataset.eval_splits["ood"]]["accuracy"]),
        }
        for variable in VARIABLE_NAMES:
            variable_row = variable_summary[variable_summary["variable"] == variable].iloc[0]
            faithfulness_row = faithfulness_summary[faithfulness_summary["variable"] == variable].iloc[0]
            row[f"{variable}_site"] = str(variable_row["best_site"])
            row[f"{variable}_score"] = float(variable_row["pooled_score"])
            row[f"{variable}_family_min_score"] = float(variable_row["family_min_score"])
            row[f"{variable}_faithfulness_site"] = str(faithfulness_row["site"])
            row[f"{variable}_faithfulness_score"] = float(faithfulness_row["pooled_score"])
            row[f"{variable}_faithfulness_family_min_score"] = float(faithfulness_row["family_min_score"])
        routing_row = operator_candidates[operator_candidates["candidate_type"] == "routing"].iloc[0]
        copy_row = operator_candidates[operator_candidates["candidate_type"] == "copy"].iloc[0]
        row.update(
            {
                "routing_candidate": str(routing_row["head_name"]),
                "routing_score": float(routing_row["routing_score"]),
                "routing_family_min_score": float(routing_row["routing_family_min_score"]),
                "copy_candidate": str(copy_row["head_name"]),
                "copy_score": float(copy_row["copy_score"]),
                "copy_family_min_score": float(copy_row["copy_family_min_score"]),
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    return rows


def collect_run_feature_rows(run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
    if not manifest.sae_tracking.enabled:
        return []
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        feature_path = run_dir / "battery" / checkpoint_id / "feature_scores.csv"
        feature_scores = pd.read_csv(feature_path)
        summary_rows = feature_scores[feature_scores["row_kind"] == "site_summary"]
        for _, row in summary_rows.iterrows():
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_id": str(payload.get("run_id") or run_dir.name),
                    "checkpoint_id": checkpoint_id,
                    "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                    "site": str(row["site"]),
                    "feature_count": int(row["feature_count"]),
                    "active_feature_count": int(row["active_feature_count"]),
                    "top_feature_index": int(row["top_feature_index"]),
                    "top_feature_variable": str(row["top_feature_variable"]),
                    "top_feature_selectivity_score": float(row["top_feature_selectivity_score"]),
                    "mean_top_feature_selectivity_score": float(row["mean_top_feature_selectivity_score"]),
                    "mean_top_feature_activation_rate": float(row["mean_top_feature_activation_rate"]),
                    "mean_decoder_norm": float(row["mean_decoder_norm"]),
                }
            )
    return rows


def collect_run_superposition_rows(run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
    if not manifest.sae_tracking.enabled:
        return []
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        superposition_path = run_dir / "battery" / checkpoint_id / "superposition_metrics.json"
        superposition_rows = json.loads(superposition_path.read_text(encoding="utf-8"))
        if not isinstance(superposition_rows, list):
            raise ValueError(
                f"Expected a list of superposition metrics in {superposition_path}, "
                f"got {type(superposition_rows).__name__}"
            )
        for row in superposition_rows:
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected each superposition metric row to be an object in {superposition_path}"
                )
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_id": str(payload.get("run_id") or run_dir.name),
                    "checkpoint_id": checkpoint_id,
                    "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                    **row,
                }
            )
    return rows


def _linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            f"Linear CKA expects rank-2 tensors, got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"Linear CKA expects equal row counts, got {left.shape[0]} and {right.shape[0]}"
        )
    left_centered = left.float() - left.float().mean(dim=0, keepdim=True)
    right_centered = right.float() - right.float().mean(dim=0, keepdim=True)
    left_gram = left_centered.T @ left_centered
    right_gram = right_centered.T @ right_centered
    cross = left_centered.T @ right_centered
    numerator = float((cross.pow(2).sum()).item())
    denominator = float(
        torch.sqrt((left_gram.pow(2).sum()) * (right_gram.pow(2).sum())).item()
    )
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _safe_vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.reshape(-1).float()
    right_flat = right.reshape(-1).float()
    left_norm = float(left_flat.norm().item())
    right_norm = float(right_flat.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).item() / (left_norm * right_norm))


def collect_run_representation_drift_rows(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    checkpoint_records: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        canonical_path = run_dir / "battery" / checkpoint_id / "canonical_site_vectors.pt"
        canonical_payload = torch.load(canonical_path, map_location="cpu")
        metadata = canonical_payload["metadata"]
        if not isinstance(metadata, list):
            raise ValueError(f"Canonical site metadata must be a list in {canonical_path}")
        prompt_ids = [str(row["prompt_id"]) for row in metadata]
        checkpoint_records.append(
            {
                "run_id": str(payload.get("run_id") or run_dir.name),
                "checkpoint_id": checkpoint_id,
                "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                "prompt_ids": prompt_ids,
                "site_vectors": canonical_payload["site_vectors"],
            }
        )
    if not checkpoint_records:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    rows: list[dict[str, Any]] = []
    for left_index, left_record in enumerate(checkpoint_records):
        for right_index, right_record in enumerate(checkpoint_records):
            if right_index <= left_index:
                continue
            if left_record["prompt_ids"] != right_record["prompt_ids"]:
                raise ValueError(
                    "Canonical drift packs must share the same prompt_id ordering across checkpoints"
                )
            shared_sites = sorted(
                set(left_record["site_vectors"].keys()) & set(right_record["site_vectors"].keys())
            )
            if not shared_sites:
                raise ValueError("Canonical drift packs share no common sites across checkpoints")
            for site in shared_sites:
                left_vectors = left_record["site_vectors"][site].float()
                right_vectors = right_record["site_vectors"][site].float()
                mean_left = left_vectors.mean(dim=0)
                mean_right = right_vectors.mean(dim=0)
                rows.append(
                    {
                        "run_id": left_record["run_id"],
                        "site": site,
                        "checkpoint_id_left": left_record["checkpoint_id"],
                        "epoch_left": left_record["epoch"],
                        "checkpoint_id_right": right_record["checkpoint_id"],
                        "epoch_right": right_record["epoch"],
                        "epoch_gap": int(right_record["epoch"] - left_record["epoch"]),
                        "is_adjacent_pair": right_index == left_index + 1,
                        "is_origin_pair": left_index == 0,
                        "is_final_pair": right_index == len(checkpoint_records) - 1,
                        "linear_cka": _linear_cka(left_vectors, right_vectors),
                        "mean_vector_cosine": _safe_vector_cosine(mean_left, mean_right),
                        "relative_frobenius_shift": float(
                            (right_vectors - left_vectors).norm().item() / left_vectors.norm().clamp_min(1e-8).item()
                        ),
                    }
                )
    return rows


def _rank_operator_heads(
    operator_scores: pd.DataFrame,
    role_name: str,
    *,
    top_k: int,
) -> dict[str, Any]:
    head_scores = operator_scores[operator_scores["row_kind"] == "head_score"].copy()
    if head_scores.empty:
        raise ValueError("Expected operator_scores to contain head_score rows")
    role_score_column = f"{role_name}_score"
    family_column = f"{role_name}_family_min_score"
    if role_score_column not in head_scores.columns or family_column not in head_scores.columns:
        raise ValueError(
            f"Operator scores are missing required columns {role_score_column!r} and/or {family_column!r}"
        )
    ordered = head_scores.sort_values(
        [role_score_column, family_column, "layer_index", "head_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    head_names = [str(name) for name in ordered["head_name"].tolist()]
    if not head_names:
        raise ValueError(f"No heads available to rank for role {role_name!r}")
    candidate_row = operator_scores[
        (operator_scores["row_kind"] == "candidate")
        & (operator_scores["candidate_type"] == role_name)
    ]
    if candidate_row.empty:
        raise ValueError(f"Operator scores are missing the candidate row for role {role_name!r}")
    candidate_name = str(candidate_row.iloc[0]["head_name"])
    if candidate_name not in head_names:
        raise ValueError(
            f"Candidate head {candidate_name!r} is not present in the ranked head scores for {role_name!r}"
        )
    top_scores = ordered[role_score_column].astype(float).tolist()
    leader_gap = float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float(top_scores[0])
    total_score = float(sum(top_scores))
    return {
        "candidate_name": candidate_name,
        "candidate_rank": int(head_names.index(candidate_name)) + 1,
        "top_heads": head_names[:top_k],
        "leader_gap": leader_gap,
        "top_score": float(top_scores[0]),
        "top_k_score_share": (
            float(sum(top_scores[:top_k]) / total_score) if total_score > 0.0 else float("nan")
        ),
        "score_map": {
            str(head_name): float(score)
            for head_name, score in zip(head_names, top_scores, strict=True)
        },
        "ranked_heads": head_names,
    }


def collect_run_operator_handoff_rows(run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    checkpoint_entries: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        operator_scores = pd.read_csv(run_dir / "battery" / checkpoint_id / "operator_scores.csv")
        checkpoint_entries.append(
            {
                "run_id": str(payload.get("run_id") or run_dir.name),
                "seed": int(payload.get("seed", manifest.training.seed)),
                "intervention_signature": build_training_intervention_signature(manifest.training_interventions),
                "checkpoint_id": checkpoint_id,
                "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                "routing": _rank_operator_heads(
                    operator_scores,
                    "routing",
                    top_k=manifest.battery.role_top_k,
                ),
                "copy": _rank_operator_heads(
                    operator_scores,
                    "copy",
                    top_k=manifest.battery.role_top_k,
                ),
            }
        )
    if not checkpoint_entries:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    rows: list[dict[str, Any]] = []
    for previous_entry, current_entry in zip(checkpoint_entries, checkpoint_entries[1:]):
        for role_name in ["routing", "copy"]:
            previous_role = previous_entry[role_name]
            current_role = current_entry[role_name]
            previous_top = list(previous_role["top_heads"])
            current_top = list(current_role["top_heads"])
            overlap = sorted(set(previous_top) & set(current_top))
            rows.append(
                {
                    "run_id": current_entry["run_id"],
                    "seed": current_entry["seed"],
                    "intervention_signature": current_entry["intervention_signature"],
                    "role_name": role_name,
                    "checkpoint_id_previous": previous_entry["checkpoint_id"],
                    "epoch_previous": previous_entry["epoch"],
                    "checkpoint_id_current": current_entry["checkpoint_id"],
                    "epoch_current": current_entry["epoch"],
                    "previous_candidate": str(previous_role["candidate_name"]),
                    "current_candidate": str(current_role["candidate_name"]),
                    "candidate_changed": previous_role["candidate_name"] != current_role["candidate_name"],
                    "top_k_overlap_count": int(len(overlap)),
                    "top_k_overlap_fraction": float(
                        len(overlap) / manifest.battery.role_top_k
                    ),
                    "previous_top_heads": ",".join(previous_top),
                    "current_top_heads": ",".join(current_top),
                    "previous_leader_gap": float(previous_role["leader_gap"]),
                    "current_leader_gap": float(current_role["leader_gap"]),
                    "previous_top_k_score_share": float(previous_role["top_k_score_share"]),
                    "current_top_k_score_share": float(current_role["top_k_score_share"]),
                    "previous_candidate_rank_current": int(
                        current_role["ranked_heads"].index(previous_role["candidate_name"]) + 1
                    ),
                    "current_candidate_rank_previous": int(
                        previous_role["ranked_heads"].index(current_role["candidate_name"]) + 1
                    ),
                }
            )
    return rows


def summarize_cross_seed_role_matching(
    run_dirs: list[Path],
    manifests: list[RunManifest],
) -> pd.DataFrame:
    final_entries: list[dict[str, Any]] = []
    for run_dir, manifest in zip(run_dirs, manifests, strict=True):
        checkpoints = collect_run_checkpoint_rows(run_dir, manifest)
        final_checkpoint = (
            pd.DataFrame(checkpoints)
            .sort_values(["epoch", "checkpoint_id"])
            .iloc[-1]
        )
        checkpoint_id = str(final_checkpoint["checkpoint_id"])
        payload = torch.load(run_dir / "checkpoints" / f"{checkpoint_id}.pt", map_location="cpu")
        operator_scores = pd.read_csv(run_dir / "battery" / checkpoint_id / "operator_scores.csv")
        final_entries.append(
            {
                "run_id": str(final_checkpoint["run_id"]),
                "seed": int(final_checkpoint["seed"]),
                "intervention_signature": str(final_checkpoint["intervention_signature"]),
                "routing": _rank_operator_heads(
                    operator_scores,
                    "routing",
                    top_k=manifest.battery.role_top_k,
                ),
                "copy": _rank_operator_heads(
                    operator_scores,
                    "copy",
                    top_k=manifest.battery.role_top_k,
                ),
                "payload_seed": int(payload.get("seed", manifest.training.seed)),
            }
        )

    rows: list[dict[str, Any]] = []
    for left_entry, right_entry in combinations(final_entries, 2):
        if left_entry["intervention_signature"] != right_entry["intervention_signature"]:
            continue
        for role_name in ["routing", "copy"]:
            left_role = left_entry[role_name]
            right_role = right_entry[role_name]
            left_heads = left_role["ranked_heads"]
            right_heads = right_role["ranked_heads"]
            score_vector_left = torch.tensor(
                [left_role["score_map"][head_name] for head_name in left_heads],
                dtype=torch.float32,
            )
            score_vector_right = torch.tensor(
                [right_role["score_map"][head_name] for head_name in left_heads],
                dtype=torch.float32,
            )
            top_overlap = sorted(set(left_role["top_heads"]) & set(right_role["top_heads"]))
            rows.append(
                {
                    "intervention_signature": left_entry["intervention_signature"],
                    "role_name": role_name,
                    "run_id_left": left_entry["run_id"],
                    "seed_left": left_entry["seed"],
                    "run_id_right": right_entry["run_id"],
                    "seed_right": right_entry["seed"],
                    "candidate_left": str(left_role["candidate_name"]),
                    "candidate_right": str(right_role["candidate_name"]),
                    "same_candidate_identity": left_role["candidate_name"] == right_role["candidate_name"],
                    "top_k_overlap_count": int(len(top_overlap)),
                    "top_k_overlap_fraction": float(
                        len(top_overlap) / len(left_role["top_heads"])
                    ),
                    "candidate_left_rank_in_right": int(
                        right_heads.index(left_role["candidate_name"]) + 1
                    ),
                    "candidate_right_rank_in_left": int(
                        left_heads.index(right_role["candidate_name"]) + 1
                    ),
                    "candidate_left_score": float(left_role["top_score"]),
                    "candidate_right_score": float(right_role["top_score"]),
                    "score_profile_cosine": _safe_vector_cosine(score_vector_left, score_vector_right),
                }
            )
    return pd.DataFrame(rows)


def summarize_clamp_responsiveness(
    checkpoint_table: pd.DataFrame,
    emergence_table: pd.DataFrame,
) -> pd.DataFrame:
    if checkpoint_table.empty:
        raise ValueError("Expected a non-empty checkpoint table for clamp responsiveness")
    final_rows = checkpoint_table.sort_values(["epoch", "checkpoint_id"]).groupby("run_id", as_index=False).tail(1)
    baseline_rows = final_rows[final_rows["intervention_signature"] == "none"].copy()
    baseline_by_seed: dict[int, pd.Series] = {}
    for _, row in baseline_rows.iterrows():
        seed = int(row["seed"])
        if seed in baseline_by_seed:
            raise ValueError(f"Expected at most one baseline run for seed {seed}, found multiple")
        baseline_by_seed[seed] = row

    emergence_pivot = emergence_table.pivot(index="run_id", columns="metric_name", values="birth_epoch")
    tracked_birth_metrics = [
        "behavior_val_accuracy",
        "variable_query_key",
        "variable_matching_slot",
        "variable_selected_value",
        "faithfulness_query_key",
        "faithfulness_matching_slot",
        "faithfulness_selected_value",
        "operator_routing",
        "operator_copy",
    ]
    rows: list[dict[str, Any]] = []
    intervention_rows = final_rows[final_rows["intervention_signature"] != "none"].copy()
    for _, row in intervention_rows.iterrows():
        seed = int(row["seed"])
        baseline_row = baseline_by_seed.get(seed)
        output_row: dict[str, Any] = {
            "run_id": str(row["run_id"]),
            "seed": seed,
            "intervention_signature": str(row["intervention_signature"]),
            "intervention_count": int(row["intervention_count"]),
            "matched_baseline_found": baseline_row is not None,
            "matched_baseline_run_id": None if baseline_row is None else str(baseline_row["run_id"]),
        }
        for metric_name in ["val_accuracy", "test_accuracy", "ood_accuracy"]:
            output_row[f"final_{metric_name}"] = float(row[metric_name])
            output_row[f"baseline_{metric_name}"] = (
                None if baseline_row is None else float(baseline_row[metric_name])
            )
            output_row[f"{metric_name}_delta_vs_baseline"] = (
                None if baseline_row is None else float(row[metric_name] - baseline_row[metric_name])
            )
        for metric_name in tracked_birth_metrics:
            run_birth = emergence_pivot.loc[str(row["run_id"]), metric_name] if metric_name in emergence_pivot.columns else None
            baseline_birth = (
                None
                if baseline_row is None or metric_name not in emergence_pivot.columns
                else emergence_pivot.loc[str(baseline_row["run_id"]), metric_name]
            )
            output_row[f"{metric_name}_birth_epoch"] = None if pd.isna(run_birth) else int(run_birth)
            output_row[f"{metric_name}_baseline_birth_epoch"] = (
                None if baseline_birth is None or pd.isna(baseline_birth) else int(baseline_birth)
            )
            output_row[f"{metric_name}_birth_shift_vs_baseline"] = (
                None
                if baseline_birth is None or pd.isna(run_birth) or pd.isna(baseline_birth)
                else int(run_birth - baseline_birth)
            )
        rows.append(output_row)
    return pd.DataFrame(rows)


def summarize_emergence(
    checkpoint_table: pd.DataFrame,
    manifest: RunManifest,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, group in checkpoint_table.groupby("run_id"):
        ordered = group.sort_values("epoch").reset_index(drop=True)
        behavior_birth = ordered[ordered["val_accuracy"] >= manifest.summary_thresholds.behavior_birth_val_accuracy]
        rows.append(
            {
                "run_id": run_id,
                "metric_name": "behavior_val_accuracy",
                "birth_epoch": int(behavior_birth.iloc[0]["epoch"]) if not behavior_birth.empty else None,
                "birth_checkpoint_id": str(behavior_birth.iloc[0]["checkpoint_id"]) if not behavior_birth.empty else None,
            }
        )
        for variable in VARIABLE_NAMES:
            variable_birth = ordered[
                (ordered[f"{variable}_score"] >= manifest.summary_thresholds.variable_birth_score)
                & (
                    ordered[f"{variable}_family_min_score"]
                    >= manifest.summary_thresholds.variable_family_min_score
                )
            ]
            rows.append(
                {
                    "run_id": run_id,
                    "metric_name": f"variable_{variable}",
                    "birth_epoch": int(variable_birth.iloc[0]["epoch"]) if not variable_birth.empty else None,
                    "birth_checkpoint_id": str(variable_birth.iloc[0]["checkpoint_id"]) if not variable_birth.empty else None,
                }
            )
            faithfulness_birth = ordered[
                (ordered[f"{variable}_faithfulness_score"] >= manifest.summary_thresholds.faithfulness_birth_score)
                & (
                    ordered[f"{variable}_faithfulness_family_min_score"]
                    >= manifest.summary_thresholds.faithfulness_family_min_score
                )
            ]
            rows.append(
                {
                    "run_id": run_id,
                    "metric_name": f"faithfulness_{variable}",
                    "birth_epoch": int(faithfulness_birth.iloc[0]["epoch"]) if not faithfulness_birth.empty else None,
                    "birth_checkpoint_id": (
                        str(faithfulness_birth.iloc[0]["checkpoint_id"])
                        if not faithfulness_birth.empty
                        else None
                    ),
                }
            )
        for operator_name in ["routing", "copy"]:
            operator_birth = ordered[
                (ordered[f"{operator_name}_score"] >= manifest.summary_thresholds.operator_birth_score)
                & (
                    ordered[f"{operator_name}_family_min_score"]
                    >= manifest.summary_thresholds.operator_family_min_score
                )
            ]
            rows.append(
                {
                    "run_id": run_id,
                    "metric_name": f"operator_{operator_name}",
                    "birth_epoch": int(operator_birth.iloc[0]["epoch"]) if not operator_birth.empty else None,
                    "birth_checkpoint_id": str(operator_birth.iloc[0]["checkpoint_id"]) if not operator_birth.empty else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_seed_stability(checkpoint_table: pd.DataFrame) -> pd.DataFrame:
    if checkpoint_table.empty:
        raise ValueError("Expected a non-empty checkpoint table for seed-stability summarization")
    final_rows = checkpoint_table.sort_values("epoch").groupby("run_id", as_index=False).tail(1)
    summary_rows: list[dict[str, Any]] = []
    for metric_name in [
        "val_accuracy",
        "test_accuracy",
        "ood_accuracy",
        "query_key_score",
        "matching_slot_score",
        "selected_value_score",
        "query_key_faithfulness_score",
        "matching_slot_faithfulness_score",
        "selected_value_faithfulness_score",
        "routing_score",
        "copy_score",
    ]:
        metric_series = final_rows[metric_name].astype(float)
        summary_rows.append(
            {
                "metric_name": metric_name,
                "num_runs": int(len(metric_series)),
                "mean": float(metric_series.mean()),
                "std": float(metric_series.std(ddof=0)),
                "min": float(metric_series.min()),
                "max": float(metric_series.max()),
            }
        )
    return pd.DataFrame(summary_rows)


def collect_run_neuron_rows(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        neuron_path = run_dir / "battery" / checkpoint_id / "neuron_scores.csv"
        neuron_scores = pd.read_csv(neuron_path)
        top_rows = neuron_scores[neuron_scores["row_kind"] == "layer_top"]
        for _, row in top_rows.iterrows():
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_id": str(payload.get("run_id") or run_dir.name),
                    "checkpoint_id": checkpoint_id,
                    "epoch": int(payload.get("epoch", payload.get("selected_epoch"))),
                    "layer_index": int(row["layer_index"]),
                    "neuron_index": int(row["neuron_index"]),
                    "best_variable": str(row["best_variable"]),
                    "best_selectivity_score": float(row["best_selectivity_score"]),
                    "activation_write_product": float(row["activation_write_product"]),
                    "positive_rate": float(row["positive_rate"]),
                    "nonzero_rate": float(row["nonzero_rate"]),
                }
            )
    return rows


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
