#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.kv_benchmark import (
    collect_run_checkpoint_rows,
    collect_run_feature_rows,
    collect_run_neuron_rows,
    collect_run_operator_handoff_rows,
    collect_run_representation_drift_rows,
    collect_run_superposition_rows,
    discover_run_directories,
    summarize_clamp_responsiveness,
    summarize_cross_seed_role_matching,
    summarize_emergence,
    summarize_seed_stability,
)
from scripts.training_dynamics import load_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize checkpoint-level KV training dynamics for one run or a directory of runs."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Either a single run directory or a parent directory that contains run directories.",
    )
    return parser.parse_args()


def _manifest_signature(manifest) -> str:
    training_fields = dict(manifest.training.__dict__)
    training_fields.pop("seed", None)
    sae_tracking_fields = dict(manifest.sae_tracking.__dict__)
    sae_tracking_fields.pop("seed", None)
    training_interventions = [intervention.__dict__ for intervention in manifest.training_interventions]
    formation_fields = dict(manifest.formation.__dict__)
    for key in ("candidate_support_head", "candidate_retrieval_head", "candidate_placebo_head"):
        if formation_fields[key] is not None:
            formation_fields[key] = formation_fields[key].__dict__
    return json.dumps(
        {
            "benchmark": manifest.benchmark.__dict__,
            "dataset": manifest.dataset.__dict__,
            "model": manifest.model.__dict__,
            "training": training_fields,
            "checkpoint_schedule": manifest.checkpoint_schedule.__dict__,
            "battery": manifest.battery.__dict__,
            "sae_tracking": sae_tracking_fields,
            "formation": formation_fields,
            "training_interventions": training_interventions,
            "summary_thresholds": manifest.summary_thresholds.__dict__,
        },
        sort_keys=True,
    )


def _flatten_formation_scalar_rows(run_id: str, seed: int, rows: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for row in rows:
        q_metrics = row.get("Q") or {}
        r_metrics = row.get("R") or {}
        w_metrics = row.get("W") or {}
        path_gain = row.get("path_gain") or {}
        flattened.append(
            {
                "run_id": run_id,
                "seed": seed,
                "epoch": int(row["epoch"]),
                "global_step": int(row["global_step"]),
                "curriculum_stage": str(row["curriculum_stage"]),
                "candidate_mode": str(row["candidate_mode"]),
                "support_head_name": None if row.get("candidate_support_head") is None else str(row["candidate_support_head"]["name"]),
                "retrieval_head_name": None if row.get("candidate_retrieval_head") is None else str(row["candidate_retrieval_head"]["name"]),
                "placebo_head_name": None if row.get("candidate_placebo_head") is None else str(row["candidate_placebo_head"]["name"]),
                "Q_margin_mean": float(q_metrics.get("margin_mean", float("nan"))),
                "Q_top_key_accuracy": float(q_metrics.get("top_key_accuracy", float("nan"))),
                "R_slot_margin_mean": float(r_metrics.get("slot_margin_mean", float("nan"))),
                "R_top_slot_accuracy": float(r_metrics.get("top_slot_accuracy", float("nan"))),
                "R_correct_slot_attention_mean": float(r_metrics.get("correct_slot_attention_mean", float("nan"))),
                "R_attention_entropy_mean": float(r_metrics.get("attention_entropy_mean", float("nan"))),
                "W_value_margin_mean": float(w_metrics.get("value_margin_mean", float("nan"))),
                "W_top_written_value_accuracy": float(w_metrics.get("top_written_value_accuracy", float("nan"))),
                "path_gain_routing_margin_delta": float(path_gain.get("routing_margin_delta", float("nan"))),
                "path_gain_correct_slot_attention_delta": float(path_gain.get("correct_slot_attention_delta", float("nan"))),
                "path_gain_top_slot_accuracy_delta": float(path_gain.get("top_slot_accuracy_delta", float("nan"))),
                "transition_metric_name": str(row.get("transition_metric_name") or "unknown"),
                "transition_metric_value": float(row.get("transition_metric_value", float("nan"))),
                "transition_boost_activated": bool(row.get("transition_boost_activated", False)),
            }
        )
    return flattened


def _flatten_nested_formation_rows(run_id: str, seed: int, rows: list[dict], key: str) -> list[dict]:
    flattened: list[dict] = []
    for row in rows:
        nested_rows = row.get(key) or []
        for nested in nested_rows:
            flattened.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "epoch": int(row["epoch"]),
                    "global_step": int(row["global_step"]),
                    **nested,
                }
            )
    return flattened


def _summarize_formation_births(formation_table: pd.DataFrame, manifests_by_run: dict[str, object]) -> pd.DataFrame:
    if formation_table.empty:
        return pd.DataFrame(
            columns=["run_id", "seed", "metric_name", "birth_epoch", "birth_global_step", "threshold"]
        )
    rows: list[dict] = []
    metric_specs = [
        ("Q", "Q_top_key_accuracy", "q_birth_accuracy"),
        ("R", "R_top_slot_accuracy", "r_birth_accuracy"),
        ("W", "W_top_written_value_accuracy", "w_birth_accuracy"),
    ]
    for run_id, group in formation_table.groupby("run_id"):
        manifest = manifests_by_run[run_id]
        ordered = group.sort_values(["epoch", "global_step"]).reset_index(drop=True)
        for metric_name, column_name, threshold_name in metric_specs:
            threshold = float(getattr(manifest.formation, threshold_name))
            passing = ordered[ordered[column_name] >= threshold]
            if passing.empty:
                rows.append(
                    {
                        "run_id": run_id,
                        "seed": int(ordered["seed"].iloc[0]),
                        "metric_name": metric_name,
                        "birth_epoch": float("nan"),
                        "birth_global_step": float("nan"),
                        "threshold": threshold,
                    }
                )
                continue
            first_row = passing.iloc[0]
            rows.append(
                {
                    "run_id": run_id,
                    "seed": int(first_row["seed"]),
                    "metric_name": metric_name,
                    "birth_epoch": int(first_row["epoch"]),
                    "birth_global_step": int(first_row["global_step"]),
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def _summarize_formation_lags(formation_births: pd.DataFrame) -> pd.DataFrame:
    if formation_births.empty:
        return pd.DataFrame(
            columns=["run_id", "seed", "Q_birth_epoch", "R_birth_epoch", "W_birth_epoch", "Q_to_R_lag", "R_to_W_lag"]
        )
    rows: list[dict] = []
    pivot = formation_births.pivot(index="run_id", columns="metric_name", values="birth_epoch")
    seed_by_run = formation_births.groupby("run_id")["seed"].first()
    for run_id, birth_row in pivot.iterrows():
        q_birth = birth_row.get("Q")
        r_birth = birth_row.get("R")
        w_birth = birth_row.get("W")
        rows.append(
            {
                "run_id": run_id,
                "seed": int(seed_by_run[run_id]),
                "Q_birth_epoch": q_birth,
                "R_birth_epoch": r_birth,
                "W_birth_epoch": w_birth,
                "Q_to_R_lag": (r_birth - q_birth) if pd.notna(q_birth) and pd.notna(r_birth) else float("nan"),
                "R_to_W_lag": (w_birth - r_birth) if pd.notna(r_birth) and pd.notna(w_birth) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _assert_compatible_manifests(manifests: list) -> None:
    if not manifests:
        raise ValueError("Expected at least one manifest to summarize")
    first_signature = _manifest_signature(manifests[0])
    for manifest in manifests[1:]:
        signature = _manifest_signature(manifest)
        if signature != first_signature:
            raise ValueError("All runs under a summarized target directory must share the same non-seed manifest fields")


def _load_formation_history(path: Path) -> list[dict]:
    from research.phase3.scripts.kv_formation_dynamics import load_formation_history

    return load_formation_history(path)


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    run_dirs = discover_run_directories(target_dir)
    manifests = [load_run_manifest(run_dir / "manifest.json") for run_dir in run_dirs]
    _assert_compatible_manifests(manifests)
    manifest = manifests[0]

    checkpoint_rows = []
    neuron_rows = []
    feature_rows = []
    superposition_rows = []
    representation_drift_rows = []
    operator_handoff_rows = []
    formation_scalar_rows = []
    formation_family_loss_rows = []
    formation_family_gradient_rows = []
    formation_gradient_cosine_rows = []
    formation_role_pair_rows = []
    formation_optimizer_rows = []
    formation_logit_rows = []
    manifests_by_run: dict[str, object] = {}
    for run_dir, manifest in zip(run_dirs, manifests, strict=True):
        manifests_by_run[manifest.run_id] = manifest
        checkpoint_rows.extend(collect_run_checkpoint_rows(run_dir, manifest))
        neuron_rows.extend(collect_run_neuron_rows(run_dir))
        feature_rows.extend(collect_run_feature_rows(run_dir, manifest))
        superposition_rows.extend(collect_run_superposition_rows(run_dir, manifest))
        representation_drift_rows.extend(collect_run_representation_drift_rows(run_dir))
        operator_handoff_rows.extend(collect_run_operator_handoff_rows(run_dir, manifest))
        formation_rows = _load_formation_history(run_dir / "formation_history.jsonl")
        formation_scalar_rows.extend(_flatten_formation_scalar_rows(manifest.run_id, manifest.training.seed, formation_rows))
        formation_family_loss_rows.extend(_flatten_nested_formation_rows(manifest.run_id, manifest.training.seed, formation_rows, "family_losses"))
        formation_family_gradient_rows.extend(_flatten_nested_formation_rows(manifest.run_id, manifest.training.seed, formation_rows, "family_gradients"))
        formation_gradient_cosine_rows.extend(_flatten_nested_formation_rows(manifest.run_id, manifest.training.seed, formation_rows, "family_gradient_cosines"))
        formation_role_pair_rows.extend(_flatten_nested_formation_rows(manifest.run_id, manifest.training.seed, formation_rows, "role_pair_gradient_cosines"))
        formation_optimizer_rows.extend(_flatten_nested_formation_rows(manifest.run_id, manifest.training.seed, formation_rows, "optimizer_role_metrics"))
        formation_logit_rows.extend(_flatten_nested_formation_rows(manifest.run_id, manifest.training.seed, formation_rows, "logit_contributions"))
    checkpoint_table = pd.DataFrame(checkpoint_rows).sort_values(["run_id", "epoch", "checkpoint_id"]).reset_index(drop=True)
    neuron_table = pd.DataFrame(neuron_rows).sort_values(
        ["run_id", "epoch", "layer_index", "best_selectivity_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    feature_table = pd.DataFrame(feature_rows)
    if not feature_table.empty:
        feature_table = feature_table.sort_values(
            ["run_id", "epoch", "site", "top_feature_selectivity_score"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
    superposition_table = pd.DataFrame(superposition_rows)
    if not superposition_table.empty:
        superposition_table = superposition_table.sort_values(
            ["run_id", "epoch", "site"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    emergence_table = summarize_emergence(checkpoint_table, manifest)
    seed_stability_table = summarize_seed_stability(checkpoint_table)
    representation_drift_table = pd.DataFrame(representation_drift_rows)
    if not representation_drift_table.empty:
        representation_drift_table = representation_drift_table.sort_values(
            ["run_id", "site", "epoch_left", "epoch_right"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    operator_handoff_table = pd.DataFrame(operator_handoff_rows)
    if not operator_handoff_table.empty:
        operator_handoff_table = operator_handoff_table.sort_values(
            ["run_id", "role_name", "epoch_previous", "epoch_current"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    role_matching_table = summarize_cross_seed_role_matching(run_dirs, manifests)
    if not role_matching_table.empty:
        role_matching_table = role_matching_table.sort_values(
            ["intervention_signature", "role_name", "run_id_left", "run_id_right"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    clamp_responsiveness_table = summarize_clamp_responsiveness(checkpoint_table, emergence_table)
    if not clamp_responsiveness_table.empty:
        clamp_responsiveness_table = clamp_responsiveness_table.sort_values(
            ["intervention_signature", "seed", "run_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    formation_table = pd.DataFrame(formation_scalar_rows)
    if not formation_table.empty:
        formation_table = formation_table.sort_values(["run_id", "epoch", "global_step"]).reset_index(drop=True)
    formation_births_table = _summarize_formation_births(formation_table, manifests_by_run)
    formation_lags_table = _summarize_formation_lags(formation_births_table)
    formation_family_loss_table = pd.DataFrame(formation_family_loss_rows)
    if not formation_family_loss_table.empty:
        formation_family_loss_table = formation_family_loss_table.sort_values(
            ["run_id", "epoch", "global_step", "family_name"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    formation_family_gradient_table = pd.DataFrame(formation_family_gradient_rows)
    if not formation_family_gradient_table.empty:
        formation_family_gradient_table = formation_family_gradient_table.sort_values(
            ["run_id", "epoch", "global_step", "role_name", "family_name"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
    formation_gradient_cosine_table = pd.DataFrame(formation_gradient_cosine_rows)
    if not formation_gradient_cosine_table.empty:
        formation_gradient_cosine_table = formation_gradient_cosine_table.sort_values(
            ["run_id", "epoch", "global_step", "role_name", "family_left", "family_right"],
            ascending=[True, True, True, True, True, True],
        ).reset_index(drop=True)
    formation_role_pair_table = pd.DataFrame(formation_role_pair_rows)
    if not formation_role_pair_table.empty:
        formation_role_pair_table = formation_role_pair_table.sort_values(
            ["run_id", "epoch", "global_step", "family_name", "role_left", "role_right"],
            ascending=[True, True, True, True, True, True],
        ).reset_index(drop=True)
    formation_optimizer_table = pd.DataFrame(formation_optimizer_rows)
    if not formation_optimizer_table.empty:
        formation_optimizer_table = formation_optimizer_table.sort_values(
            ["run_id", "epoch", "global_step", "role_name"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    formation_logit_table = pd.DataFrame(formation_logit_rows)
    if not formation_logit_table.empty:
        formation_logit_table = formation_logit_table.sort_values(
            ["run_id", "epoch", "global_step", "role_name", "prompt_id"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)

    summary_dir = (target_dir if len(run_dirs) > 1 else run_dirs[0]) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_table.to_csv(summary_dir / "checkpoint_index.csv", index=False)
    emergence_table.to_csv(summary_dir / "emergence.csv", index=False)
    seed_stability_table.to_csv(summary_dir / "seed_stability.csv", index=False)
    neuron_table.to_csv(summary_dir / "neuron_dynamics.csv", index=False)
    feature_table.to_csv(summary_dir / "feature_dynamics.csv", index=False)
    superposition_table.to_csv(summary_dir / "superposition_dynamics.csv", index=False)
    representation_drift_table.to_csv(summary_dir / "representation_drift.csv", index=False)
    operator_handoff_table.to_csv(summary_dir / "operator_handoffs.csv", index=False)
    role_matching_table.to_csv(summary_dir / "role_matching.csv", index=False)
    clamp_responsiveness_table.to_csv(summary_dir / "clamp_responsiveness.csv", index=False)
    formation_table.to_csv(summary_dir / "formation_dynamics.csv", index=False)
    formation_births_table.to_csv(summary_dir / "formation_births.csv", index=False)
    formation_lags_table.to_csv(summary_dir / "formation_lags.csv", index=False)
    formation_family_loss_table.to_csv(summary_dir / "formation_family_losses.csv", index=False)
    formation_family_gradient_table.to_csv(summary_dir / "formation_family_gradients.csv", index=False)
    formation_gradient_cosine_table.to_csv(summary_dir / "formation_gradient_cosines.csv", index=False)
    formation_role_pair_table.to_csv(summary_dir / "formation_role_pair_gradients.csv", index=False)
    formation_optimizer_table.to_csv(summary_dir / "formation_optimizer_metrics.csv", index=False)
    formation_logit_table.to_csv(summary_dir / "formation_logit_contributions.csv", index=False)

    behavior_birth_rows = emergence_table[emergence_table["metric_name"] == "behavior_val_accuracy"]
    query_faithfulness_birth_rows = emergence_table[emergence_table["metric_name"] == "faithfulness_query_key"]
    slot_faithfulness_birth_rows = emergence_table[emergence_table["metric_name"] == "faithfulness_matching_slot"]
    value_faithfulness_birth_rows = emergence_table[emergence_table["metric_name"] == "faithfulness_selected_value"]
    routing_birth_rows = emergence_table[emergence_table["metric_name"] == "operator_routing"]
    copy_birth_rows = emergence_table[emergence_table["metric_name"] == "operator_copy"]
    formation_q_birth_rows = formation_births_table[formation_births_table["metric_name"] == "Q"]
    formation_r_birth_rows = formation_births_table[formation_births_table["metric_name"] == "R"]
    formation_w_birth_rows = formation_births_table[formation_births_table["metric_name"] == "W"]
    merged_operator_birth = routing_birth_rows.merge(
        copy_birth_rows,
        on="run_id",
        suffixes=("_routing", "_copy"),
        how="inner",
    )
    run_summary = {
        "target_dir": str(target_dir),
        "num_runs": int(checkpoint_table["run_id"].nunique()),
        "num_checkpoints": int(len(checkpoint_table)),
        "num_neuron_rows": int(len(neuron_table)),
        "num_feature_rows": int(len(feature_table)),
        "num_superposition_rows": int(len(superposition_table)),
        "num_representation_drift_rows": int(len(representation_drift_table)),
        "num_operator_handoff_rows": int(len(operator_handoff_table)),
        "num_role_matching_rows": int(len(role_matching_table)),
        "num_clamp_responsiveness_rows": int(len(clamp_responsiveness_table)),
        "num_formation_rows": int(len(formation_table)),
        "num_formation_family_gradient_rows": int(len(formation_family_gradient_table)),
        "behavior_birth_runs": int(behavior_birth_rows["birth_epoch"].notna().sum()),
        "query_faithfulness_birth_runs": int(query_faithfulness_birth_rows["birth_epoch"].notna().sum()),
        "matching_slot_faithfulness_birth_runs": int(slot_faithfulness_birth_rows["birth_epoch"].notna().sum()),
        "selected_value_faithfulness_birth_runs": int(value_faithfulness_birth_rows["birth_epoch"].notna().sum()),
        "routing_birth_runs": int(routing_birth_rows["birth_epoch"].notna().sum()),
        "copy_birth_runs": int(copy_birth_rows["birth_epoch"].notna().sum()),
        "formation_q_birth_runs": int(formation_q_birth_rows["birth_epoch"].notna().sum()),
        "formation_r_birth_runs": int(formation_r_birth_rows["birth_epoch"].notna().sum()),
        "formation_w_birth_runs": int(formation_w_birth_rows["birth_epoch"].notna().sum()),
        "routing_before_copy_runs": int(
            (
                merged_operator_birth["birth_epoch_routing"].notna()
                & merged_operator_birth["birth_epoch_copy"].notna()
                & (merged_operator_birth["birth_epoch_routing"] <= merged_operator_birth["birth_epoch_copy"])
            ).sum()
        ),
    }
    (summary_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
