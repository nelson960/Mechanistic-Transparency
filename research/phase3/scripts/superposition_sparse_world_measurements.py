from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRACKED_STEP_METRICS = [
    "batch_loss",
    "epoch",
    "global_step",
    "batch_index",
    "batch_rows",
    "total_grad_norm",
    "total_param_norm_pre",
    "total_param_norm_post",
    "total_update_norm",
    "relative_total_update_norm",
    "parameter_metrics",
]

TRACKED_CHECKPOINT_METRICS = {
    "behavior": [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
        "ood_loss",
        "ood_accuracy",
        "margins",
    ],
    "variables": [
        "query_color",
        "query_position",
        "query_state",
        "matched_entity",
        "selected_label",
    ],
    "operators": [
        "attribute_filtering_heads",
        "entity_binding_heads",
        "label_transport_heads",
    ],
    "weights": [
        "parameter_norms",
        "gradient_norms",
        "update_norms",
        "relative_update_norms",
        "singular_values",
        "matrix_alignment",
    ],
    "neurons": [
        "activation_mean",
        "activation_std",
        "positive_rate",
        "feature_selectivity",
    ],
    "features": [
        "sae_feature_birth",
        "sae_feature_persistence",
        "decoder_overlap",
        "superposition_overlap_fraction",
    ],
    "dynamics": [
        "representation_drift",
        "operator_handoff",
        "variable_faithfulness",
    ],
}


def build_measurement_plan() -> dict[str, Any]:
    return {
        "dataset_name": "superposition_sparse_world",
        "tracking_style": "same_as_previous_experiment_but_with_sparse_multi_attribute_data",
        "step_metrics": TRACKED_STEP_METRICS,
        "checkpoint_metrics": TRACKED_CHECKPOINT_METRICS,
        "notes": [
            "Track cheap global dynamics every optimizer step.",
            "Track richer mechanistic artifacts at saved checkpoints.",
            "Use the same tiny-transformer architecture family as the KV runs.",
            "Focus on emergence of compressed, polysemantic internal structure.",
        ],
    }


def save_measurement_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_measurement_plan(), indent=2), encoding="utf-8")


def build_tracking_story() -> str:
    plan = build_measurement_plan()
    lines = [
        f"Dataset: {plan['dataset_name']}",
        f"Tracking style: {plan['tracking_style']}",
        "",
        "Step metrics:",
    ]
    lines.extend(f"- {metric}" for metric in plan["step_metrics"])
    lines.append("")
    lines.append("Checkpoint groups:")
    for group_name, metrics in plan["checkpoint_metrics"].items():
        lines.append(f"- {group_name}: " + ", ".join(metrics))
    return "\n".join(lines)
