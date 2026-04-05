from __future__ import annotations

from collections import defaultdict

import pandas as pd
import torch

from scripts.kv_algorithm_record import RecordedPrompt


VARIABLE_NAMES = ["query_key", "matching_slot", "selected_value"]


def _eta_squared(values: torch.Tensor, labels: list[str]) -> float:
    if values.ndim != 1:
        raise ValueError(f"Expected a rank-1 value vector for eta-squared, got {tuple(values.shape)}")
    if len(labels) != values.shape[0]:
        raise ValueError("Value/label length mismatch in eta-squared computation")

    grand_mean = values.mean()
    total_ss = float(((values - grand_mean) ** 2).sum().item())
    if total_ss <= 0.0:
        return 0.0

    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[str(label)].append(index)

    between_ss = 0.0
    for indices in label_to_indices.values():
        group_values = values[torch.tensor(indices, dtype=torch.long)]
        group_mean = group_values.mean()
        between_ss += float(len(indices) * ((group_mean - grand_mean) ** 2).item())
    return between_ss / total_ss


def _group_mean_gap(values: torch.Tensor, labels: list[str]) -> float:
    label_to_means: dict[str, float] = {}
    for label in sorted(set(labels)):
        mask = torch.tensor([candidate == label for candidate in labels], dtype=torch.bool)
        label_to_means[label] = float(values[mask].mean().item())
    if not label_to_means:
        return 0.0
    return max(label_to_means.values()) - min(label_to_means.values())


def build_mlp_neuron_score_table(
    model: torch.nn.Module,
    recorded_prompts: list[RecordedPrompt],
    *,
    position: int = -1,
) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt for neuron tracking")

    label_sets = {
        variable: [str(getattr(recorded.annotation, variable)) for recorded in recorded_prompts]
        for variable in VARIABLE_NAMES
    }
    rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        write_weight = block.mlp.down_proj.weight.detach().cpu()
        activations = torch.stack(
            [
                recorded.cache["blocks"][layer_index]["mlp"]["activated"][0, position, :].detach().cpu()
                for recorded in recorded_prompts
            ]
        )
        if activations.ndim != 2:
            raise ValueError(
                f"Expected neuron activation matrix with shape [rows, neurons], got {tuple(activations.shape)}"
            )
        for neuron_index in range(activations.shape[1]):
            neuron_values = activations[:, neuron_index]
            variable_scores = {
                variable: _eta_squared(neuron_values, labels)
                for variable, labels in label_sets.items()
            }
            variable_gaps = {
                variable: _group_mean_gap(neuron_values, labels)
                for variable, labels in label_sets.items()
            }
            best_variable = max(variable_scores, key=variable_scores.get)
            rows.append(
                {
                    "layer_index": layer_index,
                    "layer_name": f"block{layer_index + 1}",
                    "neuron_index": neuron_index,
                    "component": f"block{layer_index + 1}.mlp.neuron_{neuron_index}",
                    "mean_activation": float(neuron_values.mean().item()),
                    "mean_abs_activation": float(neuron_values.abs().mean().item()),
                    "activation_std": float(neuron_values.std(unbiased=False).item()),
                    "positive_rate": float((neuron_values > 0.0).float().mean().item()),
                    "nonzero_rate": float((neuron_values.abs() > 1e-8).float().mean().item()),
                    "max_activation": float(neuron_values.max().item()),
                    "min_activation": float(neuron_values.min().item()),
                    "write_norm": float(write_weight[:, neuron_index].norm().item()),
                    "activation_write_product": float(
                        neuron_values.abs().mean().item() * write_weight[:, neuron_index].norm().item()
                    ),
                    "query_key_eta2": variable_scores["query_key"],
                    "query_key_group_gap": variable_gaps["query_key"],
                    "matching_slot_eta2": variable_scores["matching_slot"],
                    "matching_slot_group_gap": variable_gaps["matching_slot"],
                    "selected_value_eta2": variable_scores["selected_value"],
                    "selected_value_group_gap": variable_gaps["selected_value"],
                    "best_variable": best_variable,
                    "best_selectivity_score": float(variable_scores[best_variable]),
                    "best_variable_group_gap": float(variable_gaps[best_variable]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["layer_index", "best_selectivity_score", "activation_write_product"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_layer_top_neuron_table(
    neuron_score_table: pd.DataFrame,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    if neuron_score_table.empty:
        raise ValueError("Expected a non-empty neuron score table")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return (
        neuron_score_table
        .sort_values(["layer_index", "best_selectivity_score", "activation_write_product"], ascending=[True, False, False])
        .groupby("layer_index", as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )
