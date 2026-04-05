from __future__ import annotations

import math

import pandas as pd
import torch

from scripts.kv_algorithm_record import RecordedSiteDataset


def _standardize(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _fit_ridge_classifier(
    train_x: torch.Tensor,
    train_labels: list[object],
    eval_x: torch.Tensor,
    ridge_lambda: float,
) -> tuple[list[object], torch.Tensor]:
    classes = sorted({str(label) for label in train_labels})
    if len(classes) < 2:
        raise ValueError("Need at least two classes to fit a variable probe")
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = torch.zeros((train_x.shape[0], len(classes)), dtype=train_x.dtype)
    for row_index, label in enumerate(train_labels):
        y[row_index, class_to_index[str(label)]] = 1.0
    identity = torch.eye(train_x.shape[1], dtype=train_x.dtype)
    xtx = train_x.T @ train_x
    xty = train_x.T @ y
    weights = torch.linalg.solve(xtx + (ridge_lambda * identity), xty)
    logits = eval_x @ weights
    predicted_indices = logits.argmax(dim=1).tolist()
    predicted_labels = [classes[index] for index in predicted_indices]
    return predicted_labels, weights


def _fit_probe_weights(
    dataset: RecordedSiteDataset,
    site: str,
    variable: str,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    *,
    ridge_lambda: float = 1e-3,
) -> tuple[pd.DataFrame, list[str], list[str], torch.Tensor]:
    if site not in dataset.site_vectors:
        raise ValueError(f"Unknown site {site!r} in recorded dataset")
    if variable not in dataset.metadata.columns:
        raise ValueError(f"Unknown variable {variable!r} in recorded dataset metadata")
    if len(train_mask) != len(dataset.metadata) or len(eval_mask) != len(dataset.metadata):
        raise ValueError("Train/eval masks must match dataset metadata length")

    train_rows = dataset.metadata.loc[train_mask].reset_index(drop=True)
    eval_rows = dataset.metadata.loc[eval_mask].reset_index(drop=True)
    if train_rows.empty:
        raise ValueError(f"Train mask selected zero rows for site {site} and variable {variable}")
    if eval_rows.empty:
        raise ValueError(f"Eval mask selected zero rows for site {site} and variable {variable}")

    train_x = dataset.site_vectors[site][torch.tensor(train_mask.to_list(), dtype=torch.bool)]
    eval_x = dataset.site_vectors[site][torch.tensor(eval_mask.to_list(), dtype=torch.bool)]
    train_x, eval_x = _standardize(train_x.float(), eval_x.float())
    train_labels = train_rows[variable].astype(str).tolist()
    eval_labels = eval_rows[variable].astype(str).tolist()
    predicted_labels, weights = _fit_ridge_classifier(
        train_x=train_x,
        train_labels=train_labels,
        eval_x=eval_x,
        ridge_lambda=ridge_lambda,
    )
    return eval_rows, train_labels, eval_labels, predicted_labels, weights


def evaluate_site_variable_probe(
    dataset: RecordedSiteDataset,
    site: str,
    variable: str,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    *,
    ridge_lambda: float = 1e-3,
) -> dict[str, object]:
    eval_rows, train_labels, eval_labels, predicted_labels, weights = _fit_probe_weights(
        dataset=dataset,
        site=site,
        variable=variable,
        train_mask=train_mask,
        eval_mask=eval_mask,
        ridge_lambda=ridge_lambda,
    )

    correct_count = sum(
        predicted == expected for predicted, expected in zip(predicted_labels, eval_labels, strict=True)
    )
    class_count = len(sorted(set(train_labels)))
    chance_accuracy = 1.0 / class_count
    accuracy = correct_count / len(eval_labels)
    weight_norm = float(weights.norm().item())
    return {
        "site": site,
        "variable": variable,
        "train_rows": int(train_mask.sum()),
        "eval_rows": len(eval_rows),
        "num_classes": class_count,
        "chance_accuracy": chance_accuracy,
        "eval_accuracy": accuracy,
        "eval_margin_over_chance": accuracy - chance_accuracy,
        "weight_norm": weight_norm,
    }


def build_variable_recovery_table(
    dataset: RecordedSiteDataset,
    sites: list[str],
    variables: list[str],
    train_mask: pd.Series,
    eval_mask: pd.Series,
    *,
    ridge_lambda: float = 1e-3,
) -> pd.DataFrame:
    if not sites:
        raise ValueError("Expected at least one site for variable recovery")
    if not variables:
        raise ValueError("Expected at least one variable for variable recovery")

    rows: list[dict[str, object]] = []
    for site in sites:
        for variable in variables:
            rows.append(
                evaluate_site_variable_probe(
                    dataset=dataset,
                    site=site,
                    variable=variable,
                    train_mask=train_mask,
                    eval_mask=eval_mask,
                    ridge_lambda=ridge_lambda,
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["variable", "eval_accuracy", "eval_margin_over_chance"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_site_variable_ranking_table(variable_recovery_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"site", "variable", "eval_accuracy", "eval_margin_over_chance"}
    missing = required_columns - set(variable_recovery_table.columns)
    if missing:
        raise ValueError(f"Variable recovery table is missing required columns: {sorted(missing)}")
    return (
        variable_recovery_table
        .sort_values(["variable", "eval_accuracy", "eval_margin_over_chance"], ascending=[True, False, False])
        .groupby("variable", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )


def build_family_variable_stability_table(
    dataset: RecordedSiteDataset,
    site: str,
    variable: str,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    *,
    ridge_lambda: float = 1e-3,
) -> pd.DataFrame:
    eval_rows, _train_labels, eval_labels, predicted_labels, _ = _fit_probe_weights(
        dataset=dataset,
        site=site,
        variable=variable,
        train_mask=train_mask,
        eval_mask=eval_mask,
        ridge_lambda=ridge_lambda,
    )
    rows: list[dict[str, object]] = []
    scored_rows = eval_rows.copy()
    scored_rows["expected_label"] = eval_labels
    scored_rows["predicted_label"] = predicted_labels
    scored_rows["correct_label"] = scored_rows["expected_label"] == scored_rows["predicted_label"]
    for base_prompt_id, group in scored_rows.groupby("base_prompt_id"):
        rows.append(
            {
                "site": site,
                "variable": variable,
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "family_accuracy": float(group["correct_label"].mean()),
                "family_name_set": ", ".join(sorted(set(group["family_name"].astype(str)))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family_accuracy", "base_prompt_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_family_variable_stability_summary_table(
    family_variable_stability_table: pd.DataFrame,
) -> pd.DataFrame:
    if family_variable_stability_table.empty:
        raise ValueError("Expected a non-empty family variable stability table")
    return pd.DataFrame(
        [
            {
                "site": str(family_variable_stability_table["site"].iloc[0]),
                "variable": str(family_variable_stability_table["variable"].iloc[0]),
                "families": int(family_variable_stability_table["base_prompt_id"].nunique()),
                "family_accuracy_mean": float(family_variable_stability_table["family_accuracy"].mean()),
                "family_accuracy_min": float(family_variable_stability_table["family_accuracy"].min()),
                "family_accuracy_max": float(family_variable_stability_table["family_accuracy"].max()),
            }
        ]
    )
