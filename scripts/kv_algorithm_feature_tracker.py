from __future__ import annotations

from collections import defaultdict

import pandas as pd
import torch

from scripts.kv_algorithm_record import RecordedSiteDataset
from scripts.kv_retrieve_features import SparseAutoencoder


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


def build_feature_score_table(
    dataset: RecordedSiteDataset,
    site: str,
    sae: SparseAutoencoder,
    eval_mask: pd.Series,
) -> pd.DataFrame:
    if site not in dataset.site_vectors:
        raise ValueError(f"Unknown site {site!r} in recorded dataset")
    if len(eval_mask) != len(dataset.metadata):
        raise ValueError("Eval mask must match dataset metadata length")

    eval_rows = dataset.metadata.loc[eval_mask].reset_index(drop=True)
    if eval_rows.empty:
        raise ValueError(f"Eval mask selected zero rows for site {site}")

    eval_vectors = dataset.site_vectors[site][torch.tensor(eval_mask.to_list(), dtype=torch.bool)].float()
    sae.eval()
    with torch.no_grad():
        feature_values = sae.encode(eval_vectors).detach().cpu()
        decoder = sae.decoder.weight.detach().cpu()

    label_sets = {
        variable: eval_rows[variable].astype(str).tolist()
        for variable in VARIABLE_NAMES
    }
    rows: list[dict[str, object]] = []
    for feature_index in range(feature_values.shape[1]):
        values = feature_values[:, feature_index]
        variable_scores = {
            variable: _eta_squared(values, labels)
            for variable, labels in label_sets.items()
        }
        variable_gaps = {
            variable: _group_mean_gap(values, labels)
            for variable, labels in label_sets.items()
        }
        best_variable = max(variable_scores, key=variable_scores.get)
        rows.append(
            {
                "row_kind": "feature",
                "site": site,
                "feature_index": feature_index,
                "input_dim": int(sae.input_dim),
                "hidden_dim": int(sae.hidden_dim),
                "mean_activation": float(values.mean().item()),
                "mean_abs_activation": float(values.abs().mean().item()),
                "activation_std": float(values.std(unbiased=False).item()),
                "activation_rate": float((values > 0.0).float().mean().item()),
                "max_activation": float(values.max().item()),
                "decoder_norm": float(decoder[:, feature_index].norm().item()),
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
        ["site", "best_selectivity_score", "mean_abs_activation", "activation_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_feature_site_summary_table(
    feature_score_table: pd.DataFrame,
    *,
    top_features_per_site: int,
) -> pd.DataFrame:
    if feature_score_table.empty:
        raise ValueError("Expected a non-empty feature score table")
    if top_features_per_site <= 0:
        raise ValueError(f"top_features_per_site must be positive, got {top_features_per_site}")

    rows: list[dict[str, object]] = []
    for site, site_df in feature_score_table.groupby("site"):
        ordered = site_df.sort_values(
            ["best_selectivity_score", "mean_abs_activation", "activation_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        top_df = ordered.head(top_features_per_site)
        top_row = top_df.iloc[0]
        rows.append(
            {
                "row_kind": "site_summary",
                "site": str(site),
                "feature_count": int(len(site_df)),
                "active_feature_count": int((site_df["activation_rate"] > 0.0).sum()),
                "top_feature_indices": ",".join(str(int(index)) for index in top_df["feature_index"].tolist()),
                "top_feature_index": int(top_row["feature_index"]),
                "top_feature_variable": str(top_row["best_variable"]),
                "top_feature_selectivity_score": float(top_row["best_selectivity_score"]),
                "mean_top_feature_selectivity_score": float(top_df["best_selectivity_score"].mean()),
                "mean_top_feature_activation_rate": float(top_df["activation_rate"].mean()),
                "mean_decoder_norm": float(site_df["decoder_norm"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)


def build_superposition_metrics(
    site: str,
    sae: SparseAutoencoder,
    eval_vectors: torch.Tensor,
    feature_score_table: pd.DataFrame,
    history_table: pd.DataFrame,
    *,
    cosine_threshold: float,
) -> dict[str, object]:
    if eval_vectors.ndim != 2:
        raise ValueError(f"Expected eval activations to be rank 2, got {tuple(eval_vectors.shape)}")
    if history_table.empty:
        raise ValueError("Expected a non-empty SAE training history table")
    if cosine_threshold < 0.0 or cosine_threshold > 1.0:
        raise ValueError(f"cosine_threshold must lie in [0, 1], got {cosine_threshold}")

    sae.eval()
    with torch.no_grad():
        feature_values = sae.encode(eval_vectors.float()).detach().cpu()
        decoder = sae.decoder.weight.detach().cpu()

    active_mask = feature_values > 0.0
    decoder_norms = decoder.norm(dim=0).clamp_min(1e-8)
    normalized_decoder = decoder / decoder_norms.unsqueeze(0)
    cosine_matrix = normalized_decoder.T @ normalized_decoder
    pair_mask = torch.triu(torch.ones_like(cosine_matrix, dtype=torch.bool), diagonal=1)
    pairwise_abs_cosines = cosine_matrix.abs()[pair_mask]

    singular_values = torch.linalg.svdvals(decoder.float())
    spectral_norm = float(singular_values[0].item()) if singular_values.numel() > 0 else 0.0
    fro_norm = float(decoder.float().norm().item())
    stable_rank = (fro_norm ** 2) / (spectral_norm ** 2) if spectral_norm > 0.0 else 0.0

    ordered_scores = feature_score_table.sort_values(
        ["best_selectivity_score", "mean_abs_activation"],
        ascending=[False, False],
    ).reset_index(drop=True)
    top_feature_mean = float(ordered_scores.head(min(5, len(ordered_scores)))["best_selectivity_score"].mean())

    return {
        "site": site,
        "input_dim": int(sae.input_dim),
        "hidden_dim": int(sae.hidden_dim),
        "hidden_to_input_ratio": float(sae.hidden_dim / sae.input_dim),
        "active_feature_count": int(active_mask.any(dim=0).sum().item()),
        "mean_active_features_per_example": float(active_mask.float().sum(dim=1).mean().item()),
        "activation_density": float(active_mask.float().mean().item()),
        "decoder_mean_norm": float(decoder_norms.mean().item()),
        "decoder_max_norm": float(decoder_norms.max().item()),
        "decoder_mean_abs_pairwise_cosine": (
            float(pairwise_abs_cosines.mean().item()) if pairwise_abs_cosines.numel() > 0 else 0.0
        ),
        "decoder_max_abs_pairwise_cosine": (
            float(pairwise_abs_cosines.max().item()) if pairwise_abs_cosines.numel() > 0 else 0.0
        ),
        "decoder_overlap_fraction": (
            float((pairwise_abs_cosines >= cosine_threshold).float().mean().item())
            if pairwise_abs_cosines.numel() > 0
            else 0.0
        ),
        "decoder_stable_rank": float(stable_rank),
        "top_feature_selectivity_max": float(ordered_scores["best_selectivity_score"].max()),
        "top_feature_selectivity_mean_top5": top_feature_mean,
        "sae_final_train_loss": float(history_table.iloc[-1]["train_loss"]),
        "sae_final_val_loss": float(history_table.iloc[-1]["val_loss"]),
        "sae_final_val_recon_loss": float(history_table.iloc[-1]["val_recon_loss"]),
        "sae_final_val_l1_loss": float(history_table.iloc[-1]["val_l1_loss"]),
        "sae_final_val_mean_active_features": float(history_table.iloc[-1]["val_mean_active_features"]),
    }
