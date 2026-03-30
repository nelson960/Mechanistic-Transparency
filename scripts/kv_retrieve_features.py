from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.kv_retrieve_analysis import (
    DatasetBundle,
    apply_rope,
    head_residual_contribution,
    source_component_tensor,
    make_query_swap_prompt,
    residual_vector_to_logits,
    run_prompt,
    summarize_logits_against_target,
)


FEATURE_SITES = {
    "block1_final_resid": "Block 1 residual stream after MLP at the final position",
    "block1_final_l1h0": "L1H0 residual contribution at the final position",
    "l2h0_final_q": "L2H0 query vector at the final position",
}


@dataclass(frozen=True)
class ActivationRecord:
    split: str
    index: int
    prompt: str
    target: str
    query_key: str


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return features, reconstruction

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min_(1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.bias
        return F.relu(self.encoder(centered))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.bias


def _extract_activation_from_cache(model: nn.Module, cache: dict, site: str) -> torch.Tensor:
    if site == "block1_final_resid":
        return cache["blocks"][0]["resid_after_mlp"][0, -1].detach().cpu()
    if site == "block1_final_l1h0":
        return head_residual_contribution(model, cache, layer_index=0, head_index=0)[0, -1].detach().cpu()
    if site == "l2h0_final_q":
        return cache["blocks"][1]["attention"]["q"][0, 0, -1].detach().cpu()
    raise ValueError(f"Unsupported feature site: {site}")


def collect_split_activations(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    site: str,
    limit: int | None = None,
) -> tuple[torch.Tensor, list[ActivationRecord]]:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split: {split}")
    if site not in FEATURE_SITES:
        raise ValueError(f"Unknown feature site: {site}")

    rows = bundle.raw_splits[split]
    if limit is not None:
        rows = rows[:limit]

    activations = []
    records: list[ActivationRecord] = []
    device = model.token_embed.weight.device
    for index, row in enumerate(rows):
        result, cache = run_prompt(
            model,
            bundle,
            row["prompt"],
            device=device,
            expected_target=row["target"],
            return_cache=True,
        )
        if cache is None:
            raise ValueError("Expected a cache when collecting split activations")
        activations.append(_extract_activation_from_cache(model, cache, site))
        records.append(
            ActivationRecord(
                split=split,
                index=index,
                prompt=row["prompt"],
                target=row["target"],
                query_key=row["query_key"],
            )
        )

    return torch.stack(activations), records


def collect_query_swap_pairs(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    site: str,
    limit: int | None = None,
) -> dict[str, object]:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split: {split}")
    if site not in FEATURE_SITES:
        raise ValueError(f"Unknown feature site: {site}")

    rows = bundle.raw_splits[split]
    if limit is not None:
        rows = rows[:limit]

    clean_activations = []
    corrupt_activations = []
    pair_rows = []
    q_deltas = []
    device = model.token_embed.weight.device

    for index, row in enumerate(rows):
        clean_prompt = row["prompt"]
        corrupt_prompt, corrupt_target = make_query_swap_prompt(row)

        clean_result, clean_cache = run_prompt(
            model,
            bundle,
            clean_prompt,
            device=device,
            expected_target=row["target"],
            return_cache=True,
        )
        corrupt_result, corrupt_cache = run_prompt(
            model,
            bundle,
            corrupt_prompt,
            device=device,
            expected_target=corrupt_target,
            return_cache=True,
        )
        if clean_cache is None or corrupt_cache is None:
            raise ValueError("Expected caches when collecting clean/corrupt feature pairs")

        clean_activations.append(_extract_activation_from_cache(model, clean_cache, site))
        corrupt_activations.append(_extract_activation_from_cache(model, corrupt_cache, site))

        clean_q = clean_cache["blocks"][1]["attention"]["q"][0, 0, -1].detach().cpu()
        corrupt_q = corrupt_cache["blocks"][1]["attention"]["q"][0, 0, -1].detach().cpu()
        q_deltas.append(clean_q - corrupt_q)

        pair_rows.append(
            {
                "index": index,
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "clean_target": row["target"],
                "clean_query_key": row["query_key"],
                "clean_predicted_token": clean_result["predicted_token"],
                "corrupt_predicted_token": corrupt_result["predicted_token"],
            }
        )

    return {
        "clean_activations": torch.stack(clean_activations),
        "corrupt_activations": torch.stack(corrupt_activations),
        "mean_q_delta": torch.stack(q_deltas).mean(dim=0),
        "rows": pair_rows,
    }


def sae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    l1_coeff: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss = F.mse_loss(reconstruction, target)
    l1_loss = features.abs().mean()
    loss = recon_loss + l1_coeff * l1_loss
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "recon_loss": float(recon_loss.detach().cpu().item()),
        "l1_loss": float(l1_loss.detach().cpu().item()),
        "mean_active_features": float((features > 0).float().sum(dim=1).mean().detach().cpu().item()),
    }
    return loss, metrics


def _evaluate_sae(sae: SparseAutoencoder, activations: torch.Tensor, l1_coeff: float) -> dict[str, float]:
    sae.eval()
    with torch.no_grad():
        features, reconstruction = sae(activations)
        _, metrics = sae_loss(reconstruction, activations, features, l1_coeff)
    return metrics


def train_sae(
    train_activations: torch.Tensor,
    val_activations: torch.Tensor,
    hidden_dim: int,
    l1_coeff: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> tuple[SparseAutoencoder, pd.DataFrame]:
    if train_activations.ndim != 2:
        raise ValueError(f"Expected train activations to be rank 2, got {tuple(train_activations.shape)}")
    if val_activations.ndim != 2:
        raise ValueError(f"Expected val activations to be rank 2, got {tuple(val_activations.shape)}")
    if train_activations.shape[1] != val_activations.shape[1]:
        raise ValueError("Train and val activations must have the same feature dimension")

    torch.manual_seed(seed)
    sae = SparseAutoencoder(input_dim=train_activations.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    history_rows = []

    for epoch in range(1, epochs + 1):
        sae.train()
        permutation = torch.randperm(train_activations.shape[0])
        epoch_metrics = {"loss": 0.0, "recon_loss": 0.0, "l1_loss": 0.0, "mean_active_features": 0.0}
        batch_count = 0

        for start in range(0, train_activations.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch = train_activations[indices]
            features, reconstruction = sae(batch)
            loss, metrics = sae_loss(reconstruction, batch, features, l1_coeff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder_columns()

            for key, value in metrics.items():
                epoch_metrics[key] += value
            batch_count += 1

        averaged_train = {key: value / batch_count for key, value in epoch_metrics.items()}
        val_metrics = _evaluate_sae(sae, val_activations, l1_coeff)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": averaged_train["loss"],
                "train_recon_loss": averaged_train["recon_loss"],
                "train_l1_loss": averaged_train["l1_loss"],
                "train_mean_active_features": averaged_train["mean_active_features"],
                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_l1_loss": val_metrics["l1_loss"],
                "val_mean_active_features": val_metrics["mean_active_features"],
            }
        )

    return sae, pd.DataFrame(history_rows)


def build_feature_delta_table(
    sae: SparseAutoencoder,
    clean_activations: torch.Tensor,
    corrupt_activations: torch.Tensor,
) -> pd.DataFrame:
    sae.eval()
    with torch.no_grad():
        clean_features, _ = sae(clean_activations)
        corrupt_features, _ = sae(corrupt_activations)

    clean_mean = clean_features.mean(dim=0)
    corrupt_mean = corrupt_features.mean(dim=0)
    delta_mean = clean_mean - corrupt_mean
    activation_rate = (clean_features > 0).float().mean(dim=0)

    rows = []
    for feature_index in range(clean_features.shape[1]):
        rows.append(
            {
                "feature_index": feature_index,
                "mean_clean_activation": float(clean_mean[feature_index].item()),
                "mean_corrupt_activation": float(corrupt_mean[feature_index].item()),
                "mean_delta": float(delta_mean[feature_index].item()),
                "abs_mean_delta": float(delta_mean[feature_index].abs().item()),
                "clean_activation_rate": float(activation_rate[feature_index].item()),
            }
        )

    return pd.DataFrame(rows)


def build_feature_projection_table(
    model: nn.Module,
    bundle: DatasetBundle,
    sae: SparseAutoencoder,
    mean_q_delta: torch.Tensor,
    final_position_index: int,
    site: str,
) -> pd.DataFrame:
    if site not in FEATURE_SITES:
        raise ValueError(f"Unknown feature site for projection table: {site}")

    l2h0_attn = model.blocks[1].attn
    head_dim = l2h0_attn.head_dim
    w_q_head = l2h0_attn.q_proj.weight[:head_dim, :].detach().cpu()
    cos_final = l2h0_attn.rope_cos[final_position_index].detach().cpu().view(1, 1, 1, -1)
    sin_final = l2h0_attn.rope_sin[final_position_index].detach().cpu().view(1, 1, 1, -1)
    mean_q_delta = mean_q_delta.detach().cpu()

    rows = []
    for feature_index in range(sae.hidden_dim):
        decoder_vector = sae.decoder.weight[:, feature_index].detach().cpu()
        if decoder_vector.shape[0] == w_q_head.shape[1]:
            projected_q = torch.matmul(w_q_head, decoder_vector)
            projected_q = apply_rope(projected_q.view(1, 1, 1, -1), cos_final, sin_final).view(-1)
            logits = residual_vector_to_logits(model, decoder_vector)
            top_logits, top_indices = torch.topk(logits, k=min(5, logits.shape[0]))
            top_logit_tokens = [bundle.id_to_token[int(idx.item())] for idx in top_indices]
            top_logit_values = [float(value.item()) for value in top_logits]
            projection_space = "residual_to_query"
        elif decoder_vector.shape[0] == mean_q_delta.shape[0]:
            projected_q = decoder_vector
            top_logit_tokens = []
            top_logit_values = []
            projection_space = "query_space"
        else:
            raise ValueError(
                "Unsupported decoder dimension for feature projection table: "
                f"decoder has dim {decoder_vector.shape[0]}, "
                f"W_Q expects {w_q_head.shape[1]}, "
                f"mean_q_delta has dim {mean_q_delta.shape[0]}"
            )
        rows.append(
            {
                "feature_index": feature_index,
                "projection_space": projection_space,
                "decoder_norm": float(decoder_vector.norm().item()),
                "projected_q_norm": float(projected_q.norm().item()),
                "query_alignment": float(torch.dot(projected_q, mean_q_delta).item()),
                "query_cosine": float(
                    F.cosine_similarity(projected_q.unsqueeze(0), mean_q_delta.unsqueeze(0)).item()
                ),
                "top_logit_tokens": top_logit_tokens,
                "top_logit_values": top_logit_values,
            }
        )

    return pd.DataFrame(rows)


def build_top_feature_examples(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    records: list[ActivationRecord],
    feature_indices: list[int],
    top_k: int,
) -> dict[int, list[dict[str, object]]]:
    sae.eval()
    with torch.no_grad():
        features, _ = sae(activations)

    result: dict[int, list[dict[str, object]]] = {}
    for feature_index in feature_indices:
        values = features[:, feature_index]
        top_values, top_positions = torch.topk(values, k=min(top_k, values.shape[0]))
        examples = []
        for activation_value, record_position in zip(top_values.tolist(), top_positions.tolist()):
            record = records[record_position]
            examples.append(
                {
                    "activation": float(activation_value),
                    "split": record.split,
                    "index": record.index,
                    "prompt": record.prompt,
                    "target": record.target,
                    "query_key": record.query_key,
                }
            )
        result[feature_index] = examples

    return result


def _feature_column_name(feature_index: int) -> str:
    return f"feature_{feature_index}_activation"


def select_feature_panel(
    feature_summary_df: pd.DataFrame,
    available_example_feature_ids: list[int] | None = None,
    support_count: int = 2,
    control_count: int = 1,
) -> dict[str, list[int]]:
    required_columns = {
        "feature_index",
        "mean_delta",
        "abs_mean_delta",
        "query_alignment",
        "mechanism_score",
    }
    missing_columns = sorted(required_columns - set(feature_summary_df.columns))
    if missing_columns:
        raise ValueError(f"Feature summary missing required columns: {missing_columns}")
    if support_count <= 0:
        raise ValueError(f"support_count must be positive, got {support_count}")
    if control_count <= 0:
        raise ValueError(f"control_count must be positive, got {control_count}")

    pool = feature_summary_df.copy()
    pool["feature_index"] = pool["feature_index"].astype(int)
    if available_example_feature_ids is not None:
        allowed = {int(feature_id) for feature_id in available_example_feature_ids}
        pool = pool[pool["feature_index"].isin(allowed)].copy()
        if pool.empty:
            raise ValueError("Feature pool is empty after filtering to available-example features")

    support_df = pool[(pool["mean_delta"] > 0.0) & (pool["query_alignment"] > 0.0)].copy()
    support_df = support_df.sort_values(
        ["mechanism_score", "abs_mean_delta"],
        ascending=[False, False],
    )
    if len(support_df) < support_count:
        raise ValueError(
            f"Need at least {support_count} support features, found {len(support_df)}"
        )
    support_features = support_df.head(support_count)["feature_index"].tolist()

    remaining_df = pool[~pool["feature_index"].isin(support_features)].copy()
    if remaining_df.empty:
        raise ValueError("No remaining features available for control selection")
    remaining_df["abs_mechanism_score"] = remaining_df["mechanism_score"].abs()
    control_df = remaining_df.sort_values(
        ["abs_mechanism_score", "abs_mean_delta"],
        ascending=[True, True],
    )
    if len(control_df) < control_count:
        raise ValueError(
            f"Need at least {control_count} control features, found {len(control_df)}"
        )
    control_features = control_df.head(control_count)["feature_index"].tolist()

    return {
        "support_features": [int(feature_id) for feature_id in support_features],
        "control_features": [int(feature_id) for feature_id in control_features],
        "panel_features": [int(feature_id) for feature_id in support_features + control_features],
    }


def build_feature_activation_table(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    records: list[ActivationRecord],
    bundle: DatasetBundle,
    feature_indices: list[int],
) -> pd.DataFrame:
    if activations.ndim != 2:
        raise ValueError(f"Expected activations to be rank 2, got {tuple(activations.shape)}")
    if activations.shape[0] != len(records):
        raise ValueError(
            "Activation/record length mismatch: "
            f"{activations.shape[0]} activations vs {len(records)} records"
        )
    if not feature_indices:
        raise ValueError("Expected at least one feature index")

    normalized_feature_indices = [int(feature_index) for feature_index in feature_indices]
    invalid_feature_indices = [
        feature_index
        for feature_index in normalized_feature_indices
        if feature_index < 0 or feature_index >= sae.hidden_dim
    ]
    if invalid_feature_indices:
        raise ValueError(f"Invalid feature indices for activation table: {invalid_feature_indices}")

    sae.eval()
    with torch.no_grad():
        feature_values = sae.encode(activations)

    rows: list[dict[str, object]] = []
    for row_index, record in enumerate(records):
        if record.split not in bundle.raw_splits:
            raise ValueError(f"Unknown split in activation record: {record.split}")
        split_rows = bundle.raw_splits[record.split]
        if record.index < 0 or record.index >= len(split_rows):
            raise ValueError(
                f"Activation record index out of range for split={record.split}: {record.index}"
            )
        raw_row = split_rows[record.index]
        prompt_tokens = record.prompt.split()
        correct_value_positions = [
            position for position, token in enumerate(prompt_tokens) if token == record.target
        ]
        if len(correct_value_positions) != 1:
            raise ValueError(
                f"Expected exactly one correct value position for prompt {record.prompt!r}, "
                f"found {correct_value_positions}"
            )
        query_context_positions = [
            position
            for position, token in enumerate(prompt_tokens[:-2])
            if token == record.query_key
        ]
        if len(query_context_positions) != 1:
            raise ValueError(
                f"Expected exactly one context occurrence of query key {record.query_key!r}, "
                f"found {query_context_positions}"
            )
        query_pair_matches = [
            pair for pair in raw_row["context_pairs"] if pair["key"] == record.query_key
        ]
        if len(query_pair_matches) != 1:
            raise ValueError(
                f"Expected exactly one query-pair match for {record.query_key!r}, "
                f"found {len(query_pair_matches)}"
            )
        _, corrupt_target = make_query_swap_prompt(raw_row)
        corrupt_value_positions = [
            position for position, token in enumerate(prompt_tokens) if token == corrupt_target
        ]
        if len(corrupt_value_positions) != 1:
            raise ValueError(
                f"Expected exactly one corrupt value position for token {corrupt_target!r}, "
                f"found {corrupt_value_positions}"
            )

        table_row: dict[str, object] = {
            "split": record.split,
            "index": record.index,
            "prompt": record.prompt,
            "target": record.target,
            "query_key": record.query_key,
            "num_pairs": int(raw_row["num_pairs"]),
            "prompt_length": len(prompt_tokens),
            "query_suffix_position": len(prompt_tokens) - 2,
            "final_position": len(prompt_tokens) - 1,
            "query_context_position": query_context_positions[0],
            "correct_value_position": correct_value_positions[0],
            "corrupt_target": corrupt_target,
            "corrupt_value_position": corrupt_value_positions[0],
            "query_pair_index": int(query_pair_matches[0]["pair_index"]),
        }
        for feature_index in normalized_feature_indices:
            table_row[_feature_column_name(feature_index)] = float(
                feature_values[row_index, feature_index].item()
            )
        rows.append(table_row)

    return pd.DataFrame(rows)


def build_feature_group_summary_table(
    feature_activation_df: pd.DataFrame,
    feature_indices: list[int],
    group_column: str,
) -> pd.DataFrame:
    if group_column not in feature_activation_df.columns:
        raise ValueError(f"Unknown group column for feature summary: {group_column}")
    if not feature_indices:
        raise ValueError("Expected at least one feature index for grouped summary")

    normalized_feature_indices = [int(feature_index) for feature_index in feature_indices]
    missing_feature_columns = [
        _feature_column_name(feature_index)
        for feature_index in normalized_feature_indices
        if _feature_column_name(feature_index) not in feature_activation_df.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            f"Feature activation table is missing columns for grouped summary: {missing_feature_columns}"
        )

    rows: list[dict[str, object]] = []
    grouped = feature_activation_df.groupby(group_column, dropna=False)
    for group_value, group_df in grouped:
        row: dict[str, object] = {
            group_column: group_value,
            "num_examples": int(len(group_df)),
        }
        for feature_index in normalized_feature_indices:
            column_name = _feature_column_name(feature_index)
            row[column_name] = float(group_df[column_name].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def build_feature_encoder_contribution_table(
    sae: SparseAutoencoder,
    vector: torch.Tensor,
    feature_index: int,
) -> pd.DataFrame:
    feature_index = int(feature_index)
    if feature_index < 0 or feature_index >= sae.hidden_dim:
        raise ValueError(
            f"feature_index must be in [0, {sae.hidden_dim}), got {feature_index}"
        )
    if vector.ndim != 1:
        raise ValueError(f"Expected vector to be rank 1, got {tuple(vector.shape)}")
    if vector.shape[0] != sae.input_dim:
        raise ValueError(
            f"Vector/input mismatch for encoder contribution table: "
            f"vector dim {vector.shape[0]} vs SAE input dim {sae.input_dim}"
        )

    with torch.no_grad():
        centered = vector.to(dtype=sae.bias.dtype) - sae.bias.detach().cpu()
        encoder_weight = sae.encoder.weight[feature_index].detach().cpu()
        encoder_bias = float(sae.encoder.bias[feature_index].detach().cpu().item())
        contributions = centered * encoder_weight
        feature_pre_activation = float(contributions.sum().item() + encoder_bias)
        feature_post_activation = float(F.relu(torch.tensor(feature_pre_activation)).item())

    rows = []
    for dimension in range(vector.shape[0]):
        rows.append(
            {
                "dimension": dimension,
                "input_value": float(vector[dimension].item()),
                "centered_input": float(centered[dimension].item()),
                "encoder_weight": float(encoder_weight[dimension].item()),
                "contribution": float(contributions[dimension].item()),
                "abs_contribution": float(contributions[dimension].abs().item()),
                "feature_pre_activation": feature_pre_activation,
                "feature_post_activation": feature_post_activation,
                "encoder_bias": encoder_bias,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["abs_contribution", "contribution"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_feature_analysis(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_sae_checkpoint(
    path: Path,
    sae: SparseAutoencoder,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": sae.input_dim,
            "hidden_dim": sae.hidden_dim,
            "state_dict": sae.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_sae_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[dict, SparseAutoencoder]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    sae = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    return checkpoint, sae


def intervene_on_sae_features(
    sae: SparseAutoencoder,
    base_vector: torch.Tensor,
    feature_indices: list[int],
    mode: str,
    source_vector: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if base_vector.ndim != 1:
        raise ValueError(f"Expected base_vector to be rank 1, got {tuple(base_vector.shape)}")
    if source_vector is not None and source_vector.ndim != 1:
        raise ValueError(f"Expected source_vector to be rank 1, got {tuple(source_vector.shape)}")
    if not feature_indices:
        raise ValueError("Expected at least one feature index for intervention")

    device = sae.bias.device
    dtype = sae.bias.dtype
    base_vector = base_vector.to(device=device, dtype=dtype)
    if source_vector is not None:
        source_vector = source_vector.to(device=device, dtype=dtype)

    with torch.no_grad():
        base_features = sae.encode(base_vector.unsqueeze(0))
        modified_features = base_features.clone()
        source_features = None

        if mode == "ablate":
            modified_features[:, feature_indices] = 0.0
        elif mode == "patch":
            if source_vector is None:
                raise ValueError("Expected source_vector for feature patching")
            source_features = sae.encode(source_vector.unsqueeze(0))
            modified_features[:, feature_indices] = source_features[:, feature_indices]
        else:
            raise ValueError(f"Unsupported feature intervention mode: {mode}")

        reconstructed = sae.decode(modified_features)[0].detach().cpu()

    result = {
        "base_features": base_features[0].detach().cpu(),
        "modified_features": modified_features[0].detach().cpu(),
        "reconstructed": reconstructed,
    }
    if source_features is not None:
        result["source_features"] = source_features[0].detach().cpu()
    return result


def forward_with_modified_source(
    model: nn.Module,
    input_ids: torch.Tensor,
    base_cache: dict,
    source_patch: dict,
    modified_source_tensor: torch.Tensor,
    destination_layer_index: int,
) -> tuple[torch.Tensor, dict]:
    source_layer_index = source_patch["layer_index"]
    if destination_layer_index <= 0:
        raise ValueError("Destination layer must be after the first layer")
    if source_layer_index != destination_layer_index - 1:
        raise ValueError(
            "This helper only supports modifying a source component from the immediately previous layer "
            f"(got source layer {source_layer_index + 1} and destination layer {destination_layer_index + 1})."
        )

    resid = base_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    base_source = source_component_tensor(model, base_cache, source_patch, resid.device, resid.dtype)
    if modified_source_tensor.shape != base_source.shape:
        raise ValueError(
            "Modified source tensor shape mismatch: "
            f"expected {tuple(base_source.shape)}, got {tuple(modified_source_tensor.shape)}"
        )
    modified_source_tensor = modified_source_tensor.to(device=resid.device, dtype=resid.dtype)

    source_positions = source_patch.get("source_positions")
    if source_positions is None:
        patched_resid_for_destination = resid - base_source + modified_source_tensor
    else:
        invalid_positions = [
            position for position in source_positions if position < 0 or position >= resid.shape[1]
        ]
        if invalid_positions:
            raise ValueError(f"Invalid source positions for modified source intervention: {invalid_positions}")
        source_delta = modified_source_tensor - base_source
        position_mask = torch.zeros_like(source_delta)
        for position in source_positions:
            position_mask[:, position, :] = 1.0
        patched_resid_for_destination = resid + (source_delta * position_mask)

    block = model.blocks[destination_layer_index]
    resid, destination_cache = block(patched_resid_for_destination, capture=True)
    downstream_caches = []
    for layer_index in range(destination_layer_index + 1, len(model.blocks)):
        resid, block_cache = model.blocks[layer_index](resid, capture=True)
        downstream_caches.append(block_cache)

    final_hidden = model.norm_final(resid)
    logits = final_hidden @ model.token_embed.weight.T
    return logits, {
        "patched_resid_for_destination": patched_resid_for_destination.detach().cpu(),
        "base_source": base_source.detach().cpu(),
        "modified_source": modified_source_tensor.detach().cpu(),
        "destination_cache": destination_cache,
        "downstream_caches": downstream_caches,
    }


def score_feature_intervention(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    target_token: str,
    base_cache: dict,
    source_patch: dict,
    modified_source_tensor: torch.Tensor,
    destination_layer_index: int,
    device: torch.device,
) -> dict:
    input_ids = torch.tensor(
        [[bundle.token_to_id[token] for token in prompt.split()]],
        device=device,
    )
    with torch.no_grad():
        logits, details = forward_with_modified_source(
            model=model,
            input_ids=input_ids,
            base_cache=base_cache,
            source_patch=source_patch,
            modified_source_tensor=modified_source_tensor,
            destination_layer_index=destination_layer_index,
        )
    final_logits = logits[0, -1].detach().cpu()
    result = summarize_logits_against_target(final_logits, bundle, target_token)
    result["details"] = details
    return result


def forward_with_modified_query(
    model: nn.Module,
    input_ids: torch.Tensor,
    base_cache: dict,
    layer_index: int,
    head_index: int,
    position_index: int,
    modified_query_vector: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    if layer_index <= 0:
        raise ValueError("Modified-query helper currently expects a non-first-layer destination")

    resid = base_cache["blocks"][layer_index - 1]["resid_after_mlp"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    block = model.blocks[layer_index]
    attn = block.attn

    base_q = base_cache["blocks"][layer_index]["attention"]["q"].to(
        device=resid.device,
        dtype=resid.dtype,
    )
    base_k = base_cache["blocks"][layer_index]["attention"]["k"].to(
        device=resid.device,
        dtype=resid.dtype,
    )
    base_v = base_cache["blocks"][layer_index]["attention"]["v"].to(
        device=resid.device,
        dtype=resid.dtype,
    )
    base_head_out = base_cache["blocks"][layer_index]["attention"]["head_out"].to(
        device=resid.device,
        dtype=resid.dtype,
    )

    if head_index < 0 or head_index >= base_q.shape[1]:
        raise ValueError(f"Invalid head index {head_index} for modified-query intervention")
    if position_index < 0 or position_index >= base_q.shape[2]:
        raise ValueError(f"Invalid position index {position_index} for modified-query intervention")

    modified_query_vector = modified_query_vector.to(device=resid.device, dtype=resid.dtype)
    if modified_query_vector.ndim != 1 or modified_query_vector.shape[0] != base_q.shape[-1]:
        raise ValueError(
            "Modified query vector shape mismatch: "
            f"expected {(base_q.shape[-1],)}, got {tuple(modified_query_vector.shape)}"
        )

    q = base_q.clone()
    q[:, head_index, position_index, :] = modified_query_vector

    scores = torch.matmul(q, base_k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    seq_len = q.shape[2]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=resid.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    pattern = scores.softmax(dim=-1)
    patched_head_out = torch.matmul(pattern, base_v)

    mixed_head_out = base_head_out.clone()
    mixed_head_out[:, head_index, :, :] = patched_head_out[:, head_index, :, :]

    attn_out = attn.o_proj(attn.merge_heads(mixed_head_out))
    resid = resid + attn_out
    resid = resid + block.mlp(block.norm2(resid))

    downstream_caches = []
    for downstream_layer in range(layer_index + 1, len(model.blocks)):
        resid, block_cache = model.blocks[downstream_layer](resid, capture=True)
        downstream_caches.append(block_cache)

    final_hidden = model.norm_final(resid)
    logits = final_hidden @ model.token_embed.weight.T
    return logits, {
        "modified_q": q.detach().cpu(),
        "base_q": base_q.detach().cpu(),
        "scores": scores.detach().cpu(),
        "pattern": pattern.detach().cpu(),
        "patched_head_out": patched_head_out.detach().cpu(),
        "downstream_caches": downstream_caches,
    }


def score_query_feature_intervention(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    target_token: str,
    base_cache: dict,
    layer_index: int,
    head_index: int,
    position_index: int,
    modified_query_vector: torch.Tensor,
    device: torch.device,
) -> dict:
    input_ids = torch.tensor(
        [[bundle.token_to_id[token] for token in prompt.split()]],
        device=device,
    )
    with torch.no_grad():
        logits, details = forward_with_modified_query(
            model=model,
            input_ids=input_ids,
            base_cache=base_cache,
            layer_index=layer_index,
            head_index=head_index,
            position_index=position_index,
            modified_query_vector=modified_query_vector,
        )
    final_logits = logits[0, -1].detach().cpu()
    result = summarize_logits_against_target(final_logits, bundle, target_token)
    result["details"] = details
    return result
