from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scripts.kv_algorithm_record import build_final_position_site_list, extract_site_vector
from scripts.kv_algorithm_variable_finder import (
    build_family_variable_stability_summary_table,
    build_family_variable_stability_table,
    build_site_variable_ranking_table,
    build_variable_recovery_table,
)
from scripts.kv_benchmark import build_weight_metrics
from scripts.kv_retrieve_analysis import (
    DatasetBundle,
    encode_prompt,
    decode_token,
    forward_with_head_ablation,
    head_residual_contribution,
    ov_source_logits,
    run_prompt,
    score_patched_prompt,
    score_qkv_patched_prompt,
)
from scripts.kv_retrieve_features import save_sae_checkpoint, train_sae, SparseAutoencoder
from scripts.tiny_transformer_core import load_tiny_decoder_checkpoint
from scripts.training_dynamics import evaluate_next_token_rows


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
BLOCK_SITE_PATTERN = re.compile(r"^block(?P<block>\d+)_final_(?P<kind>resid_after_mlp|mlp_out)$")
HEAD_SITE_PATTERN = re.compile(
    r"^block(?P<block>\d+)_head(?P<head>\d+)_final_(?P<kind>q|k|v|head_out|resid_contribution)$"
)

STORY_VARIABLE_NAMES = [
    "previous_token",
    "target_seen_in_context",
    "repeat_distance_bucket",
    "target_token",
]


@dataclass(frozen=True)
class StoryPromptAnnotation:
    prompt_id: str
    base_prompt_id: str
    family_name: str
    family_value: str
    source_kind: str
    split: str
    prompt: str
    target: str
    prompt_tokens: list[str]
    prompt_length: int
    previous_token: str
    previous_token_position: int
    target_token: str
    target_seen_in_context: str
    last_target_occurrence_position: int | None
    last_target_occurrence_distance: int | None
    repeat_distance_bucket: str
    target_token_kind: str
    target_is_capitalized: str


@dataclass(frozen=True)
class StoryRecordedPrompt:
    row: dict[str, object]
    annotation: StoryPromptAnnotation
    result: dict[str, object]
    cache: dict[str, Any]


@dataclass(frozen=True)
class StoryRecordedSiteDataset:
    metadata: pd.DataFrame
    site_vectors: dict[str, torch.Tensor]


def _is_punctuation_token(token: str) -> bool:
    return bool(token) and all(not character.isalnum() and not character.isspace() for character in token)


def _token_kind(token: str) -> str:
    if _is_punctuation_token(token):
        return "punctuation"
    if token[:1].isupper():
        return "capitalized_word"
    return "word"


def _repeat_distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "none"
    if distance == 1:
        return "1"
    if 2 <= distance <= 4:
        return "2_4"
    if 5 <= distance <= 8:
        return "5_8"
    return "9_plus"


def tokenize_story_text(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        raise ValueError("Story text produced zero tokens under the configured tokenizer")
    return tokens


def _build_story_row(
    *,
    row_id: str,
    split: str,
    source_kind: str,
    base_prompt_id: str,
    family_name: str,
    family_value: str,
    context_tokens: list[str],
    target_token: str,
    target_index: int,
) -> dict[str, object]:
    if not context_tokens:
        raise ValueError("Expected at least one context token when building a story row")
    prompt_tokens = ["<bos>"] + list(context_tokens)
    prompt = " ".join(prompt_tokens)
    seen_positions = [index for index, token in enumerate(prompt_tokens) if index > 0 and token == target_token]
    last_occurrence_position = max(seen_positions) if seen_positions else None
    last_occurrence_distance = (
        (len(prompt_tokens) - 1 - last_occurrence_position)
        if last_occurrence_position is not None
        else None
    )
    return {
        "id": row_id,
        "task": "story_text_circuit_origins",
        "split": split,
        "source_kind": source_kind,
        "base_prompt_id": base_prompt_id,
        "family_name": family_name,
        "family_value": family_value,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "target": target_token,
        "target_index": target_index,
        "prompt_length": len(prompt_tokens),
        "context_length": len(context_tokens),
        "previous_token": context_tokens[-1],
        "previous_token_position": len(prompt_tokens) - 1,
        "target_seen_in_context": "yes" if last_occurrence_position is not None else "no",
        "last_target_occurrence_position": last_occurrence_position,
        "last_target_occurrence_distance": last_occurrence_distance,
        "repeat_distance_bucket": _repeat_distance_bucket(last_occurrence_distance),
        "target_token_kind": _token_kind(target_token),
        "target_is_capitalized": "yes" if target_token[:1].isupper() and not _is_punctuation_token(target_token) else "no",
    }


def _build_split_rows(
    *,
    tokens: list[str],
    split: str,
    start_target_index: int,
    stop_target_index: int,
    context_length: int,
    stride: int,
) -> list[dict[str, object]]:
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if start_target_index < context_length:
        start_target_index = context_length
    rows: list[dict[str, object]] = []
    for target_index in range(start_target_index, stop_target_index, stride):
        context_tokens = tokens[target_index - context_length:target_index]
        row_id = f"{split}_{target_index:06d}"
        rows.append(
            _build_story_row(
                row_id=row_id,
                split=split,
                source_kind="dataset",
                base_prompt_id=row_id,
                family_name="base",
                family_value="full_context",
                context_tokens=context_tokens,
                target_token=tokens[target_index],
                target_index=target_index,
            )
        )
    if not rows:
        raise ValueError(
            f"Split {split!r} produced zero rows; context_length={context_length}, "
            f"target_index range=[{start_target_index}, {stop_target_index})"
        )
    return rows


def build_story_bundle(
    *,
    text_path: Path,
    context_length: int,
    ood_context_length: int,
    stride: int,
    train_fraction: float,
    val_fraction: float,
) -> DatasetBundle:
    if not text_path.exists():
        raise FileNotFoundError(f"Missing story dataset text file: {text_path}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if ood_context_length < context_length:
        raise ValueError(
            f"ood_context_length must be at least context_length, got {ood_context_length} < {context_length}"
        )
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1.0")

    tokens = tokenize_story_text(text_path.read_text(encoding="utf-8"))
    train_stop = int(len(tokens) * train_fraction)
    val_stop = int(len(tokens) * (train_fraction + val_fraction))
    if train_stop <= context_length:
        raise ValueError("Not enough tokens to build a non-empty training split")
    if val_stop <= train_stop:
        raise ValueError("Computed validation boundary does not exceed training boundary")
    if len(tokens) <= val_stop:
        raise ValueError("Not enough tokens to build a non-empty test split")

    raw_splits = {
        "train": _build_split_rows(
            tokens=tokens,
            split="train",
            start_target_index=context_length,
            stop_target_index=train_stop,
            context_length=context_length,
            stride=stride,
        ),
        "val": _build_split_rows(
            tokens=tokens,
            split="val",
            start_target_index=train_stop,
            stop_target_index=val_stop,
            context_length=context_length,
            stride=stride,
        ),
        "test": _build_split_rows(
            tokens=tokens,
            split="test",
            start_target_index=val_stop,
            stop_target_index=len(tokens),
            context_length=context_length,
            stride=stride,
        ),
        "test_ood_longer_context": _build_split_rows(
            tokens=tokens,
            split="test_ood_longer_context",
            start_target_index=max(val_stop, ood_context_length),
            stop_target_index=len(tokens),
            context_length=ood_context_length,
            stride=stride,
        ),
    }
    vocab = ["<bos>"] + sorted(set(tokens))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    metadata = {
        "name": "random_story_dataset_v1",
        "source_text_path": str(text_path),
        "token_count": len(tokens),
        "tokenizer": "regex_word_or_punctuation",
        "context_length": context_length,
        "ood_context_length": ood_context_length,
        "stride": stride,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "splits": {name: len(rows) for name, rows in raw_splits.items()},
        "target": "Single next-token prediction over the story token stream.",
    }
    return DatasetBundle(
        metadata=metadata,
        raw_splits=raw_splits,
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        value_tokens=[],
    )


def save_story_bundle(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")
    for split_name, rows in bundle.raw_splits.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                serializable = dict(row)
                handle.write(json.dumps(serializable) + "\n")


def annotate_story_row(row: dict[str, object]) -> StoryPromptAnnotation:
    required_keys = {
        "id",
        "base_prompt_id",
        "family_name",
        "family_value",
        "source_kind",
        "split",
        "prompt",
        "prompt_tokens",
        "target",
        "prompt_length",
        "previous_token",
        "previous_token_position",
        "target_seen_in_context",
        "repeat_distance_bucket",
        "target_token_kind",
        "target_is_capitalized",
    }
    missing = sorted(required_keys - set(row))
    if missing:
        raise ValueError(f"Story row is missing required keys: {missing}")
    prompt_tokens = row["prompt_tokens"]
    if not isinstance(prompt_tokens, list) or not prompt_tokens:
        raise ValueError("Expected row['prompt_tokens'] to be a non-empty list")
    return StoryPromptAnnotation(
        prompt_id=str(row["id"]),
        base_prompt_id=str(row["base_prompt_id"]),
        family_name=str(row["family_name"]),
        family_value=str(row["family_value"]),
        source_kind=str(row["source_kind"]),
        split=str(row["split"]),
        prompt=str(row["prompt"]),
        target=str(row["target"]),
        prompt_tokens=[str(token) for token in prompt_tokens],
        prompt_length=int(row["prompt_length"]),
        previous_token=str(row["previous_token"]),
        previous_token_position=int(row["previous_token_position"]),
        target_token=str(row["target"]),
        target_seen_in_context=str(row["target_seen_in_context"]),
        last_target_occurrence_position=(
            None
            if row.get("last_target_occurrence_position") is None
            else int(row["last_target_occurrence_position"])
        ),
        last_target_occurrence_distance=(
            None
            if row.get("last_target_occurrence_distance") is None
            else int(row["last_target_occurrence_distance"])
        ),
        repeat_distance_bucket=str(row["repeat_distance_bucket"]),
        target_token_kind=str(row["target_token_kind"]),
        target_is_capitalized=str(row["target_is_capitalized"]),
    )


def story_annotation_to_dict(annotation: StoryPromptAnnotation) -> dict[str, object]:
    return {
        "prompt_id": annotation.prompt_id,
        "base_prompt_id": annotation.base_prompt_id,
        "family_name": annotation.family_name,
        "family_value": annotation.family_value,
        "source_kind": annotation.source_kind,
        "split": annotation.split,
        "prompt": annotation.prompt,
        "target": annotation.target,
        "target_token": annotation.target_token,
        "previous_token": annotation.previous_token,
        "target_seen_in_context": annotation.target_seen_in_context,
        "repeat_distance_bucket": annotation.repeat_distance_bucket,
        "target_token_kind": annotation.target_token_kind,
        "target_is_capitalized": annotation.target_is_capitalized,
        "prompt_length": annotation.prompt_length,
        "previous_token_position": annotation.previous_token_position,
        "last_target_occurrence_position": annotation.last_target_occurrence_position,
        "last_target_occurrence_distance": annotation.last_target_occurrence_distance,
    }


def _variant_prompt_tokens(prompt_tokens: list[str], variant_name: str) -> list[str] | None:
    if not prompt_tokens or prompt_tokens[0] != "<bos>":
        raise ValueError("Expected prompt tokens to begin with <bos>")
    body = prompt_tokens[1:]
    if variant_name == "full_context":
        return list(prompt_tokens)
    if variant_name == "half_context":
        keep = max(4, len(body) // 2)
        if keep >= len(body):
            return None
        return ["<bos>"] + body[-keep:]
    if variant_name == "quarter_context":
        keep = max(4, len(body) // 4)
        if keep >= len(body):
            return None
        return ["<bos>"] + body[-keep:]
    if variant_name == "last8_context":
        keep = min(8, len(body))
        if keep >= len(body):
            return None
        return ["<bos>"] + body[-keep:]
    raise ValueError(f"Unknown variant_name {variant_name!r}")


def build_story_sweeps(base_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not base_rows:
        raise ValueError("Expected at least one base row to build story sweeps")
    sweep_rows: list[dict[str, object]] = []
    for base_row in base_rows:
        prompt_tokens = [str(token) for token in base_row["prompt_tokens"]]
        target_token = str(base_row["target"])
        for variant_name in ["full_context", "half_context", "quarter_context", "last8_context"]:
            variant_prompt_tokens = _variant_prompt_tokens(prompt_tokens, variant_name)
            if variant_prompt_tokens is None:
                continue
            sweep_rows.append(
                _build_story_row(
                    row_id=f"{base_row['id']}::{variant_name}",
                    split="sweep",
                    source_kind="sweep",
                    base_prompt_id=str(base_row["id"]),
                    family_name="context_variant",
                    family_value=variant_name,
                    context_tokens=variant_prompt_tokens[1:],
                    target_token=target_token,
                    target_index=int(base_row["target_index"]),
                )
            )
        last_occurrence_position = base_row.get("last_target_occurrence_position")
        if last_occurrence_position is not None:
            last_occurrence_position = int(last_occurrence_position)
            body = prompt_tokens[1:]
            if 0 < last_occurrence_position < len(prompt_tokens) - 1:
                trimmed_context = body[last_occurrence_position:]
                if len(trimmed_context) >= 4:
                    sweep_rows.append(
                        _build_story_row(
                            row_id=f"{base_row['id']}::drop_repeat_prefix",
                            split="sweep",
                            source_kind="sweep",
                            base_prompt_id=str(base_row["id"]),
                            family_name="context_variant",
                            family_value="drop_repeat_prefix",
                            context_tokens=trimmed_context,
                            target_token=target_token,
                            target_index=int(base_row["target_index"]),
                        )
                    )
    if not sweep_rows:
        raise ValueError("Story sweep generation produced zero rows")
    return sweep_rows


def build_story_eval_pack(bundle: DatasetBundle, sweep_base_limit: int) -> list[dict[str, object]]:
    test_rows = bundle.raw_splits["test"]
    if len(test_rows) < sweep_base_limit:
        raise ValueError(
            f"Test split contains only {len(test_rows)} rows; expected at least {sweep_base_limit} for sweep generation"
        )
    return build_story_sweeps(test_rows[:sweep_base_limit])


def record_story_prompt_rows(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    device: torch.device,
) -> list[StoryRecordedPrompt]:
    if not rows:
        raise ValueError("Expected at least one row to record")
    recorded_prompts: list[StoryRecordedPrompt] = []
    for row in rows:
        annotation = annotate_story_row(row)
        result, cache = run_prompt(
            model,
            bundle,
            annotation.prompt,
            device=device,
            expected_target=annotation.target,
            return_cache=True,
        )
        if cache is None:
            raise ValueError(f"Expected a cache for prompt {annotation.prompt_id}")
        recorded_prompts.append(
            StoryRecordedPrompt(
                row=row,
                annotation=annotation,
                result=result,
                cache=cache,
            )
        )
    return recorded_prompts


def build_story_site_dataset(
    model: torch.nn.Module,
    recorded_prompts: list[StoryRecordedPrompt],
    sites: list[str],
) -> StoryRecordedSiteDataset:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt")
    if not sites:
        raise ValueError("Expected at least one site")
    metadata_rows: list[dict[str, object]] = []
    site_vectors: dict[str, list[torch.Tensor]] = {site: [] for site in sites}
    for recorded in recorded_prompts:
        row_metadata = {
            **story_annotation_to_dict(recorded.annotation),
            "predicted_token": str(recorded.result["predicted_token"]),
            "correct": bool(recorded.result.get("correct", False)),
            "margin": float(recorded.result.get("margin", float("nan"))),
        }
        metadata_rows.append(row_metadata)
        for site in sites:
            site_vectors[site].append(extract_site_vector(model, recorded.cache, site))
    stacked = {site: torch.stack(vectors) for site, vectors in site_vectors.items()}
    return StoryRecordedSiteDataset(metadata=pd.DataFrame(metadata_rows), site_vectors=stacked)


def build_story_recording_summary_table(recorded_prompts: list[StoryRecordedPrompt]) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt")
    rows = []
    for recorded in recorded_prompts:
        rows.append(
            {
                "prompt_id": recorded.annotation.prompt_id,
                "base_prompt_id": recorded.annotation.base_prompt_id,
                "family_name": recorded.annotation.family_name,
                "family_value": recorded.annotation.family_value,
                "previous_token": recorded.annotation.previous_token,
                "target_token": recorded.annotation.target_token,
                "target_seen_in_context": recorded.annotation.target_seen_in_context,
                "repeat_distance_bucket": recorded.annotation.repeat_distance_bucket,
                "predicted_token": recorded.result["predicted_token"],
                "correct": bool(recorded.result.get("correct", False)),
                "margin": float(recorded.result.get("margin", float("nan"))),
                "prompt": recorded.annotation.prompt,
            }
        )
    return pd.DataFrame(rows)


def build_behavior_artifact(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    eval_rows: list[dict[str, object]],
    device: torch.device,
    eval_batch_size: int,
) -> dict[str, Any]:
    split_metrics = {}
    for split_name, rows in bundle.raw_splits.items():
        result = evaluate_next_token_rows(
            model,
            rows,
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=eval_batch_size,
        )
        split_metrics[split_name] = {
            "rows": result["rows"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "margin": result["margin"],
        }
    sweep_eval = evaluate_next_token_rows(
        model,
        eval_rows,
        token_to_id=bundle.token_to_id,
        id_to_token=bundle.id_to_token,
        device=device,
        batch_size=eval_batch_size,
    )
    prediction_lookup = {row["prompt_id"]: row for row in sweep_eval["predictions"]}
    family_breakdown: list[dict[str, Any]] = []
    family_labels = sorted({f"{row['family_name']}::{row['family_value']}" for row in eval_rows})
    for family_label in family_labels:
        family_rows = [
            prediction_lookup[str(row["id"])]
            for row in eval_rows
            if f"{row['family_name']}::{row['family_value']}" == family_label
        ]
        margins = [float(row["margin"]) for row in family_rows]
        family_breakdown.append(
            {
                "family_label": family_label,
                "rows": len(family_rows),
                "accuracy": sum(bool(row["correct"]) for row in family_rows) / len(family_rows),
                "margin": sum(margins) / len(margins),
            }
        )
    return {
        "split_metrics": split_metrics,
        "family_breakdown": family_breakdown,
    }


def _eta_squared(values: torch.Tensor, labels: list[str]) -> float:
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
    means = []
    for label in sorted(set(labels)):
        mask = torch.tensor([candidate == label for candidate in labels], dtype=torch.bool)
        means.append(float(values[mask].mean().item()))
    if not means:
        return 0.0
    return max(means) - min(means)


def build_mlp_neuron_score_table(
    model: torch.nn.Module,
    recorded_prompts: list[StoryRecordedPrompt],
) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt for neuron tracking")
    label_sets = {
        variable: [str(getattr(recorded.annotation, variable)) for recorded in recorded_prompts]
        for variable in STORY_VARIABLE_NAMES
    }
    rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        write_weight = block.mlp.down_proj.weight.detach().cpu()
        activations = torch.stack(
            [recorded.cache["blocks"][layer_index]["mlp"]["activated"][0, -1, :].detach().cpu() for recorded in recorded_prompts]
        )
        for neuron_index in range(activations.shape[1]):
            neuron_values = activations[:, neuron_index]
            variable_scores = {variable: _eta_squared(neuron_values, labels) for variable, labels in label_sets.items()}
            variable_gaps = {variable: _group_mean_gap(neuron_values, labels) for variable, labels in label_sets.items()}
            best_variable = max(variable_scores, key=variable_scores.get)
            row = {
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
                "activation_write_product": float(neuron_values.abs().mean().item() * write_weight[:, neuron_index].norm().item()),
                "best_variable": best_variable,
                "best_selectivity_score": float(variable_scores[best_variable]),
                "best_variable_group_gap": float(variable_gaps[best_variable]),
            }
            for variable in STORY_VARIABLE_NAMES:
                row[f"{variable}_eta2"] = float(variable_scores[variable])
                row[f"{variable}_group_gap"] = float(variable_gaps[variable])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["layer_index", "best_selectivity_score", "activation_write_product"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_layer_top_neuron_table(neuron_score_table: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if neuron_score_table.empty:
        raise ValueError("Expected a non-empty neuron score table")
    return (
        neuron_score_table
        .sort_values(["layer_index", "best_selectivity_score", "activation_write_product"], ascending=[True, False, False])
        .groupby("layer_index", as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def build_feature_score_table(
    dataset: StoryRecordedSiteDataset,
    site: str,
    sae: SparseAutoencoder,
    eval_mask: pd.Series,
) -> pd.DataFrame:
    if site not in dataset.site_vectors:
        raise ValueError(f"Unknown site {site!r}")
    eval_rows = dataset.metadata.loc[eval_mask].reset_index(drop=True)
    if eval_rows.empty:
        raise ValueError(f"Eval mask selected zero rows for site {site}")
    eval_vectors = dataset.site_vectors[site][torch.tensor(eval_mask.to_list(), dtype=torch.bool)].float()
    sae.eval()
    with torch.no_grad():
        feature_values = sae.encode(eval_vectors).detach().cpu()
        decoder = sae.decoder.weight.detach().cpu()
    label_sets = {variable: eval_rows[variable].astype(str).tolist() for variable in STORY_VARIABLE_NAMES}
    rows: list[dict[str, object]] = []
    for feature_index in range(feature_values.shape[1]):
        values = feature_values[:, feature_index]
        variable_scores = {variable: _eta_squared(values, labels) for variable, labels in label_sets.items()}
        variable_gaps = {variable: _group_mean_gap(values, labels) for variable, labels in label_sets.items()}
        best_variable = max(variable_scores, key=variable_scores.get)
        row = {
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
            "best_variable": best_variable,
            "best_selectivity_score": float(variable_scores[best_variable]),
            "best_variable_group_gap": float(variable_gaps[best_variable]),
        }
        for variable in STORY_VARIABLE_NAMES:
            row[f"{variable}_eta2"] = float(variable_scores[variable])
            row[f"{variable}_group_gap"] = float(variable_gaps[variable])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["site", "best_selectivity_score", "mean_abs_activation", "activation_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_feature_site_summary_table(feature_score_table: pd.DataFrame, top_features_per_site: int) -> pd.DataFrame:
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
        "decoder_mean_abs_pairwise_cosine": float(pairwise_abs_cosines.mean().item()) if pairwise_abs_cosines.numel() > 0 else 0.0,
        "decoder_max_abs_pairwise_cosine": float(pairwise_abs_cosines.max().item()) if pairwise_abs_cosines.numel() > 0 else 0.0,
        "decoder_overlap_fraction": (
            float((pairwise_abs_cosines >= cosine_threshold).float().mean().item())
            if pairwise_abs_cosines.numel() > 0
            else 0.0
        ),
        "decoder_stable_rank": float(stable_rank),
        "top_feature_selectivity_max": float(ordered_scores["best_selectivity_score"].max()),
        "top_feature_selectivity_mean_top5": float(
            ordered_scores.head(min(5, len(ordered_scores)))["best_selectivity_score"].mean()
        ),
        "sae_final_train_loss": float(history_table.iloc[-1]["train_loss"]),
        "sae_final_val_loss": float(history_table.iloc[-1]["val_loss"]),
        "sae_final_val_recon_loss": float(history_table.iloc[-1]["val_recon_loss"]),
        "sae_final_val_l1_loss": float(history_table.iloc[-1]["val_l1_loss"]),
        "sae_final_val_mean_active_features": float(history_table.iloc[-1]["val_mean_active_features"]),
    }


def build_variable_scores(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    sweep_rows: list[dict[str, object]],
    train_probe_limit: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_rows = bundle.raw_splits["train"][:train_probe_limit]
    if len(train_rows) < train_probe_limit:
        raise ValueError(
            f"Train split contains only {len(train_rows)} rows; expected at least {train_probe_limit} for probe fitting"
        )
    train_recorded = record_story_prompt_rows(model, bundle, train_rows, device)
    sweep_recorded = record_story_prompt_rows(model, bundle, sweep_rows, device)
    candidate_sites = build_final_position_site_list(model)
    recorded_dataset = build_story_site_dataset(model, train_recorded + sweep_recorded, candidate_sites)
    train_mask = (
        (recorded_dataset.metadata["source_kind"] == "dataset")
        & (recorded_dataset.metadata["split"] == "train")
    )
    eval_mask = recorded_dataset.metadata["source_kind"] == "sweep"
    variable_recovery_table = build_variable_recovery_table(
        recorded_dataset,
        sites=candidate_sites,
        variables=STORY_VARIABLE_NAMES,
        train_mask=train_mask,
        eval_mask=eval_mask,
    )
    variable_recovery_rows = variable_recovery_table.copy()
    variable_recovery_rows["row_kind"] = "probe"
    ranking_table = build_site_variable_ranking_table(variable_recovery_table)
    summary_rows: list[dict[str, object]] = []
    for variable in STORY_VARIABLE_NAMES:
        best_row = ranking_table.query("variable == @variable").iloc[0]
        family_table = build_family_variable_stability_table(
            recorded_dataset,
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
    return pd.concat([variable_recovery_rows, pd.DataFrame(summary_rows)], ignore_index=True, sort=False), {
        "candidate_sites": candidate_sites,
        "recording_summary": build_story_recording_summary_table(train_recorded + sweep_recorded).to_dict(orient="records"),
        "sweep_recorded": sweep_recorded,
        "sweep_site_dataset": build_story_site_dataset(model, sweep_recorded, candidate_sites),
        "ranking_table": ranking_table,
    }


def _site_patch_spec(site: str) -> dict[str, Any] | None:
    if site == "final_hidden":
        return None
    block_match = BLOCK_SITE_PATTERN.match(site)
    if block_match is not None:
        layer_index = int(block_match.group("block")) - 1
        kind = block_match.group("kind")
        if kind == "resid_after_mlp":
            return {"mode": "activation_patch", "patch": {"kind": "resid_after_block", "layer_index": layer_index}}
        if kind == "mlp_out":
            return {"mode": "activation_patch", "patch": {"kind": "mlp_out", "layer_index": layer_index}}
    head_match = HEAD_SITE_PATTERN.match(site)
    if head_match is not None:
        layer_index = int(head_match.group("block")) - 1
        head_index = int(head_match.group("head"))
        kind = head_match.group("kind")
        if kind in {"head_out", "resid_contribution"}:
            return {
                "mode": "activation_patch",
                "patch": {"kind": "head_out", "layer_index": layer_index, "head_index": head_index},
            }
        if kind in {"q", "k", "v"}:
            return {
                "mode": "qkv_patch",
                "destination": {"layer_index": layer_index, "head_index": head_index},
                "components": [kind],
            }
    return None


def build_variable_faithfulness_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    variable_scores: pd.DataFrame,
    ranking_table: pd.DataFrame,
    sweep_recorded: list[StoryRecordedPrompt],
    *,
    max_pairs_per_variable: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in STORY_VARIABLE_NAMES:
        variable_rankings = ranking_table[ranking_table["variable"] == variable].reset_index(drop=True)
        supported_site = None
        for _, ranking_row in variable_rankings.iterrows():
            candidate_site = str(ranking_row["site"])
            patch_spec = _site_patch_spec(candidate_site)
            if patch_spec is not None:
                supported_site = (candidate_site, patch_spec)
                break
        best_probe_site = str(variable_scores.query("row_kind == 'summary' and variable == @variable").iloc[0]["best_site"])
        if supported_site is None:
            rows.append(
                {
                    "row_kind": "summary",
                    "variable": variable,
                    "best_probe_site": best_probe_site,
                    "best_patchable_site": "",
                    "patch_mode": "unsupported",
                    "pooled_score": float("nan"),
                    "family_min_score": float("nan"),
                    "rows": 0,
                    "supported": False,
                }
            )
            continue
        site, patch_spec = supported_site
        candidate_pairs: list[tuple[StoryRecordedPrompt, StoryRecordedPrompt]] = []
        for source_recorded, corrupt_recorded in combinations(sweep_recorded, 2):
            source_label = str(getattr(source_recorded.annotation, variable))
            corrupt_label = str(getattr(corrupt_recorded.annotation, variable))
            if source_label == corrupt_label:
                continue
            if source_recorded.annotation.target_token == corrupt_recorded.annotation.target_token:
                continue
            if patch_spec["mode"] == "qkv_patch" and (
                source_recorded.annotation.prompt_length != corrupt_recorded.annotation.prompt_length
            ):
                continue
            candidate_pairs.append((source_recorded, corrupt_recorded))
            candidate_pairs.append((corrupt_recorded, source_recorded))
            if len(candidate_pairs) >= max_pairs_per_variable:
                break
        detail_rows: list[dict[str, object]] = []
        for source_recorded, corrupt_recorded in candidate_pairs[:max_pairs_per_variable]:
            if patch_spec["mode"] == "activation_patch":
                result = score_patched_prompt(
                    model,
                    bundle,
                    clean_prompt=source_recorded.annotation.prompt,
                    corrupt_prompt=corrupt_recorded.annotation.prompt,
                    clean_target=source_recorded.annotation.target_token,
                    device=device,
                    patch=patch_spec["patch"],
                    clean_cache=source_recorded.cache,
                )
            elif patch_spec["mode"] == "qkv_patch":
                result = score_qkv_patched_prompt(
                    model,
                    bundle,
                    clean_prompt=source_recorded.annotation.prompt,
                    corrupt_prompt=corrupt_recorded.annotation.prompt,
                    clean_target=source_recorded.annotation.target_token,
                    device=device,
                    destination=patch_spec["destination"],
                    components=patch_spec["components"],
                    clean_cache=source_recorded.cache,
                    corrupt_cache=corrupt_recorded.cache,
                )
            else:
                raise ValueError(f"Unsupported patch mode {patch_spec['mode']!r}")
            detail_rows.append(
                {
                    "row_kind": "detail",
                    "variable": variable,
                    "site": site,
                    "patch_mode": patch_spec["mode"],
                    "source_prompt_id": source_recorded.annotation.prompt_id,
                    "corrupt_prompt_id": corrupt_recorded.annotation.prompt_id,
                    "source_base_prompt_id": source_recorded.annotation.base_prompt_id,
                    "corrupt_base_prompt_id": corrupt_recorded.annotation.base_prompt_id,
                    "source_label": getattr(source_recorded.annotation, variable),
                    "corrupt_label": getattr(corrupt_recorded.annotation, variable),
                    "source_target": source_recorded.annotation.target_token,
                    "corrupt_target": corrupt_recorded.annotation.target_token,
                    "predicted_token": str(result["predicted_token"]),
                    "answer_follows_source": bool(result["correct"]),
                    "margin": float(result["margin"]),
                }
            )
        detail_table = pd.DataFrame(detail_rows)
        if detail_table.empty:
            rows.append(
                {
                    "row_kind": "summary",
                    "variable": variable,
                    "best_probe_site": best_probe_site,
                    "best_patchable_site": site,
                    "patch_mode": patch_spec["mode"],
                    "pooled_score": float("nan"),
                    "family_min_score": float("nan"),
                    "rows": 0,
                    "supported": True,
                }
            )
            continue
        grouped = detail_table.groupby("corrupt_base_prompt_id")["answer_follows_source"].mean()
        rows.extend(detail_rows)
        rows.append(
            {
                "row_kind": "summary",
                "variable": variable,
                "best_probe_site": best_probe_site,
                "best_patchable_site": site,
                "patch_mode": patch_spec["mode"],
                "pooled_score": float(detail_table["answer_follows_source"].mean()),
                "family_min_score": float(grouped.min()),
                "rows": int(len(detail_table)),
                "supported": True,
            }
        )
    return pd.DataFrame(rows)


def build_head_attention_operator_table(
    recorded_prompts: list[StoryRecordedPrompt],
    *,
    layer_index: int,
    head_index: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recorded in recorded_prompts:
        annotation = recorded.annotation
        attention = recorded.cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, -1, :].detach().cpu()
        top_position = int(attention.argmax().item())
        repeat_attention = (
            float(attention[annotation.last_target_occurrence_position].item())
            if annotation.last_target_occurrence_position is not None
            else float("nan")
        )
        rows.append(
            {
                "prompt_id": annotation.prompt_id,
                "base_prompt_id": annotation.base_prompt_id,
                "family_name": annotation.family_name,
                "family_value": annotation.family_value,
                "previous_token": annotation.previous_token,
                "target_token": annotation.target_token,
                "target_seen_in_context": annotation.target_seen_in_context,
                "repeat_distance_bucket": annotation.repeat_distance_bucket,
                "top_attention_position": top_position,
                "top_attention_token": annotation.prompt_tokens[top_position],
                "top_is_previous_token": top_position == annotation.previous_token_position,
                "top_is_repeat_target": (
                    annotation.last_target_occurrence_position is not None
                    and top_position == annotation.last_target_occurrence_position
                ),
                "previous_token_attention": float(attention[annotation.previous_token_position].item()),
                "repeat_target_attention": repeat_attention,
            }
        )
    return pd.DataFrame(rows)


def build_head_attention_operator_summary_table(attention_table: pd.DataFrame, label: str) -> pd.DataFrame:
    repeat_rows = attention_table[attention_table["target_seen_in_context"] == "yes"].copy()
    return pd.DataFrame(
        [
            {
                "label": label,
                "rows": int(len(attention_table)),
                "previous_token_top_rate": float(attention_table["top_is_previous_token"].mean()),
                "repeat_target_top_rate": float(repeat_rows["top_is_repeat_target"].mean()) if not repeat_rows.empty else 0.0,
                "previous_token_attention_mean": float(attention_table["previous_token_attention"].mean()),
                "repeat_target_attention_mean": float(repeat_rows["repeat_target_attention"].mean()) if not repeat_rows.empty else 0.0,
            }
        ]
    )


def build_head_attention_family_stability_table(attention_table: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base_prompt_id, group in attention_table.groupby("base_prompt_id"):
        repeat_rows = group[group["target_seen_in_context"] == "yes"].copy()
        rows.append(
            {
                "label": label,
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "previous_token_top_rate": float(group["top_is_previous_token"].mean()),
                "repeat_target_top_rate": float(repeat_rows["top_is_repeat_target"].mean()) if not repeat_rows.empty else 0.0,
                "previous_token_attention_mean": float(group["previous_token_attention"].mean()),
                "repeat_target_attention_mean": float(repeat_rows["repeat_target_attention"].mean()) if not repeat_rows.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["repeat_target_top_rate", "previous_token_top_rate", "base_prompt_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_head_copy_rule_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    recorded_prompts: list[StoryRecordedPrompt],
    *,
    layer_index: int,
    head_index: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recorded in recorded_prompts:
        annotation = recorded.annotation
        if annotation.last_target_occurrence_position is None:
            continue
        source_position = annotation.last_target_occurrence_position
        source_logits = ov_source_logits(
            model=model,
            cache=recorded.cache,
            layer_index=layer_index,
            head_index=head_index,
            source_position=source_position,
        )
        top_token_id = int(source_logits.argmax().item())
        sorted_indices = torch.argsort(source_logits, descending=True)
        target_token_id = bundle.token_to_id[annotation.target_token]
        target_rank = int((sorted_indices == target_token_id).nonzero(as_tuple=False)[0].item()) + 1
        attention_weight = float(
            recorded.cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, -1, source_position].item()
        )
        rows.append(
            {
                "layer_index": layer_index,
                "head_index": head_index,
                "prompt_id": annotation.prompt_id,
                "base_prompt_id": annotation.base_prompt_id,
                "family_name": annotation.family_name,
                "family_value": annotation.family_value,
                "target_token": annotation.target_token,
                "source_position": source_position,
                "source_attention": attention_weight,
                "top_written_token": decode_token(top_token_id, bundle.id_to_token),
                "target_rank_among_vocab": target_rank,
                "target_is_top_written_token": top_token_id == target_token_id,
            }
        )
    return pd.DataFrame(rows)


def build_head_copy_rule_summary_table(copy_rule_table: pd.DataFrame) -> pd.DataFrame:
    if copy_rule_table.empty:
        return pd.DataFrame(
            [{"rows": 0, "target_top_written_rate": 0.0, "target_rank_mean": float("inf"), "source_attention_mean": 0.0}]
        )
    return pd.DataFrame(
        [
            {
                "rows": int(len(copy_rule_table)),
                "target_top_written_rate": float(copy_rule_table["target_is_top_written_token"].mean()),
                "target_rank_mean": float(copy_rule_table["target_rank_among_vocab"].mean()),
                "source_attention_mean": float(copy_rule_table["source_attention"].mean()),
            }
        ]
    )


def build_head_copy_rule_family_stability_table(copy_rule_table: pd.DataFrame) -> pd.DataFrame:
    if copy_rule_table.empty:
        return pd.DataFrame(columns=["base_prompt_id", "rows", "target_top_written_rate", "target_rank_mean", "source_attention_mean"])
    rows: list[dict[str, object]] = []
    for base_prompt_id, group in copy_rule_table.groupby("base_prompt_id"):
        rows.append(
            {
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "target_top_written_rate": float(group["target_is_top_written_token"].mean()),
                "target_rank_mean": float(group["target_rank_among_vocab"].mean()),
                "source_attention_mean": float(group["source_attention"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target_top_written_rate", "base_prompt_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def score_all_head_operators(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    sweep_recorded: list[StoryRecordedPrompt],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    head_rows: list[dict[str, object]] = []
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
                "routing_score": float(attention_summary["repeat_target_top_rate"]),
                "routing_family_min_score": (
                    float(attention_family["repeat_target_top_rate"].min()) if not attention_family.empty else 0.0
                ),
                "recency_score": float(attention_summary["previous_token_top_rate"]),
                "recency_family_min_score": (
                    float(attention_family["previous_token_top_rate"].min()) if not attention_family.empty else 0.0
                ),
                "copy_score": float(copy_summary["target_top_written_rate"]) if copy_summary["rows"] > 0 else 0.0,
                "copy_family_min_score": (
                    float(copy_family["target_top_written_rate"].min()) if not copy_family.empty else 0.0
                ),
                "copy_rank_mean": float(copy_summary["target_rank_mean"]) if copy_summary["rows"] > 0 else float("inf"),
            }
            head_rows.append(row)
            if routing_candidate is None or (
                row["routing_score"],
                row["routing_family_min_score"],
                row["recency_score"],
                -row["layer_index"],
                -row["head_index"],
            ) > (
                routing_candidate["routing_score"],
                routing_candidate["routing_family_min_score"],
                routing_candidate["recency_score"],
                -routing_candidate["layer_index"],
                -routing_candidate["head_index"],
            ):
                routing_candidate = dict(row)
            if copy_candidate is None or (
                row["copy_score"],
                row["copy_family_min_score"],
                -row["copy_rank_mean"],
                -row["layer_index"],
                -row["head_index"],
            ) > (
                copy_candidate["copy_score"],
                copy_candidate["copy_family_min_score"],
                -copy_candidate["copy_rank_mean"],
                -copy_candidate["layer_index"],
                -copy_candidate["head_index"],
            ):
                copy_candidate = dict(row)
    if routing_candidate is None or copy_candidate is None:
        raise ValueError("Failed to identify routing and copy candidates")
    return pd.DataFrame(head_rows), routing_candidate, copy_candidate


def build_localization_scores(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    sweep_rows: list[dict[str, object]],
    *,
    device: torch.device,
) -> pd.DataFrame:
    def score_story_rows_with_optional_ablation(
        rows: list[dict[str, object]],
        *,
        ablation: dict[str, int] | None,
    ) -> pd.DataFrame:
        if not rows:
            raise ValueError("Expected at least one row to score.")
        prompt_lengths = {len(str(row["prompt"]).split()) for row in rows}
        if len(prompt_lengths) != 1:
            raise ValueError(f"Expected a fixed prompt length within the batch, found {sorted(prompt_lengths)}")

        input_ids = torch.stack(
            [encode_prompt(str(row["prompt"]), bundle.token_to_id) for row in rows]
        ).to(device)
        target_ids = torch.tensor(
            [bundle.token_to_id[str(row["target"])] for row in rows],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = forward_with_head_ablation(model, input_ids, ablation=ablation)
            final_logits = logits[:, -1, :].detach().cpu()

        predicted_ids = final_logits.argmax(dim=-1)
        target_ids_cpu = target_ids.detach().cpu()
        target_logits = final_logits.gather(1, target_ids_cpu.unsqueeze(1)).squeeze(1)
        competing_logits = final_logits.clone()
        competing_logits.scatter_(1, target_ids_cpu.unsqueeze(1), float("-inf"))
        foil_logits, foil_ids = competing_logits.max(dim=-1)

        records: list[dict[str, object]] = []
        for row_index, row in enumerate(rows):
            predicted_token = decode_token(int(predicted_ids[row_index].item()), bundle.id_to_token)
            target_token = str(row["target"])
            records.append(
                {
                    "prompt_id": str(row.get("id", row["prompt"])),
                    "predicted_token": predicted_token,
                    "target_token": target_token,
                    "foil_token": decode_token(int(foil_ids[row_index].item()), bundle.id_to_token),
                    "target_logit": float(target_logits[row_index].item()),
                    "foil_logit": float(foil_logits[row_index].item()),
                    "margin": float((target_logits[row_index] - foil_logits[row_index]).item()),
                    "correct": predicted_token == target_token,
                }
            )
        return pd.DataFrame(records)

    grouped_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in sweep_rows:
        grouped_rows[len(str(row["prompt"]).split())].append(row)
    baseline_tables = [
        score_story_rows_with_optional_ablation(rows, ablation=None)
        for _, rows in sorted(grouped_rows.items())
    ]
    baseline_table = pd.concat(baseline_tables, ignore_index=True)
    baseline_accuracy = float(baseline_table["correct"].mean())
    baseline_margin = float(baseline_table["margin"].mean())
    ablation_rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        for head_index in range(block.attn.n_heads):
            ablated_tables = [
                score_story_rows_with_optional_ablation(
                    rows,
                    ablation={"layer_index": layer_index, "head_index": head_index},
                )
                for _, rows in sorted(grouped_rows.items())
            ]
            ablated_table = pd.concat(ablated_tables, ignore_index=True)
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
    return pd.DataFrame(ablation_rows)


def build_feature_tracking(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    sweep_rows: list[dict[str, object]],
    *,
    sae_sites: list[str],
    sae_train_limit: int,
    sae_val_limit: int,
    sae_hidden_multiplier: int,
    sae_l1_coeff: float,
    sae_learning_rate: float,
    sae_batch_size: int,
    sae_epochs: int,
    sae_seed: int,
    top_features_per_site: int,
    superposition_cosine_threshold: float,
    device: torch.device,
    battery_dir: Path,
    checkpoint_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_rows = bundle.raw_splits["train"][:sae_train_limit]
    val_rows = bundle.raw_splits["val"][:sae_val_limit]
    if len(train_rows) < sae_train_limit:
        raise ValueError(f"Train split contains only {len(train_rows)} rows; expected {sae_train_limit}")
    if len(val_rows) < sae_val_limit:
        raise ValueError(f"Val split contains only {len(val_rows)} rows; expected {sae_val_limit}")
    train_recorded = record_story_prompt_rows(model, bundle, train_rows, device)
    val_recorded = record_story_prompt_rows(model, bundle, val_rows, device)
    sweep_recorded = record_story_prompt_rows(model, bundle, sweep_rows, device)
    recorded_dataset = build_story_site_dataset(model, train_recorded + val_recorded + sweep_recorded, sae_sites)
    train_mask = (
        (recorded_dataset.metadata["source_kind"] == "dataset")
        & (recorded_dataset.metadata["split"] == "train")
    )
    val_mask = (
        (recorded_dataset.metadata["source_kind"] == "dataset")
        & (recorded_dataset.metadata["split"] == "val")
    )
    sweep_mask = recorded_dataset.metadata["source_kind"] == "sweep"
    train_mask_tensor = torch.tensor(train_mask.to_list(), dtype=torch.bool)
    val_mask_tensor = torch.tensor(val_mask.to_list(), dtype=torch.bool)
    sweep_mask_tensor = torch.tensor(sweep_mask.to_list(), dtype=torch.bool)
    sae_dir = battery_dir / "sae"
    sae_history_dir = battery_dir / "sae_history"
    sae_dir.mkdir(parents=True, exist_ok=True)
    sae_history_dir.mkdir(parents=True, exist_ok=True)
    feature_tables: list[pd.DataFrame] = []
    superposition_rows: list[dict[str, Any]] = []
    for site in sae_sites:
        train_activations = recorded_dataset.site_vectors[site][train_mask_tensor].float()
        val_activations = recorded_dataset.site_vectors[site][val_mask_tensor].float()
        sweep_activations = recorded_dataset.site_vectors[site][sweep_mask_tensor].float()
        hidden_dim = int(train_activations.shape[1] * sae_hidden_multiplier)
        sae, history_table = train_sae(
            train_activations=train_activations,
            val_activations=val_activations,
            hidden_dim=hidden_dim,
            l1_coeff=sae_l1_coeff,
            learning_rate=sae_learning_rate,
            batch_size=sae_batch_size,
            epochs=sae_epochs,
            seed=sae_seed,
        )
        save_sae_checkpoint(
            sae_dir / f"{site}.pt",
            sae,
            metadata={
                "checkpoint_id": checkpoint_id,
                "site": site,
                "train_rows": int(train_activations.shape[0]),
                "val_rows": int(val_activations.shape[0]),
                "sweep_rows": int(sweep_activations.shape[0]),
            },
        )
        history_table.to_csv(sae_history_dir / f"{site}.csv", index=False)
        feature_table = build_feature_score_table(recorded_dataset, site, sae, eval_mask=sweep_mask)
        site_summary = build_feature_site_summary_table(feature_table, top_features_per_site=top_features_per_site)
        feature_tables.extend([feature_table, site_summary])
        superposition_rows.append(
            build_superposition_metrics(
                site,
                sae,
                sweep_activations,
                feature_table,
                history_table,
                cosine_threshold=superposition_cosine_threshold,
            )
        )
    return pd.concat(feature_tables, ignore_index=True, sort=False), superposition_rows


def run_story_checkpoint_battery(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    bundle: DatasetBundle,
    sweep_rows: list[dict[str, object]],
    train_probe_limit: int,
    eval_batch_size: int,
    sae_train_limit: int,
    sae_val_limit: int,
    sae_hidden_multiplier: int,
    sae_l1_coeff: float,
    sae_learning_rate: float,
    sae_batch_size: int,
    sae_epochs: int,
    sae_seed: int,
    top_features_per_site: int,
    superposition_cosine_threshold: float,
    role_top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_payload, model = load_tiny_decoder_checkpoint(checkpoint_path, device)
    checkpoint_id = checkpoint_path.stem
    battery_dir = run_dir / "battery" / checkpoint_id
    battery_dir.mkdir(parents=True, exist_ok=True)
    behavior_artifact = build_behavior_artifact(
        model,
        bundle,
        eval_rows=sweep_rows,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    variable_scores, variable_artifacts = build_variable_scores(
        model,
        bundle,
        sweep_rows=sweep_rows,
        train_probe_limit=train_probe_limit,
        device=device,
    )
    variable_faithfulness = build_variable_faithfulness_table(
        model,
        bundle,
        variable_scores,
        variable_artifacts["ranking_table"],
        variable_artifacts["sweep_recorded"],
        max_pairs_per_variable=64,
        device=device,
    )
    head_scores, routing_candidate, copy_candidate = score_all_head_operators(
        model,
        bundle,
        variable_artifacts["sweep_recorded"],
    )
    candidate_rows = pd.DataFrame(
        [
            {**routing_candidate, "row_kind": "candidate", "candidate_type": "routing"},
            {**copy_candidate, "row_kind": "candidate", "candidate_type": "copy"},
        ]
    )
    operator_scores = pd.concat([head_scores, candidate_rows], ignore_index=True, sort=False)
    localization_scores = build_localization_scores(model, bundle, sweep_rows, device=device)
    weight_metrics = build_weight_metrics(model)
    neuron_scores = build_mlp_neuron_score_table(model, variable_artifacts["sweep_recorded"])
    neuron_scores["row_kind"] = "neuron"
    neuron_top = build_layer_top_neuron_table(neuron_scores)
    neuron_top["row_kind"] = "layer_top"
    sae_sites = build_final_position_site_list(model)
    feature_scores, superposition_metrics = build_feature_tracking(
        model,
        bundle,
        sweep_rows,
        sae_sites=sae_sites,
        sae_train_limit=sae_train_limit,
        sae_val_limit=sae_val_limit,
        sae_hidden_multiplier=sae_hidden_multiplier,
        sae_l1_coeff=sae_l1_coeff,
        sae_learning_rate=sae_learning_rate,
        sae_batch_size=sae_batch_size,
        sae_epochs=sae_epochs,
        sae_seed=sae_seed,
        top_features_per_site=top_features_per_site,
        superposition_cosine_threshold=superposition_cosine_threshold,
        device=device,
        battery_dir=battery_dir,
        checkpoint_id=checkpoint_id,
    )
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
    feature_scores.to_csv(battery_dir / "feature_scores.csv", index=False)
    (battery_dir / "superposition_metrics.json").write_text(json.dumps(superposition_metrics, indent=2), encoding="utf-8")
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
            "routing_candidate": routing_candidate,
            "copy_candidate": copy_candidate,
            "role_top_k": int(role_top_k),
        },
        battery_dir / "tensors.pt",
    )
    return {
        "checkpoint_id": checkpoint_id,
        "routing_candidate": routing_candidate,
        "copy_candidate": copy_candidate,
    }


def _linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_gram = left_centered @ left_centered.T
    right_gram = right_centered @ right_centered.T
    numerator = float((left_gram * right_gram).sum().item())
    left_norm = float(torch.linalg.norm(left_gram).item())
    right_norm = float(torch.linalg.norm(right_gram).item())
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _safe_vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def _rank_operator_heads(operator_scores: pd.DataFrame, score_column: str, top_k: int) -> dict[str, Any]:
    head_scores = operator_scores[operator_scores["row_kind"] == "head_score"].copy()
    ordered = head_scores.sort_values(
        [score_column, f"{score_column.split('_score')[0]}_family_min_score", "layer_index", "head_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    top = ordered.head(top_k)
    leader_gap = 0.0
    if len(ordered) >= 2:
        leader_gap = float(ordered.iloc[0][score_column] - ordered.iloc[1][score_column])
    score_sum = float(top[score_column].sum()) if not top.empty else 0.0
    return {
        "top_heads": top["head_name"].astype(str).tolist(),
        "leader_gap": leader_gap,
        "top_score_share": (
            float(top.iloc[0][score_column] / score_sum) if len(top) > 0 and score_sum > 0.0 else 0.0
        ),
    }


def summarize_story_run(
    *,
    run_dir: Path,
    role_top_k: int,
    behavior_birth_val_accuracy: float,
    variable_birth_score: float,
    variable_family_min_score: float,
    operator_birth_score: float,
    operator_family_min_score: float,
    faithfulness_birth_score: float,
    faithfulness_family_min_score: float,
) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_paths = sorted(checkpoints_dir.glob("*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found under {checkpoints_dir}")
    checkpoint_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    superposition_rows: list[dict[str, Any]] = []
    site_vectors_by_checkpoint: dict[str, dict[str, torch.Tensor]] = {}
    checkpoint_epochs: dict[str, int] = {}
    routing_rank_rows: list[dict[str, Any]] = []
    copy_rank_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_id = checkpoint_path.stem
        epoch = int(payload["epoch"])
        checkpoint_epochs[checkpoint_id] = epoch
        battery_dir = run_dir / "battery" / checkpoint_id
        behavior = json.loads((battery_dir / "behavior.json").read_text(encoding="utf-8"))
        variable_scores = pd.read_csv(battery_dir / "variable_scores.csv")
        variable_faithfulness = pd.read_csv(battery_dir / "variable_faithfulness.csv")
        operator_scores = pd.read_csv(battery_dir / "operator_scores.csv")
        neuron_score_table = pd.read_csv(battery_dir / "neuron_scores.csv")
        feature_score_table = pd.read_csv(battery_dir / "feature_scores.csv")
        superposition_metric_rows = json.loads((battery_dir / "superposition_metrics.json").read_text(encoding="utf-8"))
        canonical_vectors = torch.load(battery_dir / "canonical_site_vectors.pt", map_location="cpu")
        site_vectors_by_checkpoint[checkpoint_id] = canonical_vectors["site_vectors"]

        variable_summary = variable_scores[variable_scores["row_kind"] == "summary"].copy()
        faithfulness_summary = variable_faithfulness[variable_faithfulness["row_kind"] == "summary"].copy()
        operator_candidates = operator_scores[operator_scores["row_kind"] == "candidate"].copy()
        routing_candidate = operator_candidates[operator_candidates["candidate_type"] == "routing"].iloc[0]
        copy_candidate = operator_candidates[operator_candidates["candidate_type"] == "copy"].iloc[0]
        row = {
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "checkpoint_id": checkpoint_id,
            "epoch": epoch,
            "save_reason": str(payload["save_reason"]),
            "val_accuracy": float(behavior["split_metrics"]["val"]["accuracy"]),
            "test_accuracy": float(behavior["split_metrics"]["test"]["accuracy"]),
            "ood_accuracy": float(behavior["split_metrics"]["test_ood_longer_context"]["accuracy"]),
            "routing_candidate": str(routing_candidate["head_name"]),
            "routing_score": float(routing_candidate["routing_score"]),
            "routing_family_min_score": float(routing_candidate["routing_family_min_score"]),
            "copy_candidate": str(copy_candidate["head_name"]),
            "copy_score": float(copy_candidate["copy_score"]),
            "copy_family_min_score": float(copy_candidate["copy_family_min_score"]),
        }
        for variable in STORY_VARIABLE_NAMES:
            variable_row = variable_summary[variable_summary["variable"] == variable].iloc[0]
            faithfulness_row = faithfulness_summary[faithfulness_summary["variable"] == variable].iloc[0]
            row[f"{variable}_site"] = str(variable_row["best_site"])
            row[f"{variable}_score"] = float(variable_row["pooled_score"])
            row[f"{variable}_family_min_score"] = float(variable_row["family_min_score"])
            row[f"{variable}_faithfulness_site"] = str(faithfulness_row["best_patchable_site"])
            row[f"{variable}_faithfulness_score"] = float(faithfulness_row["pooled_score"])
            row[f"{variable}_faithfulness_family_min_score"] = float(faithfulness_row["family_min_score"])
        checkpoint_rows.append(row)

        neuron_epoch_rows = neuron_score_table.copy()
        neuron_epoch_rows["run_id"] = run_dir.name
        neuron_epoch_rows["checkpoint_id"] = checkpoint_id
        neuron_epoch_rows["epoch"] = epoch
        neuron_rows.extend(neuron_epoch_rows.to_dict(orient="records"))

        feature_epoch_rows = feature_score_table.copy()
        feature_epoch_rows["run_id"] = run_dir.name
        feature_epoch_rows["checkpoint_id"] = checkpoint_id
        feature_epoch_rows["epoch"] = epoch
        feature_rows.extend(feature_epoch_rows.to_dict(orient="records"))

        for metric_row in superposition_metric_rows:
            copied = dict(metric_row)
            copied["run_id"] = run_dir.name
            copied["checkpoint_id"] = checkpoint_id
            copied["epoch"] = epoch
            superposition_rows.append(copied)

        routing_rank_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "epoch": epoch,
                "role_name": "routing",
                **_rank_operator_heads(operator_scores, "routing_score", role_top_k),
            }
        )
        copy_rank_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "epoch": epoch,
                "role_name": "copy",
                **_rank_operator_heads(operator_scores, "copy_score", role_top_k),
            }
        )

    checkpoint_table = pd.DataFrame(checkpoint_rows).sort_values(["epoch", "checkpoint_id"]).reset_index(drop=True)
    neuron_table = pd.DataFrame(neuron_rows).sort_values(
        ["epoch", "layer_index", "best_selectivity_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    feature_table = pd.DataFrame(feature_rows).sort_values(
        ["epoch", "site", "row_kind", "best_selectivity_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    superposition_table = pd.DataFrame(superposition_rows).sort_values(
        ["epoch", "site"],
        ascending=[True, True],
    ).reset_index(drop=True)

    emergence_rows: list[dict[str, Any]] = []
    def _birth_epoch(series: pd.Series, epochs: pd.Series, threshold: float) -> float | None:
        mask = series >= threshold
        if not mask.any():
            return None
        return float(epochs[mask].iloc[0])

    emergence_rows.append(
        {
            "run_id": run_dir.name,
            "metric_name": "behavior_val_accuracy",
            "birth_epoch": _birth_epoch(
                checkpoint_table["val_accuracy"],
                checkpoint_table["epoch"],
                behavior_birth_val_accuracy,
            ),
        }
    )
    for variable in STORY_VARIABLE_NAMES:
        variable_mask = (
            (checkpoint_table[f"{variable}_score"] >= variable_birth_score)
            & (checkpoint_table[f"{variable}_family_min_score"] >= variable_family_min_score)
        )
        emergence_rows.append(
            {
                "run_id": run_dir.name,
                "metric_name": f"variable_{variable}",
                "birth_epoch": (
                    float(checkpoint_table.loc[variable_mask, "epoch"].iloc[0]) if variable_mask.any() else None
                ),
            }
        )
        faithfulness_mask = (
            (checkpoint_table[f"{variable}_faithfulness_score"] >= faithfulness_birth_score)
            & (checkpoint_table[f"{variable}_faithfulness_family_min_score"] >= faithfulness_family_min_score)
        )
        emergence_rows.append(
            {
                "run_id": run_dir.name,
                "metric_name": f"faithfulness_{variable}",
                "birth_epoch": (
                    float(checkpoint_table.loc[faithfulness_mask, "epoch"].iloc[0]) if faithfulness_mask.any() else None
                ),
            }
        )
    routing_mask = (
        (checkpoint_table["routing_score"] >= operator_birth_score)
        & (checkpoint_table["routing_family_min_score"] >= operator_family_min_score)
    )
    copy_mask = (
        (checkpoint_table["copy_score"] >= operator_birth_score)
        & (checkpoint_table["copy_family_min_score"] >= operator_family_min_score)
    )
    emergence_rows.append(
        {
            "run_id": run_dir.name,
            "metric_name": "operator_routing",
            "birth_epoch": float(checkpoint_table.loc[routing_mask, "epoch"].iloc[0]) if routing_mask.any() else None,
        }
    )
    emergence_rows.append(
        {
            "run_id": run_dir.name,
            "metric_name": "operator_copy",
            "birth_epoch": float(checkpoint_table.loc[copy_mask, "epoch"].iloc[0]) if copy_mask.any() else None,
        }
    )
    emergence_table = pd.DataFrame(emergence_rows)

    representation_rows: list[dict[str, Any]] = []
    checkpoint_ids = checkpoint_table["checkpoint_id"].tolist()
    for left_index, checkpoint_id_left in enumerate(checkpoint_ids):
        left_vectors = site_vectors_by_checkpoint[checkpoint_id_left]
        epoch_left = checkpoint_epochs[checkpoint_id_left]
        for right_index in range(left_index + 1, len(checkpoint_ids)):
            checkpoint_id_right = checkpoint_ids[right_index]
            right_vectors = site_vectors_by_checkpoint[checkpoint_id_right]
            epoch_right = checkpoint_epochs[checkpoint_id_right]
            common_sites = sorted(set(left_vectors) & set(right_vectors))
            for site in common_sites:
                left_tensor = left_vectors[site].float()
                right_tensor = right_vectors[site].float()
                representation_rows.append(
                    {
                        "run_id": run_dir.name,
                        "site": site,
                        "checkpoint_id_left": checkpoint_id_left,
                        "epoch_left": epoch_left,
                        "checkpoint_id_right": checkpoint_id_right,
                        "epoch_right": epoch_right,
                        "epoch_gap": epoch_right - epoch_left,
                        "is_adjacent_pair": right_index == left_index + 1,
                        "is_origin_pair": left_index == 0,
                        "is_final_pair": right_index == len(checkpoint_ids) - 1,
                        "linear_cka": _linear_cka(left_tensor, right_tensor),
                        "mean_vector_cosine": _safe_vector_cosine(left_tensor.mean(dim=0), right_tensor.mean(dim=0)),
                        "relative_frobenius_shift": float(
                            (right_tensor - left_tensor).norm().item() / max(left_tensor.norm().item(), 1e-8)
                        ),
                    }
                )
    representation_drift_table = pd.DataFrame(representation_rows).sort_values(
        ["site", "epoch_left", "epoch_right"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    operator_handoff_rows: list[dict[str, Any]] = []
    for rank_rows in [routing_rank_rows, copy_rank_rows]:
        ordered = sorted(rank_rows, key=lambda row: (row["epoch"], row["checkpoint_id"]))
        for previous_row, current_row in zip(ordered[:-1], ordered[1:], strict=True):
            previous_heads = list(previous_row["top_heads"])
            current_heads = list(current_row["top_heads"])
            overlap = len(set(previous_heads) & set(current_heads))
            operator_handoff_rows.append(
                {
                    "run_id": run_dir.name,
                    "role_name": previous_row["role_name"],
                    "checkpoint_id_previous": previous_row["checkpoint_id"],
                    "epoch_previous": previous_row["epoch"],
                    "checkpoint_id_current": current_row["checkpoint_id"],
                    "epoch_current": current_row["epoch"],
                    "previous_candidate": previous_heads[0] if previous_heads else "",
                    "current_candidate": current_heads[0] if current_heads else "",
                    "candidate_changed": (previous_heads[0] if previous_heads else "") != (current_heads[0] if current_heads else ""),
                    "top_k_overlap_count": overlap,
                    "top_k_overlap_fraction": float(overlap / role_top_k),
                    "previous_top_heads": ",".join(previous_heads),
                    "current_top_heads": ",".join(current_heads),
                    "previous_leader_gap": float(previous_row["leader_gap"]),
                    "current_leader_gap": float(current_row["leader_gap"]),
                    "previous_top_k_score_share": float(previous_row["top_score_share"]),
                    "current_top_k_score_share": float(current_row["top_score_share"]),
                    "previous_candidate_rank_current": (
                        current_heads.index(previous_heads[0]) + 1 if previous_heads and previous_heads[0] in current_heads else role_top_k + 1
                    ),
                    "current_candidate_rank_previous": (
                        previous_heads.index(current_heads[0]) + 1 if current_heads and current_heads[0] in previous_heads else role_top_k + 1
                    ),
                }
            )
    operator_handoff_table = pd.DataFrame(operator_handoff_rows).sort_values(
        ["role_name", "epoch_previous", "epoch_current"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    summary_dir = run_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_table.to_csv(summary_dir / "checkpoint_index.csv", index=False)
    emergence_table.to_csv(summary_dir / "emergence.csv", index=False)
    neuron_table.to_csv(summary_dir / "neuron_dynamics.csv", index=False)
    feature_table.to_csv(summary_dir / "feature_dynamics.csv", index=False)
    superposition_table.to_csv(summary_dir / "superposition_dynamics.csv", index=False)
    representation_drift_table.to_csv(summary_dir / "representation_drift.csv", index=False)
    operator_handoff_table.to_csv(summary_dir / "operator_handoffs.csv", index=False)
    pd.DataFrame().to_csv(summary_dir / "role_matching.csv", index=False)
    pd.DataFrame().to_csv(summary_dir / "clamp_responsiveness.csv", index=False)
    run_summary = {
        "target_dir": str(run_dir),
        "num_runs": 1,
        "num_checkpoints": int(len(checkpoint_table)),
        "num_neuron_rows": int(len(neuron_table)),
        "num_feature_rows": int(len(feature_table)),
        "num_superposition_rows": int(len(superposition_table)),
        "num_representation_drift_rows": int(len(representation_drift_table)),
        "num_operator_handoff_rows": int(len(operator_handoff_table)),
        "behavior_birth_runs": int(emergence_table[emergence_table["metric_name"] == "behavior_val_accuracy"]["birth_epoch"].notna().sum()),
        "routing_birth_runs": int(emergence_table[emergence_table["metric_name"] == "operator_routing"]["birth_epoch"].notna().sum()),
        "copy_birth_runs": int(emergence_table[emergence_table["metric_name"] == "operator_copy"]["birth_epoch"].notna().sum()),
    }
    for variable in STORY_VARIABLE_NAMES:
        run_summary[f"{variable}_faithfulness_birth_runs"] = int(
            emergence_table[emergence_table["metric_name"] == f"faithfulness_{variable}"]["birth_epoch"].notna().sum()
        )
    (summary_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
