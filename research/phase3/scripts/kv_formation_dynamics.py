from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from scripts.kv_algorithm_oracle import annotate_row
from scripts.kv_algorithm_sweeps import (
    generate_query_key_sweep,
    generate_slot_permutation_sweep,
    generate_value_permutation_sweep,
)
from scripts.kv_retrieve_analysis import DatasetBundle, encode_prompt, head_residual_contribution, ov_source_logits
from scripts.tiny_transformer_core import TinyDecoderTransformer, forward_tiny_decoder_with_interventions
from scripts.training_dynamics import FormationConfig, FormationHeadConfig, RunManifest, append_jsonl


FORMATION_FAMILY_ORDER = ["original", "query_swap", "slot_permutation", "value_permutation"]


@dataclass
class FormationRunContext:
    config: FormationConfig
    history_path: Path
    family_rows: dict[str, list[dict[str, object]]]
    boost_steps_remaining: int = 0
    last_transition_metric_value: float | None = None


def _head_name(ref: FormationHeadConfig | None) -> str | None:
    if ref is None:
        return None
    return f"block{ref.layer_index + 1}_head{ref.head_index}"


def _head_dict(ref: FormationHeadConfig | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "layer_index": int(ref.layer_index),
        "head_index": int(ref.head_index),
        "name": _head_name(ref),
    }


def _component_key(ref: FormationHeadConfig, component: str) -> str:
    return f"block{ref.layer_index + 1}.head{ref.head_index}.{component}"


def _head_component_tensor(model: TinyDecoderTransformer, ref: FormationHeadConfig, component: str) -> torch.Tensor:
    block = model.blocks[ref.layer_index]
    head_dim = block.attn.head_dim
    head_start = ref.head_index * head_dim
    head_stop = head_start + head_dim
    if component == "q_proj_slice":
        return block.attn.q_proj.weight[head_start:head_stop, :]
    if component == "k_proj_slice":
        return block.attn.k_proj.weight[head_start:head_stop, :]
    if component == "v_proj_slice":
        return block.attn.v_proj.weight[head_start:head_stop, :]
    if component == "o_proj_slice":
        return block.attn.o_proj.weight[:, head_start:head_stop]
    raise ValueError(f"Unsupported head component {component!r}")


def _head_component_grad(full_grad: torch.Tensor, ref: FormationHeadConfig, component: str, model: TinyDecoderTransformer) -> torch.Tensor:
    head_dim = model.blocks[ref.layer_index].attn.head_dim
    head_start = ref.head_index * head_dim
    head_stop = head_start + head_dim
    if component in {"q_proj_slice", "k_proj_slice", "v_proj_slice"}:
        return full_grad[head_start:head_stop, :]
    if component == "o_proj_slice":
        return full_grad[:, head_start:head_stop]
    raise ValueError(f"Unsupported head component {component!r}")


def _head_components() -> list[str]:
    return ["q_proj_slice", "k_proj_slice", "v_proj_slice", "o_proj_slice"]


def _flatten_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1).cpu()


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = _flatten_cpu_tensor(left)
    right_flat = _flatten_cpu_tensor(right)
    left_norm = float(left_flat.norm().item())
    right_norm = float(right_flat.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).item() / (left_norm * right_norm))


def _make_original_row(base_row: dict[str, object]) -> dict[str, object]:
    base_prompt_id = str(base_row.get("id") or base_row.get("prompt_id") or base_row["prompt"])
    return {
        **base_row,
        "id": f"{base_prompt_id}::original",
        "base_prompt_id": base_prompt_id,
        "source_kind": "formation_eval",
        "family_name": "original",
        "family_value": "original",
    }


def _select_query_swap_row(base_row: dict[str, object]) -> dict[str, object]:
    annotation = annotate_row(base_row)
    for candidate in generate_query_key_sweep(base_row):
        if str(candidate["query_key"]) != annotation.query_key:
            candidate["source_kind"] = "formation_eval"
            return candidate
    raise ValueError(f"Could not construct query_swap family row for base prompt {annotation.prompt_id}")


def _select_first_non_identity(rows: list[dict[str, object]], *, base_prompt: str, family_name: str) -> dict[str, object]:
    for candidate in rows:
        if str(candidate["prompt"]) != base_prompt:
            candidate["source_kind"] = "formation_eval"
            return candidate
    raise ValueError(f"Could not construct a non-identity {family_name} family row")


def build_formation_eval_rows(
    manifest: RunManifest,
    bundle: DatasetBundle,
) -> dict[str, list[dict[str, object]]]:
    base_rows = bundle.raw_splits[manifest.dataset.sweep_base_split][: manifest.formation.eval_pack_size]
    if len(base_rows) < manifest.formation.eval_pack_size:
        raise ValueError(
            f"Formation eval pack requires {manifest.formation.eval_pack_size} rows from "
            f"{manifest.dataset.sweep_base_split!r}, found {len(base_rows)}"
        )
    family_rows: dict[str, list[dict[str, object]]] = {family_name: [] for family_name in FORMATION_FAMILY_ORDER}
    for base_row in base_rows:
        base_prompt = str(base_row["prompt"])
        family_rows["original"].append(_make_original_row(base_row))
        family_rows["query_swap"].append(_select_query_swap_row(base_row))
        family_rows["slot_permutation"].append(
            _select_first_non_identity(
                generate_slot_permutation_sweep(base_row),
                base_prompt=base_prompt,
                family_name="slot_permutation",
            )
        )
        family_rows["value_permutation"].append(
            _select_first_non_identity(
                generate_value_permutation_sweep(base_row),
                base_prompt=base_prompt,
                family_name="value_permutation",
            )
        )
    return family_rows


def build_formation_context(
    manifest: RunManifest,
    bundle: DatasetBundle,
    *,
    history_path: Path,
) -> FormationRunContext | None:
    if not manifest.formation.enabled:
        return None
    return FormationRunContext(
        config=manifest.formation,
        history_path=history_path,
        family_rows=build_formation_eval_rows(manifest, bundle),
    )


def _encode_rows(rows: list[dict[str, object]], bundle: DatasetBundle, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("Expected at least one formation row")
    prompt_lengths = {len(str(row["prompt"]).split()) for row in rows}
    if len(prompt_lengths) != 1:
        raise ValueError(f"Expected a fixed prompt length for formation rows, found {sorted(prompt_lengths)}")
    input_ids = torch.stack([encode_prompt(str(row["prompt"]), bundle.token_to_id) for row in rows]).to(device)
    target_ids = torch.tensor([bundle.token_to_id[str(row["target"])] for row in rows], dtype=torch.long, device=device)
    return input_ids, target_ids


def _run_rows_with_cache(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    *,
    device: torch.device,
    interventions: list[dict[str, object]] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[Any]]:
    input_ids, _target_ids = _encode_rows(rows, bundle, device)
    with torch.no_grad():
        if interventions:
            logits, cache = forward_tiny_decoder_with_interventions(
                model,
                input_ids,
                interventions,
                return_cache=True,
            )
        else:
            logits, cache = model(input_ids, return_cache=True)
    annotations = [annotate_row(row) for row in rows]
    return logits.detach().cpu(), cache, annotations


def _family_loss_accuracy(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    *,
    device: torch.device,
) -> dict[str, float]:
    input_ids, target_ids = _encode_rows(rows, bundle, device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, target_ids)
        predictions = final_logits.argmax(dim=-1)
    return {
        "loss": float(loss.item()),
        "accuracy": float((predictions == target_ids).float().mean().item()),
    }


def _compute_q_metrics(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    cache: dict[str, Any],
    annotations: list[Any],
    support_ref: FormationHeadConfig,
) -> dict[str, float]:
    site_vectors = cache["blocks"][support_ref.layer_index]["resid_after_mlp"][:, -1, :]
    token_embed = model.token_embed.weight.detach().cpu()
    margins: list[float] = []
    accuracies: list[float] = []
    for row_index, annotation in enumerate(annotations):
        query_id = bundle.token_to_id[annotation.query_key]
        context_key_ids = [bundle.token_to_id[key] for key in annotation.slot_keys]
        scores = torch.matmul(token_embed[context_key_ids], site_vectors[row_index])
        top_key_index = int(scores.argmax().item())
        predicted_key = annotation.slot_keys[top_key_index]
        query_score = float(torch.dot(site_vectors[row_index], token_embed[query_id]).item())
        foil_scores = [float(torch.dot(site_vectors[row_index], token_embed[token_id]).item()) for token_id in context_key_ids if token_id != query_id]
        if not foil_scores:
            raise ValueError("Query metric requires at least one foil key")
        margins.append(query_score - max(foil_scores))
        accuracies.append(1.0 if predicted_key == annotation.query_key else 0.0)
    return {
        "margin_mean": float(sum(margins) / len(margins)),
        "top_key_accuracy": float(sum(accuracies) / len(accuracies)),
    }


def _slot_total_scores(pattern: torch.Tensor, annotation: Any) -> list[float]:
    scores: list[float] = []
    for slot_index in range(annotation.num_pairs):
        scores.append(
            float(
                pattern[annotation.key_positions[slot_index]].item()
                + pattern[annotation.value_positions[slot_index]].item()
            )
        )
    return scores


def _compute_r_metrics(
    cache: dict[str, Any],
    annotations: list[Any],
    retrieval_ref: FormationHeadConfig,
) -> dict[str, float]:
    patterns = cache["blocks"][retrieval_ref.layer_index]["attention"]["pattern"][:, retrieval_ref.head_index, -1, :]
    margins: list[float] = []
    accuracies: list[float] = []
    correct_slot_attention: list[float] = []
    entropies: list[float] = []
    for row_index, annotation in enumerate(annotations):
        pattern = patterns[row_index]
        slot_scores = _slot_total_scores(pattern, annotation)
        matching_slot = int(annotation.matching_slot)
        matching_score = slot_scores[matching_slot]
        foil_scores = [score for slot_index, score in enumerate(slot_scores) if slot_index != matching_slot]
        if not foil_scores:
            raise ValueError("Routing metric requires at least one foil slot")
        top_slot = max(range(len(slot_scores)), key=lambda slot_index: slot_scores[slot_index])
        pattern_clamped = pattern.clamp_min(1e-12)
        entropies.append(float((-(pattern_clamped * pattern_clamped.log()).sum()).item()))
        margins.append(float(matching_score - max(foil_scores)))
        accuracies.append(1.0 if top_slot == matching_slot else 0.0)
        correct_slot_attention.append(float(matching_score))
    return {
        "slot_margin_mean": float(sum(margins) / len(margins)),
        "top_slot_accuracy": float(sum(accuracies) / len(accuracies)),
        "correct_slot_attention_mean": float(sum(correct_slot_attention) / len(correct_slot_attention)),
        "attention_entropy_mean": float(sum(entropies) / len(entropies)),
    }


def _compute_w_metrics(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    cache: dict[str, Any],
    annotations: list[Any],
    retrieval_ref: FormationHeadConfig,
) -> dict[str, float]:
    value_token_ids = [bundle.token_to_id[token] for token in bundle.value_tokens]
    margins: list[float] = []
    accuracies: list[float] = []
    for annotation in annotations:
        selected_value_id = bundle.token_to_id[annotation.selected_value]
        source_logits = ov_source_logits(
            model,
            cache,
            layer_index=retrieval_ref.layer_index,
            head_index=retrieval_ref.head_index,
            source_position=annotation.matching_value_position,
        )
        value_logits = source_logits[value_token_ids]
        top_value_index = int(value_logits.argmax().item())
        predicted_value = bundle.value_tokens[top_value_index]
        selected_offset = bundle.value_tokens.index(annotation.selected_value)
        selected_logit = float(value_logits[selected_offset].item())
        foil_logits = [float(value_logits[offset].item()) for offset in range(len(bundle.value_tokens)) if offset != selected_offset]
        if not foil_logits:
            raise ValueError("Write metric requires at least one foil value")
        margins.append(selected_logit - max(foil_logits))
        accuracies.append(1.0 if predicted_value == annotation.selected_value else 0.0)
        _ = selected_value_id
    return {
        "value_margin_mean": float(sum(margins) / len(margins)),
        "top_written_value_accuracy": float(sum(accuracies) / len(accuracies)),
    }


def _build_head_role_scores(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    cache: dict[str, Any],
    annotations: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_index, block in enumerate(model.blocks):
        for head_index in range(block.attn.n_heads):
            ref = FormationHeadConfig(layer_index=layer_index, head_index=head_index)
            routing = _compute_r_metrics(cache, annotations, ref)
            write = _compute_w_metrics(model, bundle, cache, annotations, ref)
            rows.append(
                {
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "head_name": _head_name(ref),
                    "top_slot_accuracy": routing["top_slot_accuracy"],
                    "slot_margin_mean": routing["slot_margin_mean"],
                    "correct_slot_attention_mean": routing["correct_slot_attention_mean"],
                    "top_written_value_accuracy": write["top_written_value_accuracy"],
                    "value_margin_mean": write["value_margin_mean"],
                }
            )
    return rows


def _select_retrieval_candidate(head_scores: list[dict[str, Any]]) -> FormationHeadConfig:
    best = max(
        head_scores,
        key=lambda row: (
            float(row["top_slot_accuracy"]),
            float(row["slot_margin_mean"]),
            float(row["top_written_value_accuracy"]),
            float(row["value_margin_mean"]),
            -int(row["layer_index"]),
            -int(row["head_index"]),
        ),
    )
    return FormationHeadConfig(layer_index=int(best["layer_index"]), head_index=int(best["head_index"]))


def _support_path_gain(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    *,
    device: torch.device,
    support_ref: FormationHeadConfig,
    retrieval_ref: FormationHeadConfig,
    base_routing_metrics: dict[str, float],
) -> dict[str, float]:
    _, ablated_cache, annotations = _run_rows_with_cache(
        model,
        bundle,
        rows,
        device=device,
        interventions=[
            {
                "name": f"{_head_name(support_ref)}_zero",
                "kind": "head_resid_final_scale",
                "layer_index": support_ref.layer_index,
                "head_index": support_ref.head_index,
                "scale": 0.0,
                "position": "final",
            }
        ],
    )
    ablated_routing = _compute_r_metrics(ablated_cache, annotations, retrieval_ref)
    return {
        "routing_margin_delta": float(base_routing_metrics["slot_margin_mean"] - ablated_routing["slot_margin_mean"]),
        "correct_slot_attention_delta": float(
            base_routing_metrics["correct_slot_attention_mean"] - ablated_routing["correct_slot_attention_mean"]
        ),
        "top_slot_accuracy_delta": float(base_routing_metrics["top_slot_accuracy"] - ablated_routing["top_slot_accuracy"]),
    }


def _head_update_norm(model: TinyDecoderTransformer, ref: FormationHeadConfig, pre_step_snapshot: dict[str, torch.Tensor] | None) -> float:
    if pre_step_snapshot is None:
        return float("nan")
    pieces: list[torch.Tensor] = []
    for component in _head_components():
        current_tensor = _head_component_tensor(model, ref, component).detach().float().cpu()
        previous_tensor = pre_step_snapshot[_component_key(ref, component)]
        pieces.append((current_tensor - previous_tensor).reshape(-1))
    return float(torch.cat(pieces).norm().item())


def _select_support_and_placebo_candidates(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    *,
    device: torch.device,
    retrieval_ref: FormationHeadConfig,
    head_scores: list[dict[str, Any]],
    base_routing_metrics: dict[str, float],
    pre_step_snapshot: dict[str, torch.Tensor] | None,
) -> tuple[FormationHeadConfig, FormationHeadConfig | None, list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    for score_row in head_scores:
        candidate_ref = FormationHeadConfig(
            layer_index=int(score_row["layer_index"]),
            head_index=int(score_row["head_index"]),
        )
        if candidate_ref == retrieval_ref:
            continue
        path_gain = _support_path_gain(
            model,
            bundle,
            rows,
            device=device,
            support_ref=candidate_ref,
            retrieval_ref=retrieval_ref,
            base_routing_metrics=base_routing_metrics,
        )
        candidate_rows.append(
            {
                **score_row,
                "path_gain_routing_margin_delta": path_gain["routing_margin_delta"],
                "path_gain_correct_slot_attention_delta": path_gain["correct_slot_attention_delta"],
                "path_gain_top_slot_accuracy_delta": path_gain["top_slot_accuracy_delta"],
                "update_norm": _head_update_norm(model, candidate_ref, pre_step_snapshot),
            }
        )
    if not candidate_rows:
        raise ValueError("Support discovery requires at least one non-retrieval head")
    support_row = max(
        candidate_rows,
        key=lambda row: (
            float(row["path_gain_routing_margin_delta"]),
            float(row["path_gain_correct_slot_attention_delta"]),
            float(row["top_slot_accuracy"]),
            float(row["value_margin_mean"]),
        ),
    )
    support_ref = FormationHeadConfig(layer_index=int(support_row["layer_index"]), head_index=int(support_row["head_index"]))

    remaining_rows = [
        row
        for row in candidate_rows
        if not (int(row["layer_index"]) == support_ref.layer_index and int(row["head_index"]) == support_ref.head_index)
    ]
    placebo_ref: FormationHeadConfig | None = None
    if remaining_rows:
        support_update_norm = float(support_row["update_norm"])
        placebo_row = min(
            remaining_rows,
            key=lambda row: (
                abs(float(row["path_gain_routing_margin_delta"])) + float(row["top_slot_accuracy"]) + float(row["top_written_value_accuracy"]),
                abs(float(row["update_norm"]) - support_update_norm) if not math.isnan(support_update_norm) and not math.isnan(float(row["update_norm"])) else float("inf"),
                int(row["layer_index"]),
                int(row["head_index"]),
            ),
        )
        placebo_ref = FormationHeadConfig(layer_index=int(placebo_row["layer_index"]), head_index=int(placebo_row["head_index"]))
    return support_ref, placebo_ref, candidate_rows


def _resolve_candidates(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    *,
    manifest: RunManifest,
    device: torch.device,
    pre_step_snapshot: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, FormationHeadConfig | None], dict[str, Any]]:
    if manifest.formation.candidate_mode == "fixed":
        candidates = {
            "support": manifest.formation.candidate_support_head,
            "retrieval": manifest.formation.candidate_retrieval_head,
            "placebo": manifest.formation.candidate_placebo_head,
        }
        return candidates, {"head_scores": [], "support_candidates": []}

    _logits, cache, annotations = _run_rows_with_cache(model, bundle, rows, device=device)
    head_scores = _build_head_role_scores(model, bundle, cache, annotations)
    retrieval_ref = _select_retrieval_candidate(head_scores)
    base_routing = _compute_r_metrics(cache, annotations, retrieval_ref)
    support_ref, placebo_ref, support_candidates = _select_support_and_placebo_candidates(
        model,
        bundle,
        rows,
        device=device,
        retrieval_ref=retrieval_ref,
        head_scores=head_scores,
        base_routing_metrics=base_routing,
        pre_step_snapshot=pre_step_snapshot,
    )
    return {
        "support": support_ref,
        "retrieval": retrieval_ref,
        "placebo": placebo_ref,
    }, {
        "head_scores": head_scores,
        "support_candidates": support_candidates,
    }


def _role_update_vector(
    model: TinyDecoderTransformer,
    ref: FormationHeadConfig,
    pre_step_snapshot: dict[str, torch.Tensor],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for component in _head_components():
        previous = pre_step_snapshot[_component_key(ref, component)]
        current = _head_component_tensor(model, ref, component).detach().float().cpu()
        pieces.append((current - previous).reshape(-1))
    return torch.cat(pieces)


def _get_optimizer_group_for_param(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> dict[str, Any]:
    for group in optimizer.param_groups:
        if any(candidate is param for candidate in group["params"]):
            return group
    raise ValueError("Could not locate optimizer parameter group for tracked tensor")


def _slice_optimizer_state(
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
    ref: FormationHeadConfig,
    component: str,
    model: TinyDecoderTransformer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = optimizer.state.get(param)
    if not state or "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
        slice_shape = _head_component_tensor(model, ref, component).shape
        zeros = torch.zeros(slice_shape, dtype=torch.float32)
        return zeros, zeros, zeros
    exp_avg = _head_component_grad(state["exp_avg"].detach().float().cpu(), ref, component, model)
    exp_avg_sq = _head_component_grad(state["exp_avg_sq"].detach().float().cpu(), ref, component, model)
    group = _get_optimizer_group_for_param(optimizer, param)
    beta1, beta2 = group["betas"]
    step_value = int(state["step"])
    bias_correction1 = 1.0 - (beta1 ** step_value)
    bias_correction2 = 1.0 - (beta2 ** step_value)
    if bias_correction1 == 0.0 or bias_correction2 == 0.0:
        effective_step = torch.zeros_like(exp_avg)
    else:
        exp_avg_hat = exp_avg / bias_correction1
        exp_avg_sq_hat = exp_avg_sq / bias_correction2
        effective_step = (
            -float(group["lr"]) * exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + float(group["eps"]))
        )
    return exp_avg, exp_avg_sq, effective_step


def _role_optimizer_metrics(
    model: TinyDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    ref: FormationHeadConfig,
    update_vector: torch.Tensor | None,
) -> dict[str, Any]:
    block = model.blocks[ref.layer_index]
    param_map = {
        "q_proj_slice": block.attn.q_proj.weight,
        "k_proj_slice": block.attn.k_proj.weight,
        "v_proj_slice": block.attn.v_proj.weight,
        "o_proj_slice": block.attn.o_proj.weight,
    }
    exp_avg_pieces: list[torch.Tensor] = []
    exp_avg_sq_pieces: list[torch.Tensor] = []
    effective_step_pieces: list[torch.Tensor] = []
    for component, param in param_map.items():
        exp_avg, exp_avg_sq, effective_step = _slice_optimizer_state(optimizer, param, ref, component, model)
        exp_avg_pieces.append(exp_avg.reshape(-1))
        exp_avg_sq_pieces.append(exp_avg_sq.reshape(-1))
        effective_step_pieces.append(effective_step.reshape(-1))
    exp_avg_vector = torch.cat(exp_avg_pieces)
    exp_avg_sq_vector = torch.cat(exp_avg_sq_pieces)
    effective_step_vector = torch.cat(effective_step_pieces)
    return {
        "role_name": None,
        "head_name": _head_name(ref),
        "exp_avg_norm": float(exp_avg_vector.norm().item()),
        "exp_avg_sq_mean": float(exp_avg_sq_vector.mean().item()),
        "effective_step_norm": float(effective_step_vector.norm().item()),
        "effective_step_cosine_to_update": (
            _safe_cosine(effective_step_vector, update_vector)
            if update_vector is not None
            else float("nan")
        ),
        "effective_step_vector": effective_step_vector,
    }


def _family_gradients_for_roles(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    family_rows: dict[str, list[dict[str, object]]],
    *,
    device: torch.device,
    roles: dict[str, FormationHeadConfig | None],
    update_vectors: dict[str, torch.Tensor],
    optimizer_metrics: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tracked_roles = {role_name: ref for role_name, ref in roles.items() if ref is not None}
    if not tracked_roles:
        return [], [], []

    parameter_order: list[tuple[str, torch.nn.Parameter]] = []
    seen_parameter_ids: set[int] = set()
    for ref in tracked_roles.values():
        block = model.blocks[ref.layer_index]
        for component, param in (
            ("q_proj_slice", block.attn.q_proj.weight),
            ("k_proj_slice", block.attn.k_proj.weight),
            ("v_proj_slice", block.attn.v_proj.weight),
            ("o_proj_slice", block.attn.o_proj.weight),
        ):
            if id(param) in seen_parameter_ids:
                continue
            seen_parameter_ids.add(id(param))
            parameter_order.append((component, param))
    unique_parameters = [param for _, param in parameter_order]

    family_role_rows: list[dict[str, Any]] = []
    family_cosine_rows: list[dict[str, Any]] = []
    role_pair_rows: list[dict[str, Any]] = []
    role_family_vectors: dict[tuple[str, str], torch.Tensor] = {}

    model.eval()
    for family_name in FORMATION_FAMILY_ORDER:
        rows = family_rows[family_name]
        input_ids, target_ids = _encode_rows(rows, bundle, device)
        logits = model(input_ids)
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, target_ids)
        predictions = final_logits.argmax(dim=-1)
        gradients = torch.autograd.grad(loss, unique_parameters, retain_graph=False, allow_unused=False)
        gradient_by_parameter = {id(param): grad.detach().float().cpu() for param, grad in zip(unique_parameters, gradients, strict=True)}

        for role_name, ref in tracked_roles.items():
            component_vectors: list[torch.Tensor] = []
            component_norms: dict[str, float] = {}
            for component in _head_components():
                block = model.blocks[ref.layer_index]
                if component == "q_proj_slice":
                    parameter = block.attn.q_proj.weight
                elif component == "k_proj_slice":
                    parameter = block.attn.k_proj.weight
                elif component == "v_proj_slice":
                    parameter = block.attn.v_proj.weight
                elif component == "o_proj_slice":
                    parameter = block.attn.o_proj.weight
                else:
                    raise ValueError(f"Unsupported head component {component!r}")
                grad_tensor = _head_component_grad(gradient_by_parameter[id(parameter)], ref, component, model)
                component_vectors.append(grad_tensor.reshape(-1))
                component_norms[f"{component}_grad_norm"] = float(grad_tensor.norm().item())
            role_gradient_vector = torch.cat(component_vectors)
            role_family_vectors[(role_name, family_name)] = role_gradient_vector
            optimizer_entry = optimizer_metrics.get(role_name)
            effective_step_vector = None if optimizer_entry is None else optimizer_entry["effective_step_vector"]
            family_role_rows.append(
                {
                    "role_name": role_name,
                    "head_name": _head_name(ref),
                    "family_name": family_name,
                    "loss": float(loss.item()),
                    "accuracy": float((predictions == target_ids).float().mean().item()),
                    "grad_norm": float(role_gradient_vector.norm().item()),
                    "grad_cosine_to_update": _safe_cosine(role_gradient_vector, update_vectors[role_name]),
                    "grad_cosine_to_effective_step": (
                        _safe_cosine(role_gradient_vector, effective_step_vector)
                        if effective_step_vector is not None
                        else float("nan")
                    ),
                    **component_norms,
                }
            )

    for role_name in tracked_roles:
        for left_index, left_family in enumerate(FORMATION_FAMILY_ORDER):
            left_vector = role_family_vectors[(role_name, left_family)]
            for right_family in FORMATION_FAMILY_ORDER[left_index + 1:]:
                right_vector = role_family_vectors[(role_name, right_family)]
                family_cosine_rows.append(
                    {
                        "role_name": role_name,
                        "family_left": left_family,
                        "family_right": right_family,
                        "grad_cosine": _safe_cosine(left_vector, right_vector),
                    }
                )

    for family_name in FORMATION_FAMILY_ORDER:
        role_names = list(tracked_roles.keys())
        for left_index, left_role in enumerate(role_names):
            left_vector = role_family_vectors[(left_role, family_name)]
            for right_role in role_names[left_index + 1:]:
                right_vector = role_family_vectors[(right_role, family_name)]
                role_pair_rows.append(
                    {
                        "family_name": family_name,
                        "role_left": left_role,
                        "role_right": right_role,
                        "grad_cosine": _safe_cosine(left_vector, right_vector),
                    }
                )
    return family_role_rows, family_cosine_rows, role_pair_rows


def _logit_contribution_rows(
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    cache: dict[str, Any],
    annotations: list[Any],
    roles: dict[str, FormationHeadConfig | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    value_token_ids = [bundle.token_to_id[token] for token in bundle.value_tokens]
    for role_name, ref in roles.items():
        if ref is None:
            continue
        residual_contribution = head_residual_contribution(model, cache, ref.layer_index, ref.head_index).detach().cpu()
        for row_index, annotation in enumerate(annotations):
            selected_value_id = bundle.token_to_id[annotation.selected_value]
            contribution_logits = torch.matmul(
                residual_contribution[row_index, -1],
                model.token_embed.weight.detach().cpu().T,
            )
            value_logits = contribution_logits[value_token_ids]
            selected_offset = bundle.value_tokens.index(annotation.selected_value)
            foil_logits = [float(value_logits[offset].item()) for offset in range(len(bundle.value_tokens)) if offset != selected_offset]
            if not foil_logits:
                raise ValueError("Logit decomposition requires at least one foil value")
            rows.append(
                {
                    "role_name": role_name,
                    "head_name": _head_name(ref),
                    "prompt_id": annotation.prompt_id,
                    "selected_value": annotation.selected_value,
                    "selected_value_logit": float(contribution_logits[selected_value_id].item()),
                    "selected_value_margin": float(value_logits[selected_offset].item() - max(foil_logits)),
                }
            )
    return rows


def maybe_record_formation_step(
    *,
    context: FormationRunContext | None,
    manifest: RunManifest,
    model: TinyDecoderTransformer,
    bundle: DatasetBundle,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    curriculum_stage: str,
    pre_step_snapshot: dict[str, torch.Tensor],
) -> FormationRunContext | None:
    if context is None:
        return None
    was_training = model.training

    should_force_log = context.boost_steps_remaining > 0
    should_log = (global_step % context.config.log_every_steps == 0) or should_force_log
    should_collect_gradients = (
        context.config.save_family_gradients
        and ((global_step % context.config.gradient_family_interval == 0) or should_force_log)
    )
    if not should_log and not should_collect_gradients:
        return context

    original_rows = context.family_rows["original"]
    candidates, candidate_debug = _resolve_candidates(
        model,
        bundle,
        original_rows,
        manifest=manifest,
        device=device,
        pre_step_snapshot=pre_step_snapshot,
    )
    support_ref = candidates["support"]
    retrieval_ref = candidates["retrieval"]
    placebo_ref = candidates["placebo"]
    if support_ref is None or retrieval_ref is None:
        raise ValueError("Formation logging requires support and retrieval candidates")

    model.eval()
    try:
        _logits, original_cache, original_annotations = _run_rows_with_cache(model, bundle, original_rows, device=device)
        q_metrics = _compute_q_metrics(model, bundle, original_cache, original_annotations, support_ref)
        r_metrics = _compute_r_metrics(original_cache, original_annotations, retrieval_ref)
        w_metrics = _compute_w_metrics(model, bundle, original_cache, original_annotations, retrieval_ref)
        path_gain = _support_path_gain(
            model,
            bundle,
            original_rows,
            device=device,
            support_ref=support_ref,
            retrieval_ref=retrieval_ref,
            base_routing_metrics=r_metrics,
        ) if context.config.measure_path_gain else {}

        family_loss_rows = [
            {
                "family_name": family_name,
                **_family_loss_accuracy(model, bundle, family_rows, device=device),
            }
            for family_name, family_rows in context.family_rows.items()
        ]

        update_vectors: dict[str, torch.Tensor] = {
            "support": _role_update_vector(model, support_ref, pre_step_snapshot),
            "retrieval": _role_update_vector(model, retrieval_ref, pre_step_snapshot),
        }
        if placebo_ref is not None:
            update_vectors["placebo"] = _role_update_vector(model, placebo_ref, pre_step_snapshot)

        optimizer_metrics: dict[str, dict[str, Any]] = {}
        for role_name, ref in (("support", support_ref), ("retrieval", retrieval_ref), ("placebo", placebo_ref)):
            if ref is None:
                continue
            entry = _role_optimizer_metrics(model, optimizer, ref, update_vectors.get(role_name))
            entry["role_name"] = role_name
            optimizer_metrics[role_name] = entry

        family_gradient_rows: list[dict[str, Any]] = []
        family_gradient_cosines: list[dict[str, Any]] = []
        role_pair_gradient_cosines: list[dict[str, Any]] = []
        if should_collect_gradients:
            family_gradient_rows, family_gradient_cosines, role_pair_gradient_cosines = _family_gradients_for_roles(
                model,
                bundle,
                context.family_rows,
                device=device,
                roles={"support": support_ref, "retrieval": retrieval_ref, "placebo": placebo_ref},
                update_vectors=update_vectors,
                optimizer_metrics=optimizer_metrics,
            )

        logit_contribution_rows = (
            _logit_contribution_rows(
                model,
                bundle,
                original_cache,
                original_annotations,
                {"support": support_ref, "retrieval": retrieval_ref, "placebo": placebo_ref},
            )
            if context.config.save_logit_decomposition
            else []
        )
    finally:
        model.train(was_training)

    scalar_metrics = {
        "Q": float(q_metrics["top_key_accuracy"]),
        "R": float(r_metrics["top_slot_accuracy"]),
        "W": float(w_metrics["top_written_value_accuracy"]),
    }
    transition_metric_value = scalar_metrics[context.config.transition_metric_name]
    transition_boost_activated = (
        context.last_transition_metric_value is not None
        and transition_metric_value - context.last_transition_metric_value >= context.config.transition_metric_delta
    )
    if transition_boost_activated:
        context.boost_steps_remaining = context.config.transition_boost_steps
    context.last_transition_metric_value = transition_metric_value
    if context.boost_steps_remaining > 0:
        context.boost_steps_remaining -= 1

    append_jsonl(
        context.history_path,
        {
            "epoch": epoch,
            "global_step": global_step,
            "curriculum_stage": curriculum_stage,
            "candidate_mode": context.config.candidate_mode,
            "candidate_support_head": _head_dict(support_ref),
            "candidate_retrieval_head": _head_dict(retrieval_ref),
            "candidate_placebo_head": _head_dict(placebo_ref),
            "Q": q_metrics,
            "R": r_metrics,
            "W": w_metrics,
            "path_gain": path_gain,
            "family_losses": family_loss_rows,
            "family_gradients": family_gradient_rows,
            "family_gradient_cosines": family_gradient_cosines,
            "role_pair_gradient_cosines": role_pair_gradient_cosines,
            "optimizer_role_metrics": [
                {
                    key: value
                    for key, value in entry.items()
                    if key != "effective_step_vector"
                }
                for entry in optimizer_metrics.values()
            ],
            "logit_contributions": logit_contribution_rows,
            "transition_metric_name": context.config.transition_metric_name,
            "transition_metric_value": transition_metric_value,
            "transition_boost_activated": transition_boost_activated,
            "candidate_debug": candidate_debug,
        },
    )
    return context


def load_formation_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank line found in formation history at {path}:{line_number}")
        rows.append(json.loads(line))
    return rows
