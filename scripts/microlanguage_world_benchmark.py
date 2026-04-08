from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.kv_retrieve_analysis import load_dataset_bundle
from scripts.tiny_transformer_core import (
    TinyDecoderTransformer,
    TinyGroupDecoderTransformer,
    TinyQueryGroupEncoderTransformer,
)
from scripts.training_dynamics import (
    _apply_answer_space_mask,
    RunManifest,
    append_jsonl,
    build_epoch_batches,
    build_step_dynamics_record,
    resolve_active_training_interventions,
    snapshot_tracked_parameter_groups,
)

MICRO_ROLE_TO_ID = {
    "bos": 0,
    "separator": 1,
    "query_marker": 2,
    "arrow": 3,
    "event_relation": 4,
    "event_subject": 5,
    "event_value": 6,
    "query_family": 7,
    "query_subject": 8,
    "story_filler": 9,
    "answer_value": 10,
}


def load_microlanguage_world_bundle(manifest: RunManifest):
    bundle = load_dataset_bundle(Path(manifest.dataset.dataset_dir).expanduser().resolve())
    validate_microlanguage_world_manifest_dataset(manifest, bundle)
    return bundle


def validate_microlanguage_world_manifest_dataset(manifest: RunManifest, bundle: Any) -> None:
    required_train_split = manifest.dataset.train_split_by_pairs.get("default")
    if not required_train_split:
        raise ValueError("Microlanguage manifest must define dataset.train_split_by_pairs['default']")
    required_splits = {required_train_split} | set(manifest.dataset.eval_splits.values())
    missing = sorted(required_splits - set(bundle.raw_splits))
    if missing:
        raise ValueError(f"Microlanguage dataset is missing required splits: {missing}")

    if manifest.dataset.sweep_base_split not in bundle.raw_splits:
        raise ValueError(
            "Microlanguage dataset is missing the configured sweep base split "
            f"{manifest.dataset.sweep_base_split!r}"
        )

    max_prompt_length = max(
        len(str(row["prompt"]).split())
        for split_rows in bundle.raw_splits.values()
        for row in split_rows
    )
    if max_prompt_length > manifest.model.max_seq_len:
        raise ValueError(
            "Manifest model.max_seq_len is too small for the microlanguage dataset: "
            f"required {max_prompt_length}, got {manifest.model.max_seq_len}"
        )

    if manifest.dataset.supervision_mode == "dense_value_answer_vocab":
        raise ValueError(
            "Microlanguage supervision_mode='dense_value_answer_vocab' is invalid for these causal prompts: "
            "event_value tokens are introduced by the prompt and are not inferable from the left context. "
            "Use supervision_mode='next_token_vocab' or 'query_target_group_head' instead."
        )

    if manifest.dataset.supervision_mode in {"query_target_group_head", "query_target_group_encoder_head"}:
        latent_state = bundle.metadata.get("latent_state")
        if not isinstance(latent_state, dict):
            raise ValueError("Microlanguage dataset metadata must define latent_state")
        query_specs = latent_state.get("query_families")
        vocab_groups = bundle.metadata.get("vocabulary_groups")
        if not isinstance(query_specs, dict) or not isinstance(vocab_groups, dict):
            raise ValueError("Microlanguage dataset metadata must define query_families and vocabulary_groups")
        target_groups = {
            spec.get("target_group")
            for spec in query_specs.values()
            if isinstance(spec, dict)
        }
        missing_groups = sorted(
            group_name
            for group_name in target_groups
            if isinstance(group_name, str) and group_name not in vocab_groups
        )
        if missing_groups:
            raise ValueError(f"Microlanguage dataset is missing vocabulary groups for target groups: {missing_groups}")
    if manifest.dataset.supervision_mode == "query_target_group_encoder_head" and not manifest.model.use_role_embeddings:
        raise ValueError("Microlanguage encoder-head supervision requires model.use_role_embeddings=true")


def _query_family_specs(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latent_state = metadata.get("latent_state")
    if not isinstance(latent_state, dict):
        raise ValueError("Dataset metadata must define latent_state")
    query_specs = latent_state.get("query_families")
    if not isinstance(query_specs, dict):
        raise ValueError("Dataset metadata latent_state.query_families must be an object")
    return query_specs


def _vocabulary_groups(metadata: dict[str, Any]) -> dict[str, list[str]]:
    vocab_groups = metadata.get("vocabulary_groups")
    if not isinstance(vocab_groups, dict):
        raise ValueError("Dataset metadata must define vocabulary_groups")
    return vocab_groups


def _row_target_group_name(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    query_family = row.get("query_family")
    if not isinstance(query_family, str) or not query_family.strip():
        raise ValueError(f"Row is missing query_family: {row!r}")
    query_specs = _query_family_specs(metadata)
    spec = query_specs.get(query_family)
    if not isinstance(spec, dict):
        raise ValueError(f"Unknown query_family {query_family!r} in dataset metadata")
    target_group = spec.get("target_group")
    if not isinstance(target_group, str) or not target_group.strip():
        raise ValueError(f"Query family {query_family!r} does not define a target_group")
    return target_group


def _group_head_output_sizes(metadata: dict[str, Any]) -> dict[str, int]:
    query_specs = _query_family_specs(metadata)
    vocab_groups = _vocabulary_groups(metadata)
    output_sizes: dict[str, int] = {}
    for spec in query_specs.values():
        if not isinstance(spec, dict):
            continue
        target_group = spec.get("target_group")
        if not isinstance(target_group, str):
            continue
        group_tokens = vocab_groups.get(target_group)
        if not isinstance(group_tokens, list) or not group_tokens:
            raise ValueError(f"Target group {target_group!r} is missing or empty")
        output_sizes[target_group] = len(group_tokens)
    if not output_sizes:
        raise ValueError("Microlanguage benchmark requires at least one target group")
    return dict(sorted(output_sizes.items()))


def build_microlanguage_world_model_config(manifest: RunManifest, bundle: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "vocab_size": len(bundle.vocab),
        "d_model": manifest.model.d_model,
        "n_heads": manifest.model.n_heads,
        "d_ff": manifest.model.d_ff,
        "n_layers": manifest.model.n_layers,
        "max_seq_len": manifest.model.max_seq_len,
    }
    if manifest.model.use_role_embeddings:
        config["num_role_ids"] = len(MICRO_ROLE_TO_ID)
    if manifest.dataset.supervision_mode in {"query_target_group_head", "query_target_group_encoder_head"}:
        config["group_head_output_sizes"] = _group_head_output_sizes(bundle.metadata)
    if manifest.dataset.supervision_mode == "query_target_group_encoder_head":
        config["classifier_role_id"] = MICRO_ROLE_TO_ID["query_subject"]
    return config


def _apply_initialization_scale(model: TinyDecoderTransformer, scale: float) -> None:
    if scale <= 0.0:
        raise ValueError(f"Initialization scale must be positive, got {scale}")
    if scale == 1.0:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.mul_(scale)


def instantiate_microlanguage_world_model(
    manifest: RunManifest,
    bundle: Any,
    device: torch.device,
) -> TinyDecoderTransformer:
    config = build_microlanguage_world_model_config(manifest, bundle)
    if manifest.dataset.supervision_mode == "query_target_group_head":
        model = TinyGroupDecoderTransformer(**config).to(device)
    elif manifest.dataset.supervision_mode == "query_target_group_encoder_head":
        model = TinyQueryGroupEncoderTransformer(**config).to(device)
    else:
        model = TinyDecoderTransformer(**config).to(device)
    _apply_initialization_scale(model, manifest.initialization.scale)
    return model


def select_microlanguage_world_training_rows(
    manifest: RunManifest,
    bundle: Any,
    epoch: int,
) -> tuple[list[dict[str, Any]], str]:
    if epoch < 1 or epoch > manifest.training.epochs:
        raise ValueError(f"Epoch {epoch} is outside the configured training range 1..{manifest.training.epochs}")
    train_split_name = manifest.dataset.train_split_by_pairs["default"]
    return bundle.raw_splits[train_split_name], train_split_name


def _encode_microlanguage_prompt_tokens_and_roles(
    prompt: str,
    token_to_id: dict[str, int],
) -> tuple[list[int], list[int]]:
    tokens = prompt.split()
    if not tokens or tokens[0] != "<bos>":
        raise ValueError(f"Prompt must start with <bos>, got {prompt!r}")
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens[1:]:
        if token == ";":
            if not current:
                raise ValueError(f"Encountered empty prompt segment in {prompt!r}")
            segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    if len(segments) < 2:
        raise ValueError(f"Prompt must contain at least one event and one query segment, got {prompt!r}")

    token_ids: list[int] = [token_to_id["<bos>"]]
    role_ids: list[int] = [MICRO_ROLE_TO_ID["bos"]]
    event_segments = segments[:-1]
    query_segment = segments[-1]
    for segment in event_segments:
        token_role_names = ["story_filler" for _ in segment]
        if len(segment) == 3:
            token_role_names = ["event_relation", "event_subject", "event_value"]
        else:
            try:
                relation_marker = segment.index("relation")
                subject_marker = segment.index("subject")
                value_marker = segment.index("value")
            except ValueError as exc:
                raise ValueError(f"Event segment is missing story markers, got {segment!r} in {prompt!r}") from exc
            marker_to_role = {
                relation_marker + 1: "event_relation",
                subject_marker + 1: "event_subject",
                value_marker + 1: "event_value",
            }
            for marker_index, role_name in marker_to_role.items():
                if marker_index >= len(segment):
                    raise ValueError(f"Event segment is missing a token after a story marker: {segment!r}")
                token_role_names[marker_index] = role_name
        for token, role_name in zip(segment, token_role_names, strict=True):
            if token not in token_to_id:
                raise ValueError(f"Unknown prompt token {token!r} in prompt {prompt!r}")
            token_ids.append(token_to_id[token])
            role_ids.append(MICRO_ROLE_TO_ID[role_name])
        token_ids.append(token_to_id[";"])
        role_ids.append(MICRO_ROLE_TO_ID["separator"])
    if len(query_segment) != 4 or query_segment[0] != "Q" or query_segment[-1] != "->":
        raise ValueError(f"Malformed query segment {query_segment!r} in prompt {prompt!r}")
    for token, role_name in (
        (query_segment[0], "query_marker"),
        (query_segment[1], "query_family"),
        (query_segment[2], "query_subject"),
        (query_segment[3], "arrow"),
    ):
        if token not in token_to_id:
            raise ValueError(f"Unknown prompt token {token!r} in prompt {prompt!r}")
        token_ids.append(token_to_id[token])
        role_ids.append(MICRO_ROLE_TO_ID[role_name])
    return token_ids, role_ids


def _encode_microlanguage_batch(
    rows: list[dict[str, Any]],
    token_to_id: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts: list[list[int]] = []
    role_prompts: list[list[int]] = []
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row must contain a non-empty prompt, got {row!r}")
        token_ids, role_ids = _encode_microlanguage_prompt_tokens_and_roles(prompt, token_to_id)
        prompts.append(token_ids)
        role_prompts.append(role_ids)
    return (
        torch.tensor(prompts, dtype=torch.long, device=device),
        torch.tensor(role_prompts, dtype=torch.long, device=device),
    )


def _encode_microlanguage_final_targets(
    rows: list[dict[str, Any]],
    token_to_id: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    target_ids: list[int] = []
    for row in rows:
        target = row.get("target")
        if not isinstance(target, str) or target not in token_to_id:
            raise ValueError(f"Unknown target token {target!r} in row {row!r}")
        target_ids.append(token_to_id[target])
    return torch.tensor(target_ids, dtype=torch.long, device=device)


def _encode_microlanguage_dense_sequence_batch(
    rows: list[dict[str, Any]],
    token_to_id: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_sequences: list[list[int]] = []
    role_sequences: list[list[int]] = []
    target_sequences: list[list[int]] = []
    loss_masks: list[list[bool]] = []
    for row in rows:
        prompt = row.get("prompt")
        target = row.get("target")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row must contain a non-empty prompt, got {row!r}")
        if not isinstance(target, str) or target not in token_to_id:
            raise ValueError(f"Unknown target token {target!r} in row {row!r}")
        prompt_ids, prompt_role_ids = _encode_microlanguage_prompt_tokens_and_roles(prompt, token_to_id)
        full_ids = [*prompt_ids, token_to_id[target]]
        full_role_ids = [*prompt_role_ids, MICRO_ROLE_TO_ID["answer_value"]]
        input_sequences.append(full_ids[:-1])
        role_sequences.append(full_role_ids[:-1])
        target_sequences.append(full_ids[1:])
        loss_masks.append(
            [
                role_id in {MICRO_ROLE_TO_ID["event_value"], MICRO_ROLE_TO_ID["answer_value"]}
                for role_id in full_role_ids[1:]
            ]
        )
    return (
        torch.tensor(input_sequences, dtype=torch.long, device=device),
        torch.tensor(role_sequences, dtype=torch.long, device=device),
        torch.tensor(target_sequences, dtype=torch.long, device=device),
        torch.tensor(loss_masks, dtype=torch.bool, device=device),
    )


def _build_local_group_answer_mask(
    *,
    rows: list[dict[str, Any]],
    group_tokens: list[str],
    token_to_local_index: dict[str, int],
    answer_space_mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if answer_space_mode in {"none", "query_target_group"}:
        return None
    if answer_space_mode != "active_query_target_group":
        raise ValueError(f"Unsupported answer_space_mode {answer_space_mode!r} for group-head supervision")
    mask = torch.zeros((len(rows), len(group_tokens)), dtype=torch.bool, device=device)
    group_token_set = set(group_tokens)
    for row_index, row in enumerate(rows):
        prompt = row.get("prompt")
        target = row.get("target")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row must contain a non-empty prompt for active group masking, got {row!r}")
        if not isinstance(target, str) or target not in token_to_local_index:
            raise ValueError(f"Row must contain a valid target token for active group masking, got {row!r}")
        active_tokens = sorted({token for token in prompt.split() if token in group_token_set})
        if not active_tokens:
            raise ValueError(f"Prompt does not contain any active tokens from target group: {prompt!r}")
        token_indices = [token_to_local_index[token] for token in active_tokens]
        mask[row_index, token_indices] = True
        if not mask[row_index, token_to_local_index[target]]:
            raise ValueError(f"Target token {target!r} is not in the active answer space for prompt {prompt!r}")
    return mask


def _apply_local_group_answer_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return logits
    if logits.shape != mask.shape:
        raise ValueError(f"Local answer mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}")
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    if not torch.isfinite(masked_logits.max(dim=-1).values).all():
        raise ValueError("Local answer-space masking removed every valid logit for at least one row")
    return masked_logits


def _group_rows_by_prompt_length_and_target_group(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Training/eval row is missing a prompt: {row!r}")
        key = (len(prompt.split()), _row_target_group_name(row, metadata))
        grouped.setdefault(key, []).append(row)
    return grouped


def _encode_group_targets(
    rows: list[dict[str, Any]],
    *,
    group_tokens: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    token_to_local_index = {token: index for index, token in enumerate(group_tokens)}
    target_ids: list[int] = []
    for row in rows:
        target = row.get("target")
        if not isinstance(target, str) or target not in token_to_local_index:
            raise ValueError(f"Unknown target token {target!r} for group-head supervision")
        target_ids.append(token_to_local_index[target])
    return torch.tensor(target_ids, dtype=torch.long, device=device), token_to_local_index


def _evaluate_microlanguage_vocab_rows(
    model: TinyDecoderTransformer,
    rows: list[dict[str, Any]],
    *,
    manifest: RunManifest,
    bundle: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    grouped = {}
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Training/eval row is missing a prompt: {row!r}")
        grouped.setdefault(len(prompt.split()), []).append(row)

    total_rows = 0
    total_loss = 0.0
    total_correct = 0
    total_margin = 0.0
    prediction_rows: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for prompt_length in sorted(grouped):
            split_rows = grouped[prompt_length]
            for start in range(0, len(split_rows), batch_size):
                batch_rows = split_rows[start:start + batch_size]
                input_ids, role_ids = _encode_microlanguage_batch(batch_rows, bundle.token_to_id, device)
                target_ids = _encode_microlanguage_final_targets(batch_rows, bundle.token_to_id, device)
                logits = model(input_ids, role_ids=role_ids if manifest.model.use_role_embeddings else None)
                final_logits = _apply_answer_space_mask(
                    logits[:, -1, :],
                    rows=batch_rows,
                    token_to_id=bundle.token_to_id,
                    dataset_metadata=bundle.metadata,
                    answer_space_mode=manifest.dataset.answer_space_mode,
                )
                losses = F.cross_entropy(final_logits, target_ids, reduction="none")
                predicted_ids = final_logits.argmax(dim=-1)
                target_logits = final_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                competing_logits = final_logits.clone()
                competing_logits.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
                foil_logits, foil_ids = competing_logits.max(dim=-1)

                total_rows += len(batch_rows)
                total_loss += float(losses.sum().item())
                total_correct += int((predicted_ids == target_ids).sum().item())
                total_margin += float((target_logits - foil_logits).sum().item())

                for row_index, row in enumerate(batch_rows):
                    prediction_rows.append(
                        {
                            "prompt_id": str(row.get("id") or row.get("prompt_id") or row["prompt"]),
                            "predicted_token": bundle.id_to_token[int(predicted_ids[row_index].item())],
                            "target_token": row["target"],
                            "foil_token": bundle.id_to_token[int(foil_ids[row_index].item())],
                            "margin": float((target_logits[row_index] - foil_logits[row_index]).item()),
                            "correct": bool(predicted_ids[row_index].item() == target_ids[row_index].item()),
                        }
                    )
    return {
        "rows": total_rows,
        "loss": total_loss / total_rows,
        "accuracy": total_correct / total_rows,
        "margin": total_margin / total_rows,
        "predictions": prediction_rows,
    }


def evaluate_microlanguage_world_rows(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    manifest: RunManifest,
    bundle: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if manifest.dataset.supervision_mode in {"next_token_vocab", "dense_value_answer_vocab"}:
        if not isinstance(model, TinyDecoderTransformer):
            raise ValueError(
                "Microlanguage vocab supervision requires TinyDecoderTransformer, "
                f"got {type(model).__name__}"
            )
        return _evaluate_microlanguage_vocab_rows(
            model,
            rows,
            manifest=manifest,
            bundle=bundle,
            device=device,
            batch_size=batch_size,
        )
    if manifest.dataset.supervision_mode not in {"query_target_group_head", "query_target_group_encoder_head"}:
        raise ValueError(f"Unsupported microlanguage supervision_mode {manifest.dataset.supervision_mode!r}")
    if manifest.dataset.supervision_mode == "query_target_group_head":
        if not isinstance(model, TinyGroupDecoderTransformer):
            raise ValueError(
                "Microlanguage decoder group-head supervision requires TinyGroupDecoderTransformer, "
                f"got {type(model).__name__}"
            )
    else:
        if not isinstance(model, TinyQueryGroupEncoderTransformer):
            raise ValueError(
                "Microlanguage encoder group-head supervision requires TinyQueryGroupEncoderTransformer, "
                f"got {type(model).__name__}"
            )

    grouped = _group_rows_by_prompt_length_and_target_group(rows, metadata=bundle.metadata)
    vocab_groups = _vocabulary_groups(bundle.metadata)
    total_rows = 0
    total_loss = 0.0
    total_correct = 0
    total_margin = 0.0
    prediction_rows: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for grouping_key in sorted(grouped):
            _, target_group = grouping_key
            split_rows = grouped[grouping_key]
            group_tokens = vocab_groups[target_group]
            for start in range(0, len(split_rows), batch_size):
                batch_rows = split_rows[start:start + batch_size]
                input_ids, role_ids = _encode_microlanguage_batch(batch_rows, bundle.token_to_id, device)
                target_ids, token_to_local_index = _encode_group_targets(
                    batch_rows,
                    group_tokens=group_tokens,
                    device=device,
                )
                logits = model.forward_group_logits(input_ids, group_name=target_group, role_ids=role_ids)
                if manifest.dataset.supervision_mode == "query_target_group_head":
                    logits = logits[:, -1, :]
                final_logits = _apply_local_group_answer_mask(
                    logits,
                    _build_local_group_answer_mask(
                        rows=batch_rows,
                        group_tokens=group_tokens,
                        token_to_local_index=token_to_local_index,
                        answer_space_mode=manifest.dataset.answer_space_mode,
                        device=device,
                    ),
                )
                losses = F.cross_entropy(final_logits, target_ids, reduction="none")
                predicted_ids = final_logits.argmax(dim=-1)
                target_logits = final_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                competing_logits = final_logits.clone()
                competing_logits.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
                foil_logits, foil_ids = competing_logits.max(dim=-1)

                total_rows += len(batch_rows)
                total_loss += float(losses.sum().item())
                total_correct += int((predicted_ids == target_ids).sum().item())
                total_margin += float((target_logits - foil_logits).sum().item())

                for row_index, row in enumerate(batch_rows):
                    prediction_rows.append(
                        {
                            "prompt_id": str(row.get("id") or row.get("prompt_id") or row["prompt"]),
                            "predicted_token": group_tokens[int(predicted_ids[row_index].item())],
                            "target_token": row["target"],
                            "foil_token": group_tokens[int(foil_ids[row_index].item())],
                            "margin": float((target_logits[row_index] - foil_logits[row_index]).item()),
                            "correct": bool(predicted_ids[row_index].item() == target_ids[row_index].item()),
                        }
                    )
    return {
        "rows": total_rows,
        "loss": total_loss / total_rows,
        "accuracy": total_correct / total_rows,
        "margin": total_margin / total_rows,
        "predictions": prediction_rows,
    }


def _train_microlanguage_vocab_epoch(
    model: TinyDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    *,
    manifest: RunManifest,
    bundle: Any,
    device: torch.device,
    epoch: int,
    history_path: Path,
    global_step_start: int,
    batch_seed: int,
    curriculum_stage: str,
    show_progress: bool = False,
) -> dict[str, Any]:
    batches = build_epoch_batches(rows, batch_size=manifest.training.batch_size, seed=batch_seed)
    if not batches:
        raise ValueError("Expected at least one batch in _train_microlanguage_vocab_epoch")

    active_interventions = resolve_active_training_interventions(manifest, epoch)
    if active_interventions:
        raise ValueError("Microlanguage vocab supervision does not support training interventions")

    model.train()
    total_rows = 0
    total_loss = 0.0
    global_step = global_step_start
    batch_iterator = tqdm(batches, desc=f"epoch {epoch}", leave=False) if show_progress else batches
    for batch_index, batch_rows in enumerate(batch_iterator):
        optimizer.zero_grad()
        if manifest.dataset.supervision_mode == "dense_value_answer_vocab":
            input_ids, role_ids, target_ids, loss_mask = _encode_microlanguage_dense_sequence_batch(
                batch_rows,
                bundle.token_to_id,
                device,
            )
            logits = model(input_ids, role_ids=role_ids if manifest.model.use_role_embeddings else None)
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="none",
            ).view_as(target_ids)
            selected_losses = token_losses.masked_select(loss_mask)
            if selected_losses.numel() == 0:
                raise ValueError("Dense microlanguage supervision produced an empty loss mask")
            loss = selected_losses.mean()
        elif manifest.dataset.supervision_mode == "next_token_vocab":
            input_ids, role_ids = _encode_microlanguage_batch(batch_rows, bundle.token_to_id, device)
            target_ids = _encode_microlanguage_final_targets(batch_rows, bundle.token_to_id, device)
            logits = model(input_ids, role_ids=role_ids if manifest.model.use_role_embeddings else None)
            final_logits = _apply_answer_space_mask(
                logits[:, -1, :],
                rows=batch_rows,
                token_to_id=bundle.token_to_id,
                dataset_metadata=bundle.metadata,
                answer_space_mode=manifest.dataset.answer_space_mode,
            )
            loss = F.cross_entropy(final_logits, target_ids)
        else:
            raise ValueError(f"Unsupported vocab supervision_mode {manifest.dataset.supervision_mode!r}")
        loss.backward()
        pre_step_snapshot = snapshot_tracked_parameter_groups(model)
        optimizer.step()

        batch_loss = float(loss.item())
        batch_rows_count = len(batch_rows)
        total_rows += batch_rows_count
        total_loss += batch_loss * batch_rows_count
        step_dynamics = build_step_dynamics_record(model, pre_step_snapshot)
        append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "batch_index": batch_index,
                "batch_rows": batch_rows_count,
                "batch_loss": batch_loss,
                "curriculum_stage": curriculum_stage,
                "active_interventions": active_interventions,
                "total_grad_norm": step_dynamics["total_grad_norm"],
                "total_param_norm_pre": step_dynamics["total_param_norm_pre"],
                "total_param_norm_post": step_dynamics["total_param_norm_post"],
                "total_update_norm": step_dynamics["total_update_norm"],
                "relative_total_update_norm": step_dynamics["relative_total_update_norm"],
                "parameter_metrics": step_dynamics["parameter_metrics"],
            },
        )
        if show_progress:
            batch_iterator.set_postfix(
                loss=f"{batch_loss:.4f}",
                grad=f"{step_dynamics['total_grad_norm']:.3f}",
                update=f"{step_dynamics['relative_total_update_norm']:.4f}",
            )
        global_step += 1
    if show_progress:
        batch_iterator.close()
    return {
        "epoch": epoch,
        "global_step_end": global_step,
        "train_loss": total_loss / total_rows,
        "train_rows": total_rows,
    }


def train_microlanguage_world_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    *,
    manifest: RunManifest,
    bundle: Any,
    device: torch.device,
    epoch: int,
    history_path: Path,
    global_step_start: int,
    batch_seed: int,
    curriculum_stage: str,
    show_progress: bool = False,
) -> dict[str, Any]:
    if manifest.dataset.supervision_mode in {"next_token_vocab", "dense_value_answer_vocab"}:
        if not isinstance(model, TinyDecoderTransformer):
            raise ValueError(
                "Microlanguage vocab supervision requires TinyDecoderTransformer, "
                f"got {type(model).__name__}"
            )
        return _train_microlanguage_vocab_epoch(
            model,
            optimizer,
            rows,
            manifest=manifest,
            bundle=bundle,
            device=device,
            epoch=epoch,
            history_path=history_path,
            global_step_start=global_step_start,
            batch_seed=batch_seed,
            curriculum_stage=curriculum_stage,
            show_progress=show_progress,
        )
    if manifest.dataset.supervision_mode not in {"query_target_group_head", "query_target_group_encoder_head"}:
        raise ValueError(f"Unsupported microlanguage supervision_mode {manifest.dataset.supervision_mode!r}")
    if manifest.dataset.supervision_mode == "query_target_group_head":
        if not isinstance(model, TinyGroupDecoderTransformer):
            raise ValueError(
                "Microlanguage decoder group-head supervision requires TinyGroupDecoderTransformer, "
                f"got {type(model).__name__}"
            )
    else:
        if not isinstance(model, TinyQueryGroupEncoderTransformer):
            raise ValueError(
                "Microlanguage encoder group-head supervision requires TinyQueryGroupEncoderTransformer, "
                f"got {type(model).__name__}"
            )

    grouped = _group_rows_by_prompt_length_and_target_group(rows, metadata=bundle.metadata)
    grouped_rows: list[list[dict[str, Any]]] = []
    for group_rows in grouped.values():
        grouped_rows.extend(
            build_epoch_batches(
                group_rows,
                batch_size=manifest.training.batch_size,
                seed=batch_seed,
            )
        )
    if not grouped_rows:
        raise ValueError("Expected at least one batch in train_microlanguage_world_epoch")

    model.train()
    total_rows = 0
    total_loss = 0.0
    global_step = global_step_start
    active_interventions = resolve_active_training_interventions(manifest, epoch)
    if active_interventions:
        raise ValueError("Microlanguage group-head supervision does not support training interventions")
    vocab_groups = _vocabulary_groups(bundle.metadata)
    batch_iterator = tqdm(grouped_rows, desc=f"epoch {epoch}", leave=False) if show_progress else grouped_rows
    for batch_index, batch_rows in enumerate(batch_iterator):
        target_group = _row_target_group_name(batch_rows[0], bundle.metadata)
        if any(_row_target_group_name(row, bundle.metadata) != target_group for row in batch_rows):
            raise ValueError("Each training batch must contain rows from a single target group")
        group_tokens = vocab_groups[target_group]
        input_ids, role_ids = _encode_microlanguage_batch(batch_rows, bundle.token_to_id, device)
        target_ids, token_to_local_index = _encode_group_targets(
            batch_rows,
            group_tokens=group_tokens,
            device=device,
        )
        optimizer.zero_grad()
        logits = model.forward_group_logits(input_ids, group_name=target_group, role_ids=role_ids)
        if manifest.dataset.supervision_mode == "query_target_group_head":
            logits = logits[:, -1, :]
        final_logits = _apply_local_group_answer_mask(
            logits,
            _build_local_group_answer_mask(
                rows=batch_rows,
                group_tokens=group_tokens,
                token_to_local_index=token_to_local_index,
                answer_space_mode=manifest.dataset.answer_space_mode,
                device=device,
            ),
        )
        loss = F.cross_entropy(final_logits, target_ids)
        loss.backward()
        pre_step_snapshot = snapshot_tracked_parameter_groups(model)
        optimizer.step()

        batch_loss = float(loss.item())
        batch_rows_count = len(batch_rows)
        total_rows += batch_rows_count
        total_loss += batch_loss * batch_rows_count
        step_dynamics = build_step_dynamics_record(model, pre_step_snapshot)
        append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "batch_index": batch_index,
                "batch_rows": batch_rows_count,
                "batch_loss": batch_loss,
                "target_group": target_group,
                "curriculum_stage": curriculum_stage,
                "active_interventions": active_interventions,
                "total_grad_norm": step_dynamics["total_grad_norm"],
                "total_param_norm_pre": step_dynamics["total_param_norm_pre"],
                "total_param_norm_post": step_dynamics["total_param_norm_post"],
                "total_update_norm": step_dynamics["total_update_norm"],
                "relative_total_update_norm": step_dynamics["relative_total_update_norm"],
                "parameter_metrics": step_dynamics["parameter_metrics"],
            },
        )
        if show_progress:
            batch_iterator.set_postfix(
                loss=f"{batch_loss:.4f}",
                grad=f"{step_dynamics['total_grad_norm']:.3f}",
                update=f"{step_dynamics['relative_total_update_norm']:.4f}",
            )
        global_step += 1
    if show_progress:
        batch_iterator.close()
    return {
        "epoch": epoch,
        "global_step_end": global_step,
        "train_loss": total_loss / total_rows,
        "train_rows": total_rows,
    }


def build_microlanguage_world_checkpoint_metrics(
    manifest: RunManifest,
    model: torch.nn.Module,
    bundle: Any,
    *,
    train_rows: list[dict[str, Any]],
    device: torch.device,
    train_batch_loss: float | None,
) -> dict[str, Any]:
    split_metrics = {
        "train_reference": evaluate_microlanguage_world_rows(
            model,
            train_rows,
            manifest=manifest,
            bundle=bundle,
            device=device,
            batch_size=manifest.battery.eval_batch_size,
        )
    }
    for split_name in manifest.dataset.eval_splits.values():
        split_metrics[split_name] = evaluate_microlanguage_world_rows(
            model,
            bundle.raw_splits[split_name],
            manifest=manifest,
            bundle=bundle,
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
