from __future__ import annotations

import pandas as pd
import torch

from scripts.kv_algorithm_oracle import annotate_row
from scripts.kv_algorithm_record import RecordedPrompt
from scripts.kv_retrieve_analysis import DatasetBundle, source_component_tensor, summarize_logits_against_target
from scripts.kv_retrieve_features import forward_with_modified_source


def _collect_source_tensor(
    model: torch.nn.Module,
    cache: dict,
    source_patch: dict,
) -> torch.Tensor:
    return source_component_tensor(
        model=model,
        cache=cache,
        source_patch=source_patch,
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    ).detach().cpu()


def build_class_conditional_replacement_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    base_row: dict[str, object],
    source_rows: list[dict[str, object]],
    base_cache: dict,
    source_patch: dict,
    destination_layer_index: int,
    class_variable: str,
    device: torch.device,
    analysis_head_index: int = 0,
) -> pd.DataFrame:
    if not source_rows:
        raise ValueError("Expected at least one source row for class-conditional replacement")

    base_annotation = annotate_row(base_row)
    if not hasattr(base_annotation, class_variable):
        raise ValueError(f"Unknown class variable {class_variable!r} on prompt annotations")

    base_source = _collect_source_tensor(model, base_cache, source_patch)
    base_length = len(base_annotation.prompt.split())
    grouped_tensors: dict[str, list[torch.Tensor]] = {}

    for row in source_rows:
        annotation = annotate_row(row)
        if len(annotation.prompt.split()) != base_length:
            continue
        class_value = str(getattr(annotation, class_variable))
        _, cache = model(
            torch.tensor([[bundle.token_to_id[token] for token in annotation.prompt.split()]], device=device),
            return_cache=True,
        )
        grouped_tensors.setdefault(class_value, []).append(_collect_source_tensor(model, cache, source_patch))

    if not grouped_tensors:
        raise ValueError(
            "No source rows matched the base prompt length for class-conditional replacement"
        )

    input_ids = torch.tensor(
        [[bundle.token_to_id[token] for token in base_annotation.prompt.split()]],
        device=device,
    )
    source_positions = source_patch.get("source_positions")
    rows: list[dict[str, object]] = []
    for class_value, tensors in sorted(grouped_tensors.items()):
        mean_tensor = torch.stack(tensors).mean(dim=0)
        modified_source = base_source.clone()
        if source_positions is None:
            modified_source = mean_tensor
        else:
            for position in source_positions:
                modified_source[:, position, :] = mean_tensor[:, position, :]

        with torch.no_grad():
            logits, details = forward_with_modified_source(
                model=model,
                input_ids=input_ids,
                base_cache=base_cache,
                source_patch=source_patch,
                modified_source_tensor=modified_source,
                destination_layer_index=destination_layer_index,
            )
        final_logits = logits[0, -1].detach().cpu()
        summary = summarize_logits_against_target(final_logits, bundle, base_annotation.target)
        destination_cache = details["destination_cache"]
        head_pattern = destination_cache["attention"]["pattern"][0, analysis_head_index, -1, :].detach().cpu()
        slot_value_attentions = [
            float(head_pattern[position].item()) for position in base_annotation.value_positions
        ]
        slot_key_attentions = [
            float(head_pattern[position].item()) for position in base_annotation.key_positions
        ]
        rows.append(
            {
                "class_variable": class_variable,
                "class_value": class_value,
                "num_source_prompts": len(tensors),
                "predicted_token": summary["predicted_token"],
                "target_token": summary["target_token"],
                "foil_token": summary["foil_token"],
                "margin": float(summary["margin"]),
                "correct": bool(summary["correct"]),
                "top_value_slot_after_replacement": int(
                    max(range(len(slot_value_attentions)), key=lambda index: slot_value_attentions[index])
                ),
                "top_key_slot_after_replacement": int(
                    max(range(len(slot_key_attentions)), key=lambda index: slot_key_attentions[index])
                ),
                "selected_slot_value_attention_after_replacement": slot_value_attentions[
                    base_annotation.matching_slot
                ],
                "selected_slot_key_attention_after_replacement": slot_key_attentions[
                    base_annotation.matching_slot
                ],
            }
        )

    return pd.DataFrame(rows)


def build_family_query_replacement_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    family_recorded_prompts: list[RecordedPrompt],
    *,
    source_patch: dict,
    destination_layer_index: int,
    device: torch.device,
    analysis_head_index: int = 0,
) -> pd.DataFrame:
    if not family_recorded_prompts:
        raise ValueError("Expected at least one recorded prompt for family query replacement")

    base_prompt_ids = {recorded.annotation.base_prompt_id for recorded in family_recorded_prompts}
    if len(base_prompt_ids) != 1:
        raise ValueError(
            f"Family query replacement expects prompts from one base family, got {sorted(base_prompt_ids)}"
        )

    query_groups: dict[str, list[RecordedPrompt]] = {}
    for recorded in family_recorded_prompts:
        query_groups.setdefault(recorded.annotation.query_key, []).append(recorded)

    rows: list[dict[str, object]] = []
    for base_recorded in family_recorded_prompts:
        base_annotation = base_recorded.annotation
        input_ids = torch.tensor(
            [[bundle.token_to_id[token] for token in base_annotation.prompt.split()]],
            device=device,
        )
        base_source = _collect_source_tensor(model, base_recorded.cache, source_patch)
        for replacement_query_key, source_group in sorted(query_groups.items()):
            replacement_mean = torch.stack(
                [_collect_source_tensor(model, source_recorded.cache, source_patch) for source_recorded in source_group]
            ).mean(dim=0)
            modified_source = base_source.clone()
            source_positions = source_patch.get("source_positions")
            if source_positions is None:
                modified_source = replacement_mean
            else:
                for position in source_positions:
                    modified_source[:, position, :] = replacement_mean[:, position, :]

            expected_target = source_group[0].annotation.selected_value
            expected_slot = source_group[0].annotation.matching_slot
            with torch.no_grad():
                logits, details = forward_with_modified_source(
                    model=model,
                    input_ids=input_ids,
                    base_cache=base_recorded.cache,
                    source_patch=source_patch,
                    modified_source_tensor=modified_source,
                    destination_layer_index=destination_layer_index,
                )
            final_logits = logits[0, -1].detach().cpu()
            summary = summarize_logits_against_target(final_logits, bundle, expected_target)
            destination_cache = details["destination_cache"]
            head_pattern = destination_cache["attention"]["pattern"][0, analysis_head_index, -1, :].detach().cpu()
            slot_value_attentions = [
                float(head_pattern[position].item()) for position in base_annotation.value_positions
            ]
            rows.append(
                {
                    "base_prompt_id": base_annotation.base_prompt_id,
                    "base_query_key": base_annotation.query_key,
                    "replacement_query_key": replacement_query_key,
                    "expected_slot": expected_slot,
                    "expected_target": expected_target,
                    "predicted_token": summary["predicted_token"],
                    "correct": bool(summary["correct"]),
                    "margin": float(summary["margin"]),
                    "top_value_slot_after_replacement": int(
                        max(range(len(slot_value_attentions)), key=lambda index: slot_value_attentions[index])
                    ),
                    "slot_correct": int(
                        max(range(len(slot_value_attentions)), key=lambda index: slot_value_attentions[index])
                    ) == expected_slot,
                }
            )
    return pd.DataFrame(rows)


def build_family_query_replacement_summary_table(
    family_query_replacement_table: pd.DataFrame,
) -> pd.DataFrame:
    if family_query_replacement_table.empty:
        raise ValueError("Expected a non-empty family query replacement table")
    rows: list[dict[str, object]] = []
    for base_prompt_id, group in family_query_replacement_table.groupby("base_prompt_id"):
        rows.append(
            {
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "answer_switch_accuracy": float(group["correct"].mean()),
                "slot_switch_accuracy": float(group["slot_correct"].mean()),
                "margin_mean": float(group["margin"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["answer_switch_accuracy", "base_prompt_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
