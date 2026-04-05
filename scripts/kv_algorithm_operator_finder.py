from __future__ import annotations

import pandas as pd
import torch

from scripts.kv_algorithm_record import RecordedPrompt
from scripts.kv_retrieve_analysis import DatasetBundle, decode_token, ov_source_logits


def _top_slot_from_attention(slot_scores: list[float]) -> int:
    if not slot_scores:
        raise ValueError("Expected at least one slot score when selecting a top slot")
    return int(max(range(len(slot_scores)), key=lambda index: slot_scores[index]))


def build_head_attention_operator_table(
    recorded_prompts: list[RecordedPrompt],
    *,
    layer_index: int,
    head_index: int,
    query_position: int = -1,
) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt for operator discovery")

    max_pairs = max(recorded.annotation.num_pairs for recorded in recorded_prompts)
    rows: list[dict[str, object]] = []
    for recorded in recorded_prompts:
        annotation = recorded.annotation
        prompt_tokens = annotation.prompt.split()
        attention = recorded.cache["blocks"][layer_index]["attention"]["pattern"][
            0, head_index, query_position, :
        ].detach().cpu()
        top_position = int(attention.argmax().item())
        slot_total_scores: list[float] = []
        slot_key_scores: list[float] = []
        slot_value_scores: list[float] = []
        row = {
            "prompt_id": annotation.prompt_id,
            "base_prompt_id": annotation.base_prompt_id,
            "family_name": annotation.family_name,
            "family_value": annotation.family_value,
            "query_key": annotation.query_key,
            "selected_value": annotation.selected_value,
            "matching_slot": annotation.matching_slot,
            "top_attention_position": top_position,
            "top_attention_token": prompt_tokens[top_position],
            "top_attention_role": annotation.token_roles[top_position],
            "top_attention_slot": annotation.token_slot_indices[top_position],
            "matching_key_attention": float(attention[annotation.matching_key_position].item()),
            "matching_value_attention": float(attention[annotation.matching_value_position].item()),
            "query_key_attention": float(attention[annotation.query_key_position].item()),
        }
        for slot_index in range(annotation.num_pairs):
            key_attention = float(attention[annotation.key_positions[slot_index]].item())
            value_attention = float(attention[annotation.value_positions[slot_index]].item())
            total_attention = key_attention + value_attention
            slot_key_scores.append(key_attention)
            slot_value_scores.append(value_attention)
            slot_total_scores.append(total_attention)
            row[f"slot_{slot_index}_key_attention"] = key_attention
            row[f"slot_{slot_index}_value_attention"] = value_attention
            row[f"slot_{slot_index}_total_attention"] = total_attention
        for slot_index in range(annotation.num_pairs, max_pairs):
            row[f"slot_{slot_index}_key_attention"] = float("nan")
            row[f"slot_{slot_index}_value_attention"] = float("nan")
            row[f"slot_{slot_index}_total_attention"] = float("nan")
        row["top_key_slot"] = _top_slot_from_attention(slot_key_scores)
        row["top_value_slot"] = _top_slot_from_attention(slot_value_scores)
        row["top_total_slot"] = _top_slot_from_attention(slot_total_scores)
        rows.append(row)

    return pd.DataFrame(rows)


def build_head_attention_operator_summary_table(
    attention_table: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    if attention_table.empty:
        raise ValueError("Expected a non-empty attention operator table")

    summary = {
        "label": label,
        "rows": int(len(attention_table)),
        "top_key_slot_matches": float((attention_table["top_key_slot"] == attention_table["matching_slot"]).mean()),
        "top_value_slot_matches": float((attention_table["top_value_slot"] == attention_table["matching_slot"]).mean()),
        "top_total_slot_matches": float((attention_table["top_total_slot"] == attention_table["matching_slot"]).mean()),
        "matching_key_attention_mean": float(attention_table["matching_key_attention"].mean()),
        "matching_value_attention_mean": float(attention_table["matching_value_attention"].mean()),
        "query_key_attention_mean": float(attention_table["query_key_attention"].mean()),
    }
    return pd.DataFrame([summary])


def build_head_attention_family_stability_table(
    attention_table: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    if attention_table.empty:
        raise ValueError("Expected a non-empty attention operator table")

    rows: list[dict[str, object]] = []
    for base_prompt_id, group in attention_table.groupby("base_prompt_id"):
        rows.append(
            {
                "label": label,
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "top_key_slot_matches": float((group["top_key_slot"] == group["matching_slot"]).mean()),
                "top_value_slot_matches": float((group["top_value_slot"] == group["matching_slot"]).mean()),
                "top_total_slot_matches": float((group["top_total_slot"] == group["matching_slot"]).mean()),
                "matching_key_attention_mean": float(group["matching_key_attention"].mean()),
                "matching_value_attention_mean": float(group["matching_value_attention"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_total_slot_matches", "base_prompt_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_head_copy_rule_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    recorded_prompts: list[RecordedPrompt],
    *,
    layer_index: int,
    head_index: int,
) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt for copy-rule analysis")

    value_token_ids = [bundle.token_to_id[token] for token in bundle.value_tokens]
    rows: list[dict[str, object]] = []
    for recorded in recorded_prompts:
        annotation = recorded.annotation
        selected_source_position = annotation.matching_value_position
        source_logits = ov_source_logits(
            model=model,
            cache=recorded.cache,
            layer_index=layer_index,
            head_index=head_index,
            source_position=selected_source_position,
        )
        top_token_id = int(source_logits.argmax().item())
        value_logits = source_logits[value_token_ids]
        top_value_index = int(value_logits.argmax().item())
        sorted_value_indices = torch.argsort(value_logits, descending=True)
        selected_value_offset = bundle.value_tokens.index(annotation.selected_value)
        selected_value_rank = int((sorted_value_indices == selected_value_offset).nonzero(as_tuple=False)[0].item()) + 1
        selected_slot_value_attention = float(
            recorded.cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, -1, selected_source_position].item()
        )
        rows.append(
            {
                "layer_index": layer_index,
                "head_index": head_index,
                "prompt_id": annotation.prompt_id,
                "base_prompt_id": annotation.base_prompt_id,
                "family_name": annotation.family_name,
                "family_value": annotation.family_value,
                "query_key": annotation.query_key,
                "matching_slot": annotation.matching_slot,
                "selected_value": annotation.selected_value,
                "selected_source_position": selected_source_position,
                "selected_slot_value_attention": selected_slot_value_attention,
                "top_written_token": decode_token(top_token_id, bundle.id_to_token),
                "top_written_value_token": bundle.value_tokens[top_value_index],
                "selected_value_logit": float(source_logits[bundle.token_to_id[annotation.selected_value]].item()),
                "top_written_value_logit": float(value_logits[top_value_index].item()),
                "selected_value_rank_among_values": selected_value_rank,
                "selected_value_is_top_written_value": bundle.value_tokens[top_value_index] == annotation.selected_value,
            }
        )
    return pd.DataFrame(rows)


def build_head_copy_rule_summary_table(copy_rule_table: pd.DataFrame) -> pd.DataFrame:
    if copy_rule_table.empty:
        raise ValueError("Expected a non-empty L2H0 copy-rule table")
    return pd.DataFrame(
        [
            {
                "rows": int(len(copy_rule_table)),
                "selected_value_top_written_rate": float(
                    copy_rule_table["selected_value_is_top_written_value"].mean()
                ),
                "selected_value_rank_mean": float(
                    copy_rule_table["selected_value_rank_among_values"].mean()
                ),
                "selected_slot_value_attention_mean": float(
                    copy_rule_table["selected_slot_value_attention"].mean()
                ),
            }
        ]
    )


def build_head_copy_rule_family_stability_table(copy_rule_table: pd.DataFrame) -> pd.DataFrame:
    if copy_rule_table.empty:
        raise ValueError("Expected a non-empty L2H0 copy-rule table")
    rows: list[dict[str, object]] = []
    for base_prompt_id, group in copy_rule_table.groupby("base_prompt_id"):
        rows.append(
            {
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "selected_value_top_written_rate": float(
                    group["selected_value_is_top_written_value"].mean()
                ),
                "selected_value_rank_mean": float(
                    group["selected_value_rank_among_values"].mean()
                ),
                "selected_slot_value_attention_mean": float(
                    group["selected_slot_value_attention"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selected_value_top_written_rate", "base_prompt_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_l2h0_copy_rule_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    recorded_prompts: list[RecordedPrompt],
) -> pd.DataFrame:
    return build_head_copy_rule_table(
        model,
        bundle,
        recorded_prompts,
        layer_index=1,
        head_index=0,
    )


def build_l2h0_copy_rule_summary_table(copy_rule_table: pd.DataFrame) -> pd.DataFrame:
    return build_head_copy_rule_summary_table(copy_rule_table)


def build_l2h0_copy_rule_family_stability_table(copy_rule_table: pd.DataFrame) -> pd.DataFrame:
    return build_head_copy_rule_family_stability_table(copy_rule_table)
