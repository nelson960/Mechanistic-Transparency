from __future__ import annotations

import re

import pandas as pd
import torch

from scripts.kv_algorithm_record import RecordedPrompt, resolve_site_name
from scripts.kv_retrieve_analysis import (
    DatasetBundle,
    forward_with_activation_patch,
    forward_with_qkv_patch,
    summarize_logits_against_target,
)


BLOCK_SITE_PATTERN = re.compile(
    r"^block(?P<block>\d+)_final_(?P<kind>resid_after_mlp|mlp_out)$"
)
HEAD_SITE_PATTERN = re.compile(
    r"^block(?P<block>\d+)_head(?P<head>\d+)_final_(?P<kind>q|k|v|head_out|resid_contribution)$"
)

VARIABLE_FAMILY_MAP = {
    "query_key": "query_key_sweep",
    "matching_slot": "query_key_sweep",
    "selected_value": "same_slot_different_answer",
}


def _encode_prompt(prompt: str, token_to_id: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[token_to_id[token] for token in prompt.split()]],
        dtype=torch.long,
        device=device,
    )


def _patched_final_logits(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    base_recorded: RecordedPrompt,
    source_recorded: RecordedPrompt,
    site: str,
    device: torch.device,
) -> torch.Tensor:
    resolved_site = resolve_site_name(site)
    input_ids = _encode_prompt(base_recorded.annotation.prompt, bundle.token_to_id, device)

    if resolved_site == "final_hidden":
        base_final_hidden = base_recorded.cache["final_hidden"].to(
            device=model.token_embed.weight.device,
            dtype=model.token_embed.weight.dtype,
        )
        source_final_hidden = source_recorded.cache["final_hidden"].to(
            device=base_final_hidden.device,
            dtype=base_final_hidden.dtype,
        )
        patched_final_hidden = base_final_hidden.clone()
        patched_final_hidden[:, -1, :] = source_final_hidden[:, -1, :]
        return (patched_final_hidden @ model.token_embed.weight.T)[0, -1].detach().cpu()

    block_match = BLOCK_SITE_PATTERN.match(resolved_site)
    if block_match is not None:
        layer_index = int(block_match.group("block")) - 1
        kind = block_match.group("kind")
        patch_kind = {
            "resid_after_mlp": "resid_after_block",
            "mlp_out": "mlp_out",
        }[kind]
        logits = forward_with_activation_patch(
            model,
            input_ids,
            source_recorded.cache,
            patch={
                "kind": patch_kind,
                "layer_index": layer_index,
            },
        )
        return logits[0, -1].detach().cpu()

    head_match = HEAD_SITE_PATTERN.match(resolved_site)
    if head_match is None:
        raise ValueError(f"Unsupported site for faithfulness analysis: {resolved_site}")

    layer_index = int(head_match.group("block")) - 1
    head_index = int(head_match.group("head"))
    kind = head_match.group("kind")
    if kind in {"q", "k", "v"}:
        logits = forward_with_qkv_patch(
            model,
            input_ids,
            source_recorded.cache,
            base_recorded.cache,
            destination={
                "layer_index": layer_index,
                "head_index": head_index,
            },
            components=[kind],
        )
        return logits[0, -1].detach().cpu()

    patch_kind = "head_out" if kind in {"head_out", "resid_contribution"} else None
    if patch_kind is None:
        raise ValueError(f"Unsupported head site kind for faithfulness analysis: {kind}")
    logits = forward_with_activation_patch(
        model,
        input_ids,
        source_recorded.cache,
        patch={
            "kind": patch_kind,
            "layer_index": layer_index,
            "head_index": head_index,
        },
    )
    return logits[0, -1].detach().cpu()


def score_site_interchange_prompt(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    base_recorded: RecordedPrompt,
    source_recorded: RecordedPrompt,
    site: str,
    device: torch.device,
) -> dict[str, object]:
    final_logits = _patched_final_logits(
        model,
        bundle,
        base_recorded=base_recorded,
        source_recorded=source_recorded,
        site=site,
        device=device,
    )
    return summarize_logits_against_target(
        final_logits,
        bundle,
        target_token=source_recorded.annotation.selected_value,
    )


def build_variable_faithfulness_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    recorded_prompts: list[RecordedPrompt],
    variable_best_sites: dict[str, str],
    *,
    device: torch.device,
) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt for variable faithfulness analysis")
    missing_variables = sorted(set(VARIABLE_FAMILY_MAP) - set(variable_best_sites))
    if missing_variables:
        raise ValueError(f"Missing best sites for variables: {missing_variables}")

    family_lookup: dict[str, list[RecordedPrompt]] = {}
    for recorded in recorded_prompts:
        family_lookup.setdefault(recorded.annotation.family_name, []).append(recorded)

    detail_rows: list[dict[str, object]] = []
    family_summary_rows: list[dict[str, object]] = []
    variable_summary_rows: list[dict[str, object]] = []
    for variable, family_name in VARIABLE_FAMILY_MAP.items():
        if family_name not in family_lookup:
            raise ValueError(
                f"Recorded prompts are missing the required faithfulness family {family_name!r} for {variable!r}"
            )
        site = variable_best_sites[variable]
        variable_rows: list[dict[str, object]] = []
        grouped_records: dict[str, list[RecordedPrompt]] = {}
        for recorded in family_lookup[family_name]:
            grouped_records.setdefault(recorded.annotation.base_prompt_id, []).append(recorded)
        for base_prompt_id, group_records in sorted(grouped_records.items()):
            if len(group_records) < 2:
                family_summary_rows.append(
                    {
                        "row_kind": "family_summary",
                        "variable": variable,
                        "site": site,
                        "family_name": family_name,
                        "base_prompt_id": str(base_prompt_id),
                        "rows": 0,
                        "family_accuracy": float("nan"),
                        "family_margin_mean": float("nan"),
                        "supported": False,
                        "reason": (
                            f"Faithfulness family {family_name!r} for base_prompt_id={base_prompt_id!r} "
                            "needs at least two prompt variants"
                        ),
                    }
                )
                continue
            group_detail_rows: list[dict[str, object]] = []
            for base_recorded in group_records:
                base_value = str(getattr(base_recorded.annotation, variable))
                for source_recorded in group_records:
                    source_value = str(getattr(source_recorded.annotation, variable))
                    if source_recorded.annotation.prompt_id == base_recorded.annotation.prompt_id:
                        continue
                    if source_value == base_value:
                        continue
                    result = score_site_interchange_prompt(
                        model,
                        bundle,
                        base_recorded=base_recorded,
                        source_recorded=source_recorded,
                        site=site,
                        device=device,
                    )
                    detail_row = {
                        "row_kind": "detail",
                        "variable": variable,
                        "site": site,
                        "family_name": family_name,
                        "base_prompt_id": str(base_prompt_id),
                        "base_prompt": base_recorded.annotation.prompt,
                        "source_prompt": source_recorded.annotation.prompt,
                        "base_variable_value": base_value,
                        "source_variable_value": source_value,
                        "expected_target": source_recorded.annotation.selected_value,
                        "predicted_token": str(result["predicted_token"]),
                        "target_token": str(result["target_token"]),
                        "foil_token": str(result["foil_token"]),
                        "margin": float(result["margin"]),
                        "correct": bool(result["correct"]),
                    }
                    detail_rows.append(detail_row)
                    group_detail_rows.append(detail_row)
                    variable_rows.append(detail_row)
            group_df = pd.DataFrame(group_detail_rows)
            family_summary_rows.append(
                {
                    "row_kind": "family_summary",
                    "variable": variable,
                    "site": site,
                    "family_name": family_name,
                    "base_prompt_id": str(base_prompt_id),
                    "rows": int(len(group_df)),
                    "family_accuracy": float(group_df["correct"].mean()) if not group_df.empty else float("nan"),
                    "family_margin_mean": float(group_df["margin"].mean()) if not group_df.empty else float("nan"),
                    "supported": bool(not group_df.empty),
                    "reason": None if not group_df.empty else "No valid source/base value swaps were available",
                }
            )

        variable_df = pd.DataFrame(variable_rows)
        family_df = pd.DataFrame(
            [
                row
                for row in family_summary_rows
                if row["variable"] == variable and bool(row.get("supported", True))
            ]
        )
        if variable_df.empty or family_df.empty:
            variable_summary_rows.append(
                {
                    "row_kind": "summary",
                    "variable": variable,
                    "site": site,
                    "family_name": family_name,
                    "pooled_score": float("nan"),
                    "pooled_margin_mean": float("nan"),
                    "family_min_score": float("nan"),
                    "family_mean_score": float("nan"),
                    "family_max_score": float("nan"),
                    "rows": 0,
                    "supported": False,
                    "reason": f"Faithfulness analysis produced no scored rows for variable {variable!r} at site {site!r}",
                }
            )
            continue
        variable_summary_rows.append(
            {
                "row_kind": "summary",
                "variable": variable,
                "site": site,
                "family_name": family_name,
                "pooled_score": float(variable_df["correct"].mean()),
                "pooled_margin_mean": float(variable_df["margin"].mean()),
                "family_min_score": float(family_df["family_accuracy"].min()),
                "family_mean_score": float(family_df["family_accuracy"].mean()),
                "family_max_score": float(family_df["family_accuracy"].max()),
                "rows": int(len(variable_df)),
                "supported": True,
                "reason": None,
            }
        )

    return pd.concat(
        [
            pd.DataFrame(detail_rows),
            pd.DataFrame(family_summary_rows),
            pd.DataFrame(variable_summary_rows),
        ],
        ignore_index=True,
        sort=False,
    )
