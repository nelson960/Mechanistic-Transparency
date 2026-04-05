from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
import torch

from scripts.kv_algorithm_oracle import PromptAnnotation, annotate_row, annotation_to_dict
from scripts.kv_retrieve_analysis import (
    DatasetBundle,
    head_residual_contribution,
    run_prompt,
)


@dataclass(frozen=True)
class RecordedPrompt:
    row: dict[str, object]
    annotation: PromptAnnotation
    result: dict[str, object]
    cache: dict


@dataclass(frozen=True)
class RecordedSiteDataset:
    metadata: pd.DataFrame
    site_vectors: dict[str, torch.Tensor]


SITE_REGISTRY = {
    "block1_final_resid": "Block 1 residual stream after MLP at final position",
    "block1_final_l1h0": "L1H0 residual contribution at final position",
    "block1_final_mlp_out": "Block 1 MLP output at final position",
    "l2h0_final_q": "L2H0 query vector at final position",
    "l2h0_final_out": "L2H0 residual contribution at final position",
    "l2h1_final_out": "L2H1 residual contribution at final position",
    "final_hidden": "Final normalized hidden state at final position",
}

LEGACY_SITE_ALIASES = {
    "block1_final_resid": "block1_final_resid_after_mlp",
    "block1_final_l1h0": "block1_head0_final_resid_contribution",
    "block1_final_mlp_out": "block1_final_mlp_out",
    "l2h0_final_q": "block2_head0_final_q",
    "l2h0_final_out": "block2_head0_final_resid_contribution",
    "l2h1_final_out": "block2_head1_final_resid_contribution",
    "final_hidden": "final_hidden",
}

HEAD_SITE_PATTERN = re.compile(
    r"^block(?P<block>\d+)_head(?P<head>\d+)_final_(?P<kind>q|k|v|head_out|resid_contribution)$"
)
BLOCK_SITE_PATTERN = re.compile(
    r"^block(?P<block>\d+)_final_(?P<kind>resid_after_mlp|mlp_out)$"
)


def build_final_position_site_registry(model: torch.nn.Module) -> dict[str, str]:
    if not hasattr(model, "blocks"):
        raise ValueError("Expected model to expose transformer blocks for site registry construction")

    registry: dict[str, str] = {}
    for block_index, block in enumerate(model.blocks, start=1):
        registry[f"block{block_index}_final_resid_after_mlp"] = (
            f"Block {block_index} residual stream after MLP at final position"
        )
        registry[f"block{block_index}_final_mlp_out"] = (
            f"Block {block_index} MLP output at final position"
        )
        for head_index in range(block.attn.n_heads):
            registry[f"block{block_index}_head{head_index}_final_q"] = (
                f"Block {block_index} head {head_index} query vector at final position"
            )
            registry[f"block{block_index}_head{head_index}_final_k"] = (
                f"Block {block_index} head {head_index} key vector at final position"
            )
            registry[f"block{block_index}_head{head_index}_final_v"] = (
                f"Block {block_index} head {head_index} value vector at final position"
            )
            registry[f"block{block_index}_head{head_index}_final_head_out"] = (
                f"Block {block_index} head {head_index} attention output before O projection at final position"
            )
            registry[f"block{block_index}_head{head_index}_final_resid_contribution"] = (
                f"Block {block_index} head {head_index} residual contribution at final position"
            )
    registry["final_hidden"] = "Final normalized hidden state at final position"
    return registry


def build_final_position_site_list(model: torch.nn.Module) -> list[str]:
    return list(build_final_position_site_registry(model).keys())


def _resolve_site_alias(site: str) -> str:
    return LEGACY_SITE_ALIASES.get(site, site)


def resolve_site_name(site: str) -> str:
    return _resolve_site_alias(site)


def extract_site_vector(model: torch.nn.Module, cache: dict, site: str) -> torch.Tensor:
    site = _resolve_site_alias(site)
    if site == "final_hidden":
        return cache["final_hidden"][0, -1].detach().cpu()

    block_match = BLOCK_SITE_PATTERN.match(site)
    if block_match is not None:
        block_index = int(block_match.group("block")) - 1
        if block_index < 0 or block_index >= len(model.blocks):
            raise ValueError(f"Invalid block index in site {site!r}")
        kind = block_match.group("kind")
        if kind == "resid_after_mlp":
            return cache["blocks"][block_index]["resid_after_mlp"][0, -1].detach().cpu()
        if kind == "mlp_out":
            return cache["blocks"][block_index]["mlp"]["out"][0, -1].detach().cpu()
        raise ValueError(f"Unsupported block site kind {kind!r}")

    head_match = HEAD_SITE_PATTERN.match(site)
    if head_match is not None:
        block_index = int(head_match.group("block")) - 1
        head_index = int(head_match.group("head"))
        if block_index < 0 or block_index >= len(model.blocks):
            raise ValueError(f"Invalid block index in site {site!r}")
        if head_index < 0 or head_index >= model.blocks[block_index].attn.n_heads:
            raise ValueError(f"Invalid head index in site {site!r}")
        kind = head_match.group("kind")
        attention_cache = cache["blocks"][block_index]["attention"]
        if kind == "q":
            return attention_cache["q"][0, head_index, -1].detach().cpu()
        if kind == "k":
            return attention_cache["k"][0, head_index, -1].detach().cpu()
        if kind == "v":
            return attention_cache["v"][0, head_index, -1].detach().cpu()
        if kind == "head_out":
            return attention_cache["head_out"][0, head_index, -1].detach().cpu()
        if kind == "resid_contribution":
            return head_residual_contribution(model, cache, layer_index=block_index, head_index=head_index)[
                0, -1
            ].detach().cpu()
        raise ValueError(f"Unsupported head site kind {kind!r}")

    raise ValueError(f"Unsupported site {site!r}")


def record_prompt_rows(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    rows: list[dict[str, object]],
    device: torch.device,
) -> list[RecordedPrompt]:
    if not rows:
        raise ValueError("Expected at least one row to record")

    recorded_prompts: list[RecordedPrompt] = []
    for row in rows:
        annotation = annotate_row(row)
        result, cache = run_prompt(
            model,
            bundle,
            annotation.prompt,
            device=device,
            expected_target=annotation.target,
            return_cache=True,
        )
        if cache is None:
            raise ValueError(f"Expected cache for prompt {annotation.prompt_id}")
        recorded_prompts.append(
            RecordedPrompt(
                row=row,
                annotation=annotation,
                result=result,
                cache=cache,
            )
        )
    return recorded_prompts


def build_site_dataset(
    model: torch.nn.Module,
    recorded_prompts: list[RecordedPrompt],
    sites: list[str],
) -> RecordedSiteDataset:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt")
    if not sites:
        raise ValueError("Expected at least one site to build a recorded dataset")

    metadata_rows: list[dict[str, object]] = []
    site_vectors: dict[str, list[torch.Tensor]] = {site: [] for site in sites}

    for recorded in recorded_prompts:
        annotation_dict = annotation_to_dict(recorded.annotation)
        row_metadata = {
            **annotation_dict,
            "predicted_token": recorded.result["predicted_token"],
            "correct": bool(recorded.result.get("correct", False)),
            "margin": float(recorded.result.get("margin", float("nan"))),
        }
        metadata_rows.append(row_metadata)
        for site in sites:
            site_vectors[site].append(extract_site_vector(model, recorded.cache, site))

    stacked_site_vectors: dict[str, torch.Tensor] = {}
    for site, vectors in site_vectors.items():
        if not vectors:
            raise ValueError(f"No vectors collected for site {site}")
        stacked_site_vectors[site] = torch.stack(vectors)

    return RecordedSiteDataset(
        metadata=pd.DataFrame(metadata_rows),
        site_vectors=stacked_site_vectors,
    )


def build_recording_summary_table(recorded_prompts: list[RecordedPrompt]) -> pd.DataFrame:
    if not recorded_prompts:
        raise ValueError("Expected at least one recorded prompt")
    return pd.DataFrame(
        [
            {
                "prompt_id": recorded.annotation.prompt_id,
                "base_prompt_id": recorded.annotation.base_prompt_id,
                "family_name": recorded.annotation.family_name,
                "family_value": recorded.annotation.family_value,
                "query_key": recorded.annotation.query_key,
                "matching_slot": recorded.annotation.matching_slot,
                "selected_value": recorded.annotation.selected_value,
                "predicted_token": recorded.result["predicted_token"],
                "correct": bool(recorded.result.get("correct", False)),
                "margin": float(recorded.result.get("margin", float("nan"))),
                "prompt": recorded.annotation.prompt,
            }
            for recorded in recorded_prompts
        ]
    )
