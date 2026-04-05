from __future__ import annotations

import itertools
from typing import Any

from research.phase3.scripts.superposition_sparse_world_oracle import QUERY_FAMILIES


def render_prompt(entities: list[dict[str, str]], query_attributes: dict[str, str]) -> str:
    entity_chunks = [
        " ".join(
            [
                entity["entity_id"],
                entity["T"],
                entity["C"],
                entity["P"],
                entity["S"],
                entity["R"],
                entity["Y"],
            ]
        )
        for entity in entities
    ]
    query_chunk = " ".join(["Q"] + [query_attributes[family] for family in QUERY_FAMILIES] + ["->"])
    return "<bos> " + " ; ".join(entity_chunks) + " ; " + query_chunk


def _clone_row(
    *,
    base_row: dict[str, Any],
    entities: list[dict[str, str]],
    query_attributes: dict[str, str],
    family_name: str,
    family_value: str,
    prompt_suffix: str,
) -> dict[str, Any]:
    matched = [
        entity
        for entity in entities
        if all(entity[family] == query_attributes[family] for family in QUERY_FAMILIES)
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one matched entity after sweep construction, found {len(matched)}"
        )
    matched_entity = matched[0]
    row_id = f"{base_row['id']}::{family_name}::{prompt_suffix}"
    return {
        **base_row,
        "id": row_id,
        "family_name": family_name,
        "family_value": family_value,
        "prompt": render_prompt(entities, query_attributes),
        "target": matched_entity["Y"],
        "query_attributes": dict(query_attributes),
        "matched_entity_id": matched_entity["entity_id"],
        "matched_entity_index": int(matched_entity["entity_index"]),
        "entities": entities,
        "active_latent_features": list(base_row.get("active_latent_features", [])),
    }


def generate_query_attribute_sweep(base_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = [dict(entity) for entity in base_row["entities"]]
    for entity in entities:
        query_attributes = {family: entity[family] for family in QUERY_FAMILIES}
        rows.append(
            _clone_row(
                base_row=base_row,
                entities=[dict(item) for item in entities],
                query_attributes=query_attributes,
                family_name="query_attribute_sweep",
                family_value="|".join(query_attributes[family] for family in QUERY_FAMILIES),
                prompt_suffix=entity["entity_id"],
            )
        )
    return rows


def generate_entity_permutation_sweep(base_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = [dict(entity) for entity in base_row["entities"]]
    for permutation in itertools.permutations(range(len(entities))):
        permuted = [dict(entities[index]) for index in permutation]
        for entity_index, entity in enumerate(permuted):
            entity["entity_index"] = str(entity_index)
        rows.append(
            _clone_row(
                base_row=base_row,
                entities=permuted,
                query_attributes=dict(base_row["query_attributes"]),
                family_name="entity_permutation",
                family_value="-".join(str(index) for index in permutation),
                prompt_suffix="perm_" + "_".join(str(index) for index in permutation),
            )
        )
    return rows


def generate_label_permutation_sweep(base_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = [dict(entity) for entity in base_row["entities"]]
    labels = [entity["Y"] for entity in entities]
    for permutation in itertools.permutations(range(len(labels))):
        permuted = [dict(entity) for entity in entities]
        for entity_index, source_index in enumerate(permutation):
            permuted[entity_index]["Y"] = labels[source_index]
        rows.append(
            _clone_row(
                base_row=base_row,
                entities=permuted,
                query_attributes=dict(base_row["query_attributes"]),
                family_name="label_permutation",
                family_value="-".join(str(index) for index in permutation),
                prompt_suffix="label_" + "_".join(str(index) for index in permutation),
            )
        )
    return rows


def generate_controlled_sweeps(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sweep_rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        sweep_rows.extend(generate_query_attribute_sweep(base_row))
        sweep_rows.extend(generate_entity_permutation_sweep(base_row))
        sweep_rows.extend(generate_label_permutation_sweep(base_row))
    return sweep_rows
