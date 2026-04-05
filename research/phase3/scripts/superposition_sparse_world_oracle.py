from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ATTRIBUTE_FAMILIES = ["T", "C", "P", "S", "R"]
QUERY_FAMILIES = ["C", "P", "S"]
VARIABLE_NAMES = [
    "query_color",
    "query_position",
    "query_state",
    "matched_entity",
    "selected_label",
]


@dataclass(frozen=True)
class SuperpositionAnnotation:
    prompt: str
    target: str
    query_color: str
    query_position: str
    query_state: str
    matched_entity_id: str
    matched_entity_index: int
    selected_label: str
    num_entities: int


def annotate_row(row: dict[str, Any]) -> SuperpositionAnnotation:
    required = {
        "prompt",
        "target",
        "query_attributes",
        "matched_entity_id",
        "matched_entity_index",
        "entities",
        "num_entities",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Row is missing required keys: {missing}")
    query_attributes = row["query_attributes"]
    if not isinstance(query_attributes, dict):
        raise ValueError("Expected row['query_attributes'] to be an object")
    for family in QUERY_FAMILIES:
        value = query_attributes.get(family)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Expected non-empty query attribute for family {family!r}")
    return SuperpositionAnnotation(
        prompt=str(row["prompt"]),
        target=str(row["target"]),
        query_color=str(query_attributes["C"]),
        query_position=str(query_attributes["P"]),
        query_state=str(query_attributes["S"]),
        matched_entity_id=str(row["matched_entity_id"]),
        matched_entity_index=int(row["matched_entity_index"]),
        selected_label=str(row["target"]),
        num_entities=int(row["num_entities"]),
    )


def build_variable_payload(row: dict[str, Any]) -> dict[str, str | int]:
    annotation = annotate_row(row)
    return {
        "query_color": annotation.query_color,
        "query_position": annotation.query_position,
        "query_state": annotation.query_state,
        "matched_entity": annotation.matched_entity_index,
        "selected_label": annotation.selected_label,
    }


def build_latent_feature_summary(row: dict[str, Any]) -> dict[str, Any]:
    if "entities" not in row or not isinstance(row["entities"], list):
        raise ValueError("Expected row['entities'] to be a list")
    per_entity = []
    for entity in row["entities"]:
        if not isinstance(entity, dict):
            raise ValueError("Expected each entity to be an object")
        per_entity.append(
            {
                "entity_id": str(entity["entity_id"]),
                "type": str(entity["T"]),
                "color": str(entity["C"]),
                "position": str(entity["P"]),
                "state": str(entity["S"]),
                "role": str(entity["R"]),
                "label": str(entity["Y"]),
            }
        )
    return {
        "variable_payload": build_variable_payload(row),
        "entities": per_entity,
        "active_latent_features": list(row.get("active_latent_features", [])),
    }
