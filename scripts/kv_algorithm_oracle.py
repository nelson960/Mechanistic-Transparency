from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PromptAnnotation:
    prompt_id: str
    prompt: str
    target: str
    num_pairs: int
    query_key: str
    selected_value: str
    matching_slot: int
    slot_keys: tuple[str, ...]
    slot_values: tuple[str, ...]
    key_positions: tuple[int, ...]
    value_positions: tuple[int, ...]
    matching_key_position: int
    matching_value_position: int
    query_marker_position: int
    query_key_position: int
    arrow_position: int
    token_roles: tuple[str, ...]
    token_slot_indices: tuple[int | None, ...]
    source_kind: str
    family_name: str
    family_value: str
    split: str
    base_prompt_id: str


RESERVED_PROMPT_TOKENS = {"<bos>", ";", "Q", "->"}


def _validate_slot_token(token: str, *, role: str) -> None:
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Expected a non-empty {role} token, got {token!r}")
    if token != token.strip():
        raise ValueError(f"{role} token must not contain leading or trailing whitespace: {token!r}")
    if any(character.isspace() for character in token):
        raise ValueError(f"{role} token must be a single token without whitespace: {token!r}")
    if token in RESERVED_PROMPT_TOKENS:
        raise ValueError(f"{role} token {token!r} conflicts with reserved prompt syntax tokens")


def render_kv_prompt(context_pairs: list[dict[str, object]], query_key: str) -> str:
    if not context_pairs:
        raise ValueError("Expected at least one context pair to render a KV prompt")
    _validate_slot_token(query_key, role="query key")

    segments: list[str] = ["<bos>"]
    seen_keys: set[str] = set()
    seen_values: set[str] = set()
    for pair_index, pair in enumerate(context_pairs):
        key = pair["key"]
        value = pair["value"]
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Context pair {pair_index} must contain string key/value fields")
        _validate_slot_token(key, role="key")
        _validate_slot_token(value, role="value")
        if key in seen_keys:
            raise ValueError(f"Duplicate key {key!r} in rendered prompt pairs")
        if value in seen_values:
            raise ValueError(f"Duplicate value {value!r} in rendered prompt pairs")
        seen_keys.add(key)
        seen_values.add(value)
        segments.extend([key, value, ";"])

    if query_key not in seen_keys:
        raise ValueError(
            f"Query key {query_key!r} does not exist in the rendered context keys {sorted(seen_keys)}"
        )

    segments.extend(["Q", query_key, "->"])
    return " ".join(segments)


def parse_kv_prompt(
    prompt: str,
    target: str,
    *,
    prompt_id: str,
    source_kind: str,
    family_name: str,
    family_value: str,
    split: str,
    base_prompt_id: str,
) -> PromptAnnotation:
    tokens = prompt.split()
    if len(tokens) < 7:
        raise ValueError(f"Prompt is too short to be a KV prompt: {prompt!r}")
    if tokens[0] != "<bos>":
        raise ValueError(f"Prompt must start with <bos>: {prompt!r}")
    if tokens[-3] != "Q" or tokens[-1] != "->":
        raise ValueError(f"Prompt must end with 'Q <key> ->': {prompt!r}")

    query_key = tokens[-2]
    _validate_slot_token(query_key, role="query key")
    _validate_slot_token(target, role="target")

    context_tokens = tokens[1:-3]
    if len(context_tokens) % 3 != 0:
        raise ValueError(
            "Context token span must be a repeated '<key> <value> ;' sequence, "
            f"got {context_tokens!r}"
        )

    slot_keys: list[str] = []
    slot_values: list[str] = []
    key_positions: list[int] = []
    value_positions: list[int] = []
    token_roles = ["other"] * len(tokens)
    token_slot_indices: list[int | None] = [None] * len(tokens)
    token_roles[0] = "bos"

    seen_keys: set[str] = set()
    seen_values: set[str] = set()
    for offset in range(0, len(context_tokens), 3):
        slot_index = offset // 3
        key = context_tokens[offset]
        value = context_tokens[offset + 1]
        separator = context_tokens[offset + 2]
        _validate_slot_token(key, role="key")
        _validate_slot_token(value, role="value")
        if separator != ";":
            raise ValueError(
                f"Expected ';' after slot {slot_index}, got {separator!r} in prompt {prompt!r}"
            )
        if key in seen_keys:
            raise ValueError(f"Duplicate key {key!r} in prompt {prompt!r}")
        if value in seen_values:
            raise ValueError(f"Duplicate value {value!r} in prompt {prompt!r}")
        seen_keys.add(key)
        seen_values.add(value)
        slot_keys.append(key)
        slot_values.append(value)
        key_position = 1 + offset
        value_position = key_position + 1
        separator_position = key_position + 2
        key_positions.append(key_position)
        value_positions.append(value_position)
        token_roles[key_position] = f"slot_{slot_index}_key"
        token_roles[value_position] = f"slot_{slot_index}_value"
        token_roles[separator_position] = f"slot_{slot_index}_separator"
        token_slot_indices[key_position] = slot_index
        token_slot_indices[value_position] = slot_index
        token_slot_indices[separator_position] = slot_index

    if query_key not in slot_keys:
        raise ValueError(f"Query key {query_key!r} is not present in context keys {slot_keys}")
    matching_slot = slot_keys.index(query_key)
    selected_value = slot_values[matching_slot]
    if target != selected_value:
        raise ValueError(
            f"Target {target!r} does not match selected value {selected_value!r} for prompt {prompt!r}"
        )

    query_marker_position = len(tokens) - 3
    query_key_position = len(tokens) - 2
    arrow_position = len(tokens) - 1
    token_roles[query_marker_position] = "query_marker"
    token_roles[query_key_position] = "query_key"
    token_roles[arrow_position] = "arrow"

    return PromptAnnotation(
        prompt_id=prompt_id,
        prompt=prompt,
        target=target,
        num_pairs=len(slot_keys),
        query_key=query_key,
        selected_value=selected_value,
        matching_slot=matching_slot,
        slot_keys=tuple(slot_keys),
        slot_values=tuple(slot_values),
        key_positions=tuple(key_positions),
        value_positions=tuple(value_positions),
        matching_key_position=key_positions[matching_slot],
        matching_value_position=value_positions[matching_slot],
        query_marker_position=query_marker_position,
        query_key_position=query_key_position,
        arrow_position=arrow_position,
        token_roles=tuple(token_roles),
        token_slot_indices=tuple(token_slot_indices),
        source_kind=source_kind,
        family_name=family_name,
        family_value=family_value,
        split=split,
        base_prompt_id=base_prompt_id,
    )


def annotate_row(
    row: dict[str, object],
    *,
    prompt_id: str | None = None,
    source_kind: str | None = None,
    family_name: str | None = None,
    family_value: str | None = None,
    split: str | None = None,
    base_prompt_id: str | None = None,
) -> PromptAnnotation:
    prompt = row.get("prompt")
    target = row.get("target")
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise ValueError(f"Row must contain string prompt/target fields, got {row!r}")

    resolved_prompt_id = prompt_id or str(row.get("id") or row.get("prompt_id") or prompt)
    resolved_source_kind = source_kind or str(row.get("source_kind") or "dataset")
    resolved_family_name = family_name or str(row.get("family_name") or resolved_source_kind)
    resolved_family_value = family_value or str(row.get("family_value") or "original")
    resolved_split = split or str(row.get("split") or resolved_source_kind)
    resolved_base_prompt_id = base_prompt_id or str(row.get("base_prompt_id") or resolved_prompt_id)

    return parse_kv_prompt(
        prompt,
        target,
        prompt_id=resolved_prompt_id,
        source_kind=resolved_source_kind,
        family_name=resolved_family_name,
        family_value=resolved_family_value,
        split=resolved_split,
        base_prompt_id=resolved_base_prompt_id,
    )


def annotation_to_dict(annotation: PromptAnnotation) -> dict[str, object]:
    return {
        "prompt_id": annotation.prompt_id,
        "prompt": annotation.prompt,
        "target": annotation.target,
        "num_pairs": annotation.num_pairs,
        "query_key": annotation.query_key,
        "selected_value": annotation.selected_value,
        "matching_slot": annotation.matching_slot,
        "slot_keys": list(annotation.slot_keys),
        "slot_values": list(annotation.slot_values),
        "key_positions": list(annotation.key_positions),
        "value_positions": list(annotation.value_positions),
        "matching_key_position": annotation.matching_key_position,
        "matching_value_position": annotation.matching_value_position,
        "query_marker_position": annotation.query_marker_position,
        "query_key_position": annotation.query_key_position,
        "arrow_position": annotation.arrow_position,
        "token_roles": list(annotation.token_roles),
        "token_slot_indices": list(annotation.token_slot_indices),
        "source_kind": annotation.source_kind,
        "family_name": annotation.family_name,
        "family_value": annotation.family_value,
        "split": annotation.split,
        "base_prompt_id": annotation.base_prompt_id,
    }


def build_prompt_annotation_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Expected at least one row to annotate")
    return pd.DataFrame([annotation_to_dict(annotate_row(row)) for row in rows])


def build_position_annotation_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Expected at least one row to annotate by position")

    position_rows: list[dict[str, object]] = []
    for row in rows:
        annotation = annotate_row(row)
        tokens = annotation.prompt.split()
        for position, token in enumerate(tokens):
            position_rows.append(
                {
                    "prompt_id": annotation.prompt_id,
                    "base_prompt_id": annotation.base_prompt_id,
                    "family_name": annotation.family_name,
                    "family_value": annotation.family_value,
                    "position": position,
                    "token": token,
                    "token_role": annotation.token_roles[position],
                    "slot_index": annotation.token_slot_indices[position],
                    "is_matching_key_position": position == annotation.matching_key_position,
                    "is_matching_value_position": position == annotation.matching_value_position,
                    "query_key": annotation.query_key,
                    "selected_value": annotation.selected_value,
                    "matching_slot": annotation.matching_slot,
                }
            )
    return pd.DataFrame(position_rows)


def load_and_annotate_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    from scripts.kv_retrieve_analysis import load_dataset_bundle

    bundle = load_dataset_bundle(dataset_dir)
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split: {split}")
    return build_prompt_annotation_table(bundle.raw_splits[split])
