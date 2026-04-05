from __future__ import annotations

import itertools

import pandas as pd

from scripts.kv_algorithm_oracle import annotate_row, render_kv_prompt


def _make_sweep_row(
    *,
    base_row: dict[str, object],
    context_pairs: list[dict[str, object]],
    query_key: str,
    family_name: str,
    family_value: str,
    prompt_suffix: str,
) -> dict[str, object]:
    prompt = render_kv_prompt(context_pairs, query_key)
    matching_pair = None
    for pair in context_pairs:
        if pair["key"] == query_key:
            matching_pair = pair
            break
    if matching_pair is None:
        raise ValueError(f"Query key {query_key!r} does not appear in context pairs")

    base_id = str(base_row.get("id") or base_row.get("prompt_id") or base_row["prompt"])
    return {
        "id": f"{base_id}::{prompt_suffix}",
        "base_prompt_id": base_id,
        "task": base_row.get("task", "kv_retrieve_3"),
        "split": str(base_row.get("split") or "synthetic_sweep"),
        "source_kind": "sweep",
        "family_name": family_name,
        "family_value": family_value,
        "num_pairs": len(context_pairs),
        "prompt": prompt,
        "target": matching_pair["value"],
        "query_key": query_key,
        "context_pairs": [
            {
                "key": str(pair["key"]),
                "value": str(pair["value"]),
                "pair_index": pair_index,
            }
            for pair_index, pair in enumerate(context_pairs)
        ],
    }


def generate_query_key_sweep(base_row: dict[str, object]) -> list[dict[str, object]]:
    annotation = annotate_row(base_row)
    rows: list[dict[str, object]] = []
    context_pairs = list(base_row["context_pairs"])
    for slot_index, query_key in enumerate(annotation.slot_keys):
        rows.append(
            _make_sweep_row(
                base_row=base_row,
                context_pairs=context_pairs,
                query_key=query_key,
                family_name="query_key_sweep",
                family_value=query_key,
                prompt_suffix=f"query_key_{slot_index}_{query_key}",
            )
        )
    return rows


def generate_slot_permutation_sweep(base_row: dict[str, object]) -> list[dict[str, object]]:
    annotation = annotate_row(base_row)
    context_pairs = list(base_row["context_pairs"])
    rows: list[dict[str, object]] = []
    for permutation in itertools.permutations(range(len(context_pairs))):
        permuted_pairs = [dict(context_pairs[index]) for index in permutation]
        prompt_suffix = "slot_perm_" + "".join(str(index) for index in permutation)
        rows.append(
            _make_sweep_row(
                base_row=base_row,
                context_pairs=permuted_pairs,
                query_key=annotation.query_key,
                family_name="slot_permutation",
                family_value="-".join(str(index) for index in permutation),
                prompt_suffix=prompt_suffix,
            )
        )
    return rows


def generate_value_permutation_sweep(base_row: dict[str, object]) -> list[dict[str, object]]:
    annotation = annotate_row(base_row)
    context_pairs = [dict(pair) for pair in base_row["context_pairs"]]
    base_values = [pair["value"] for pair in context_pairs]
    rows: list[dict[str, object]] = []
    for permutation in itertools.permutations(range(len(context_pairs))):
        permuted_pairs = []
        for slot_index, pair in enumerate(context_pairs):
            permuted_pair = dict(pair)
            permuted_pair["value"] = base_values[permutation[slot_index]]
            permuted_pairs.append(permuted_pair)
        prompt_suffix = "value_perm_" + "".join(str(index) for index in permutation)
        rows.append(
            _make_sweep_row(
                base_row=base_row,
                context_pairs=permuted_pairs,
                query_key=annotation.query_key,
                family_name="value_permutation",
                family_value="-".join(str(index) for index in permutation),
                prompt_suffix=prompt_suffix,
            )
        )
    return rows


def generate_same_answer_different_slot_sweep(base_row: dict[str, object]) -> list[dict[str, object]]:
    annotation = annotate_row(base_row)
    context_pairs = [dict(pair) for pair in base_row["context_pairs"]]
    rows: list[dict[str, object]] = []
    for target_slot in range(len(context_pairs)):
        if target_slot == annotation.matching_slot:
            continue
        swapped_pairs = [dict(pair) for pair in context_pairs]
        swapped_pairs[annotation.matching_slot], swapped_pairs[target_slot] = (
            swapped_pairs[target_slot],
            swapped_pairs[annotation.matching_slot],
        )
        rows.append(
            _make_sweep_row(
                base_row=base_row,
                context_pairs=swapped_pairs,
                query_key=annotation.query_key,
                family_name="same_answer_different_slot",
                family_value=str(target_slot),
                prompt_suffix=f"same_answer_slot_{target_slot}",
            )
        )
    return rows


def generate_same_slot_different_answer_sweep(base_row: dict[str, object]) -> list[dict[str, object]]:
    annotation = annotate_row(base_row)
    context_pairs = [dict(pair) for pair in base_row["context_pairs"]]
    rows: list[dict[str, object]] = []
    for alternate_slot in range(len(context_pairs)):
        if alternate_slot == annotation.matching_slot:
            continue
        swapped_pairs = [dict(pair) for pair in context_pairs]
        swapped_pairs[annotation.matching_slot], swapped_pairs[alternate_slot] = (
            swapped_pairs[alternate_slot],
            swapped_pairs[annotation.matching_slot],
        )
        alternate_query_key = str(swapped_pairs[annotation.matching_slot]["key"])
        rows.append(
            _make_sweep_row(
                base_row=base_row,
                context_pairs=swapped_pairs,
                query_key=alternate_query_key,
                family_name="same_slot_different_answer",
                family_value=str(alternate_slot),
                prompt_suffix=f"same_slot_answer_{alternate_slot}",
            )
        )
    return rows


def generate_longer_context_ood_sweep(ood_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not ood_rows:
        raise ValueError("Expected at least one OOD row for longer-context sweeps")
    rows: list[dict[str, object]] = []
    for row in ood_rows:
        annotation = annotate_row(row)
        rows.append(
            {
                **row,
                "id": f"{annotation.prompt_id}::longer_context_ood",
                "base_prompt_id": annotation.prompt_id,
                "source_kind": "sweep",
                "family_name": "longer_context_ood",
                "family_value": str(annotation.num_pairs),
                "split": row.get("split", "test_ood_4_pairs"),
            }
        )
    return rows


def generate_controlled_sweeps(
    base_rows: list[dict[str, object]],
    *,
    longer_context_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not base_rows:
        raise ValueError("Expected at least one base row to generate controlled sweeps")

    sweep_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for base_row in base_rows:
        families = (
            generate_query_key_sweep(base_row)
            + generate_slot_permutation_sweep(base_row)
            + generate_value_permutation_sweep(base_row)
            + generate_same_answer_different_slot_sweep(base_row)
            + generate_same_slot_different_answer_sweep(base_row)
        )
        for row in families:
            row_id = str(row["id"])
            if row_id in seen_ids:
                raise ValueError(f"Duplicate sweep row id generated: {row_id}")
            seen_ids.add(row_id)
            sweep_rows.append(row)
    if longer_context_rows is not None:
        for row in generate_longer_context_ood_sweep(longer_context_rows):
            row_id = str(row["id"])
            if row_id in seen_ids:
                raise ValueError(f"Duplicate longer-context sweep row id generated: {row_id}")
            seen_ids.add(row_id)
            sweep_rows.append(row)
    return sweep_rows


def build_sweep_summary_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Expected at least one sweep row")
    summary_rows = [
        {
            "prompt_id": annotation.prompt_id,
            "base_prompt_id": annotation.base_prompt_id,
            "family_name": annotation.family_name,
            "family_value": annotation.family_value,
            "query_key": annotation.query_key,
            "matching_slot": annotation.matching_slot,
            "selected_value": annotation.selected_value,
            "slot_keys": " | ".join(annotation.slot_keys),
            "slot_values": " | ".join(annotation.slot_values),
            "prompt": annotation.prompt,
        }
        for annotation in (annotate_row(row) for row in rows)
    ]
    return pd.DataFrame(summary_rows)
