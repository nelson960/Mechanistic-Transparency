from __future__ import annotations

import pandas as pd


def _slot_to_value(slot_values: list[str], slot_index: int) -> str:
    if slot_index < 0 or slot_index >= len(slot_values):
        raise ValueError(
            f"Predicted slot index {slot_index} is out of range for slot values {slot_values}"
        )
    return slot_values[slot_index]


def _build_strategy_predictions(
    merged: pd.DataFrame,
    *,
    strategy_name: str,
    slot_column: str,
    operator_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        slot_values = row["slot_values"]
        if not isinstance(slot_values, list):
            raise ValueError(
                f"Expected slot_values to be a list in program scoring, got {type(slot_values).__name__}"
            )
        predicted_slot = int(row[slot_column])
        predicted_value = _slot_to_value(slot_values, predicted_slot)
        rows.append(
            {
                "strategy": strategy_name,
                "prompt_id": row["prompt_id"],
                "base_prompt_id": row["base_prompt_id"],
                "predicted_slot": predicted_slot,
                "expected_slot": int(row["matching_slot"]),
                "slot_correct": predicted_slot == int(row["matching_slot"]),
                "predicted_value": predicted_value,
                "expected_value": row["selected_value"],
                "value_correct": predicted_value == row["selected_value"],
                "operator_count": operator_count,
            }
        )
    return pd.DataFrame(rows)


def build_program_prediction_table(
    metadata: pd.DataFrame,
    l2h1_attention_table: pd.DataFrame,
    l2h0_attention_table: pd.DataFrame,
) -> pd.DataFrame:
    if metadata.empty:
        raise ValueError("Expected non-empty metadata for program scoring")
    if l2h1_attention_table.empty or l2h0_attention_table.empty:
        raise ValueError("Expected non-empty L2H1 and L2H0 operator tables for program scoring")

    merged = metadata.merge(
        l2h1_attention_table[["prompt_id", "top_key_slot", "top_total_slot"]].rename(
            columns={
                "top_key_slot": "l2h1_top_key_slot",
                "top_total_slot": "l2h1_top_total_slot",
            }
        ),
        on="prompt_id",
        how="inner",
    ).merge(
        l2h0_attention_table[["prompt_id", "top_value_slot", "top_total_slot"]].rename(
            columns={
                "top_value_slot": "l2h0_top_value_slot",
                "top_total_slot": "l2h0_top_total_slot",
            }
        ),
        on="prompt_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("Prompt-id join between metadata and operator tables produced zero rows")

    strategy_tables = [
        _build_strategy_predictions(
            merged,
            strategy_name="l2h0_value_copy",
            slot_column="l2h0_top_value_slot",
            operator_count=1,
        ),
        _build_strategy_predictions(
            merged,
            strategy_name="l2h0_total_attention_copy",
            slot_column="l2h0_top_total_slot",
            operator_count=1,
        ),
        _build_strategy_predictions(
            merged,
            strategy_name="l2h1_key_route_then_copy",
            slot_column="l2h1_top_key_slot",
            operator_count=2,
        ),
        _build_strategy_predictions(
            merged,
            strategy_name="l2h1_total_route_then_copy",
            slot_column="l2h1_top_total_slot",
            operator_count=2,
        ),
    ]
    return pd.concat(strategy_tables, ignore_index=True)


def build_program_score_table(program_prediction_table: pd.DataFrame) -> pd.DataFrame:
    if program_prediction_table.empty:
        raise ValueError("Expected non-empty program prediction table")
    rows: list[dict[str, object]] = []
    for strategy, group in program_prediction_table.groupby("strategy"):
        rows.append(
            {
                "strategy": strategy,
                "rows": int(len(group)),
                "operator_count": int(group["operator_count"].iloc[0]),
                "slot_accuracy": float(group["slot_correct"].mean()),
                "value_accuracy": float(group["value_correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["value_accuracy", "slot_accuracy", "operator_count"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_program_family_score_table(program_prediction_table: pd.DataFrame) -> pd.DataFrame:
    if program_prediction_table.empty:
        raise ValueError("Expected non-empty program prediction table")
    rows: list[dict[str, object]] = []
    for (strategy, base_prompt_id), group in program_prediction_table.groupby(["strategy", "base_prompt_id"]):
        rows.append(
            {
                "strategy": strategy,
                "base_prompt_id": base_prompt_id,
                "rows": int(len(group)),
                "slot_accuracy": float(group["slot_correct"].mean()),
                "value_accuracy": float(group["value_correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "value_accuracy", "base_prompt_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
