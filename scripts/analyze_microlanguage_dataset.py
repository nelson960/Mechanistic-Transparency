#!/usr/bin/env python3
"""Inspect microlanguage dataset integrity, split coverage, and shortcut baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate_microlanguage_world_dataset import parse_prompt


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


def canonical_world_key(row: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    parsed = parse_prompt(str(row["prompt"]))
    return tuple(
        sorted(
            (str(event["relation"]), str(event["subject"]), str(event["value"]))
            for event in parsed["events"]
        )
    )


def canonical_world_query_key(row: dict[str, Any]) -> tuple[tuple[tuple[str, str, str], ...], str, str]:
    parsed = parse_prompt(str(row["prompt"]))
    return (
        tuple(
            sorted(
                (str(event["relation"]), str(event["subject"]), str(event["value"]))
                for event in parsed["events"]
            )
        ),
        str(parsed["query_family"]),
        str(parsed["query_subject"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a microlanguage dataset for integrity and shortcut baselines.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset directory containing jsonl splits.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_matching_value(row: dict[str, Any]) -> str:
    trace = row.get("trace")
    if isinstance(trace, list) and trace:
        final_step = trace[-1]
        if isinstance(final_step, dict):
            value = final_step.get("value")
            if isinstance(value, str) and value.strip():
                return value
    parsed = parse_prompt(str(row["prompt"]))
    for event in parsed["events"]:
        relation = str(event["relation"])
        subject = str(event["subject"])
        value = str(event["value"])
        if relation == str(parsed["query_family"]) and subject == str(parsed["query_subject"]):
            return value
    raise ValueError(f"No matching event for query in row {row.get('id', '<unknown>')}")


def _row_target_group_tokens(row: dict[str, Any], metadata: dict[str, Any]) -> set[str]:
    query_family = str(row["query_family"])
    query_specs = _query_family_specs(metadata)
    vocab_groups = _vocabulary_groups(metadata)
    spec = query_specs.get(query_family)
    if not isinstance(spec, dict):
        raise ValueError(f"Unknown query_family {query_family!r}")
    target_group = spec.get("target_group")
    if not isinstance(target_group, str):
        raise ValueError(f"Query family {query_family!r} is missing target_group")
    group_tokens = vocab_groups.get(target_group)
    if not isinstance(group_tokens, list) or not group_tokens:
        raise ValueError(f"Target group {target_group!r} is missing or empty")
    return set(group_tokens)


def _active_target_group_values(row: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    parsed = parse_prompt(str(row["prompt"]))
    valid_tokens = _row_target_group_tokens(row, metadata)
    values = [str(event["value"]) for event in parsed["events"] if str(event["value"]) in valid_tokens]
    if not values:
        raise ValueError(f"No active target-group values found in prompt for row {row.get('id', '<unknown>')}")
    return values


def first_value(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    return _active_target_group_values(row, metadata)[0]


def last_value(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    return _active_target_group_values(row, metadata)[-1]


def majority_value(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    counts = Counter(_active_target_group_values(row, metadata))
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def matching_position(row: dict[str, Any]) -> int:
    parsed = parse_prompt(str(row["prompt"]))
    trace = row.get("trace")
    if isinstance(trace, list) and trace:
        final_step = trace[-1]
        if isinstance(final_step, dict):
            target_event = (
                final_step.get("relation"),
                final_step.get("subject"),
                final_step.get("value"),
            )
            for index, event in enumerate(parsed["events"]):
                event_key = (str(event["relation"]), str(event["subject"]), str(event["value"]))
                if event_key == target_event:
                    return index
    for index, event in enumerate(parsed["events"]):
        relation = str(event["relation"])
        subject = str(event["subject"])
        if relation == str(parsed["query_family"]) and subject == str(parsed["query_subject"]):
            return index
    raise ValueError(f"No matching event for query in row {row.get('id', '<unknown>')}")


def accuracy(rows: list[dict[str, Any]], predictor) -> float:
    return sum(1 for row in rows if predictor(row) == row["target"]) / len(rows)


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_lengths = [len(str(row["prompt"]).split()) for row in rows]
    query_family_counts = Counter(str(row["query_family"]) for row in rows)
    target_counts = Counter(str(row["target"]) for row in rows)
    subject_counts = Counter(str(row["query_subject"]) for row in rows)
    position_counts = Counter(matching_position(row) for row in rows)
    return {
        "rows": len(rows),
        "prompt_length_min": min(prompt_lengths),
        "prompt_length_max": max(prompt_lengths),
        "query_family_counts": dict(sorted(query_family_counts.items())),
        "query_subject_top5": subject_counts.most_common(5),
        "target_top5": target_counts.most_common(5),
        "matching_position_counts": dict(sorted(position_counts.items())),
    }


def overlap_count(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], key_fn) -> int:
    left_keys = {key_fn(row) for row in left_rows}
    right_keys = {key_fn(row) for row in right_rows}
    return len(left_keys & right_keys)


def pair_coverage(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    train_subject_targets = {(str(row["query_subject"]), str(row["target"])) for row in train_rows}
    eval_subject_targets = {(str(row["query_subject"]), str(row["target"])) for row in eval_rows}
    train_subjects = {str(row["query_subject"]) for row in train_rows}
    eval_subjects = {str(row["query_subject"]) for row in eval_rows}
    train_targets = {str(row["target"]) for row in train_rows}
    eval_targets = {str(row["target"]) for row in eval_rows}
    return {
        "query_subject_overlap_fraction": len(train_subjects & eval_subjects) / max(1, len(eval_subjects)),
        "target_overlap_fraction": len(train_targets & eval_targets) / max(1, len(eval_targets)),
        "subject_target_pair_overlap_fraction": len(train_subject_targets & eval_subject_targets) / max(1, len(eval_subject_targets)),
    }


def train_subject_prior_predictor(train_rows: list[dict[str, Any]]):
    counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        key = (str(row["query_family"]), str(row["query_subject"]))
        counts[key][str(row["target"])] += 1
    subject_to_target = {
        key: max(target_counts.items(), key=lambda item: (item[1], item[0]))[0]
        for key, target_counts in counts.items()
    }

    def predictor(row: dict[str, Any]) -> str:
        key = (str(row["query_family"]), str(row["query_subject"]))
        if key not in subject_to_target:
            raise ValueError(f"Subject prior key {key!r} not seen in train prior predictor")
        return subject_to_target[key]

    return predictor


def world_overlap(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    train_worlds = {canonical_world_key(row) for row in train_rows}
    eval_worlds = {canonical_world_key(row) for row in eval_rows}
    train_world_queries = {canonical_world_query_key(row) for row in train_rows}
    eval_world_queries = {canonical_world_query_key(row) for row in eval_rows}
    return {
        "world_overlap_fraction": len(train_worlds & eval_worlds) / max(1, len(eval_worlds)),
        "world_query_overlap_fraction": len(train_world_queries & eval_world_queries) / max(1, len(eval_world_queries)),
        "train_unique_worlds": len(train_worlds),
        "eval_unique_worlds": len(eval_worlds),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {dataset_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    split_paths = {
        "train": dataset_dir / "train.jsonl",
        "val": dataset_dir / "val.jsonl",
        "test": dataset_dir / "test.jsonl",
        "test_ood_longer_context": dataset_dir / "test_ood_longer_context.jsonl",
    }
    rows_by_split = {name: load_rows(path) for name, path in split_paths.items()}

    print("DATASET", metadata["name"])
    print("PRESET", metadata.get("preset"))
    print("SPLITS", metadata["splits"])
    print("PROMPT_LENGTH_SUMMARY", metadata["prompt_length_summary"])
    print("GENERATION_RULES", metadata["generation_rules"])
    print()

    print("SPLIT_SUMMARIES")
    for split_name, rows in rows_by_split.items():
        print(split_name, json.dumps(summarize_split(rows), indent=None))
    print()

    print("DUPLICATE_PROMPTS")
    for split_name, rows in rows_by_split.items():
        prompts = [str(row["prompt"]) for row in rows]
        duplicate_count = len(prompts) - len(set(prompts))
        print(split_name, duplicate_count)
    print()

    print("CROSS_SPLIT_PROMPT_OVERLAP")
    split_names = list(rows_by_split)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1:]:
            overlap = overlap_count(
                rows_by_split[left_name],
                rows_by_split[right_name],
                key_fn=lambda row: str(row["prompt"]),
            )
            print(left_name, right_name, overlap)
    print()

    print("TRAIN_TO_EVAL_COVERAGE")
    train_rows = rows_by_split["train"]
    for split_name in ("val", "test", "test_ood_longer_context"):
        print(split_name, json.dumps(pair_coverage(train_rows, rows_by_split[split_name]), indent=None))
    print()

    print("TRAIN_TO_EVAL_WORLD_OVERLAP")
    for split_name in ("val", "test", "test_ood_longer_context"):
        print(split_name, json.dumps(world_overlap(train_rows, rows_by_split[split_name]), indent=None))
    print()

    print("HEURISTIC_BASELINES")
    subject_prior = train_subject_prior_predictor(train_rows)
    for split_name, rows in rows_by_split.items():
        print(
            split_name,
            json.dumps(
                {
                    "oracle_match": accuracy(rows, find_matching_value),
                    "first_value": accuracy(rows, lambda row, metadata=metadata: first_value(row, metadata)),
                    "last_value": accuracy(rows, lambda row, metadata=metadata: last_value(row, metadata)),
                    "majority_value": accuracy(rows, lambda row, metadata=metadata: majority_value(row, metadata)),
                    "train_subject_prior": accuracy(rows, subject_prior),
                },
                indent=None,
            ),
        )


if __name__ == "__main__":
    main()
