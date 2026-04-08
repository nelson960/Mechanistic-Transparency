#!/usr/bin/env python3
"""Analyze failure modes for microlanguage checkpoint predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate_microlanguage_world_dataset import parse_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze microlanguage run checkpoint failure patterns.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing manifest.json and battery/")
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Checkpoint battery directory to inspect. Defaults to the latest best_val or early_stop checkpoint from checkpoint_index.csv.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def active_target_values(row: dict[str, Any], target_tokens: set[str]) -> list[str]:
    parsed = parse_prompt(str(row["prompt"]))
    values = [str(event["value"]) for event in parsed["events"] if str(event["value"]) in target_tokens]
    if not values:
        raise ValueError(f"No target-group values found in prompt for row {row.get('id', '<unknown>')}")
    return values


def matching_position(row: dict[str, Any]) -> int:
    parsed = parse_prompt(str(row["prompt"]))
    events = [
        (str(event["relation"]), str(event["subject"]), str(event["value"]))
        for event in parsed["events"]
    ]
    query_family = str(parsed["query_family"])
    query_subject = str(parsed["query_subject"])
    trace = row.get("trace")
    if isinstance(trace, list) and trace:
        last_step = trace[-1]
        if isinstance(last_step, dict):
            target_event = (
                last_step.get("relation"),
                last_step.get("subject"),
                last_step.get("value"),
            )
            for index, event in enumerate(events):
                if event == target_event:
                    return index
    for index, (relation, subject, _) in enumerate(events):
        if relation == query_family and subject == query_subject:
            return index
    raise ValueError(f"No matching event for query in row {row.get('id', '<unknown>')}")


def choose_checkpoint_id(run_dir: Path) -> str:
    checkpoint_index_path = run_dir / "summaries" / "checkpoint_index.csv"
    rows = load_csv(checkpoint_index_path)
    preferred_prefixes = ("best_val_epoch_", "early_stop_epoch_", "scheduled_epoch_")
    for prefix in preferred_prefixes:
        matches = [row["checkpoint_id"] for row in rows if row["checkpoint_id"].startswith(prefix)]
        if matches:
            return matches[-1]
    raise ValueError(f"Could not infer a checkpoint id from {checkpoint_index_path}")


def summarize_counter(counter: Counter[str | int], *, top_k: int = 10) -> list[tuple[str | int, int]]:
    return counter.most_common(top_k)


def safe_accuracy(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Denominator must be positive")
    return numerator / denominator


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = load_json(run_dir / "manifest.json")
    dataset_dir = Path(str(manifest["dataset"]["dataset_dir"])).expanduser().resolve()
    dataset_metadata = load_json(dataset_dir / "metadata.json")
    checkpoint_id = args.checkpoint_id or choose_checkpoint_id(run_dir)

    battery_dir = run_dir / "battery" / checkpoint_id
    scored_rows = load_csv(battery_dir / "scored_rows.csv")
    behavior = load_json(battery_dir / "behavior.json")

    split_file_names = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "test_ood_longer_context": "test_ood_longer_context.jsonl",
    }
    rows_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, file_name in split_file_names.items():
        split_path = dataset_dir / file_name
        if split_path.exists():
            rows_by_split[split_name] = {str(row["id"]): row for row in load_jsonl(split_path)}

    eval_to_dataset_split = {
        "val": "val",
        "test": "test",
        "test_ood_longer_context": "test_ood_longer_context",
        "ood": "test_ood_longer_context",
        "train_selected": "train",
    }

    vocab_groups = dataset_metadata.get("vocabulary_groups")
    if not isinstance(vocab_groups, dict):
        raise ValueError("Dataset metadata must define vocabulary_groups")
    query_specs = dataset_metadata.get("latent_state", {}).get("query_families")
    if not isinstance(query_specs, dict):
        raise ValueError("Dataset metadata must define latent_state.query_families")

    analyses: dict[str, dict[str, Any]] = {}
    for split_name, metric in behavior["split_metrics"].items():
        dataset_split = eval_to_dataset_split.get(split_name)
        if dataset_split is None or dataset_split not in rows_by_split:
            continue
        split_predictions = [row for row in scored_rows if row["split_name"] == split_name]
        if not split_predictions:
            continue
        dataset_rows = rows_by_split[dataset_split]
        if len(split_predictions) != metric["rows"]:
            raise ValueError(
                f"Prediction row count mismatch for split {split_name}: "
                f"behavior has {metric['rows']}, scored_rows has {len(split_predictions)}"
            )

        total = len(split_predictions)
        correct = sum(row["correct"] == "True" for row in split_predictions)
        if abs(safe_accuracy(correct, total) - float(metric["accuracy"])) > 1e-9:
            raise ValueError(f"Accuracy mismatch for split {split_name}")

        by_position: defaultdict[int, list[bool]] = defaultdict(list)
        by_room_pair: defaultdict[str, list[bool]] = defaultdict(list)
        by_target: defaultdict[str, list[bool]] = defaultdict(list)
        by_predicted: Counter[str] = Counter()
        confusion: Counter[tuple[str, str]] = Counter()
        heuristic_agreement: Counter[str] = Counter()
        heuristic_correct: Counter[str] = Counter()
        wrong_examples: list[dict[str, Any]] = []

        for prediction in split_predictions:
            prompt_id = prediction["prompt_id"]
            if prompt_id not in dataset_rows:
                raise ValueError(f"Prompt id {prompt_id!r} missing from dataset split {dataset_split}")
            source_row = dataset_rows[prompt_id]
            query_family = str(source_row["query_family"])
            spec = query_specs.get(query_family)
            if not isinstance(spec, dict):
                raise ValueError(f"Unknown query family {query_family!r}")
            target_group_name = spec.get("target_group")
            if not isinstance(target_group_name, str):
                raise ValueError(f"Query family {query_family!r} missing target_group")
            group_tokens = vocab_groups.get(target_group_name)
            if not isinstance(group_tokens, list):
                raise ValueError(f"Target group {target_group_name!r} missing from vocabulary_groups")
            active_values = active_target_values(source_row, set(group_tokens))
            pair_key = "|".join(sorted(set(active_values)))
            position = matching_position(source_row)
            predicted_token = prediction["predicted_token"]
            target_token = prediction["target_token"]
            correct_flag = prediction["correct"] == "True"

            first_value = active_values[0]
            last_value = active_values[-1]
            majority_value = max(Counter(active_values).items(), key=lambda item: (item[1], item[0]))[0]

            by_position[position].append(correct_flag)
            by_room_pair[pair_key].append(correct_flag)
            by_target[target_token].append(correct_flag)
            by_predicted[predicted_token] += 1
            confusion[(target_token, predicted_token)] += 1
            if predicted_token == first_value:
                heuristic_agreement["predicted_first_value"] += 1
            if predicted_token == last_value:
                heuristic_agreement["predicted_last_value"] += 1
            if predicted_token == majority_value:
                heuristic_agreement["predicted_majority_value"] += 1
            if target_token == first_value:
                heuristic_correct["first_value"] += 1
            if target_token == last_value:
                heuristic_correct["last_value"] += 1
            if target_token == majority_value:
                heuristic_correct["majority_value"] += 1
            if not correct_flag and len(wrong_examples) < 12:
                wrong_examples.append(
                    {
                        "prompt_id": prompt_id,
                        "query_subject": source_row["query_subject"],
                        "target": target_token,
                        "predicted": predicted_token,
                        "matching_position": position,
                        "active_values": active_values,
                        "prompt": source_row["prompt"],
                    }
                )

        analyses[split_name] = {
            "rows": total,
            "accuracy": safe_accuracy(correct, total),
            "accuracy_by_matching_position": {
                str(position): safe_accuracy(sum(values), len(values))
                for position, values in sorted(by_position.items())
            },
            "accuracy_by_target_token_top": [
                {"target_token": token, "rows": len(values), "accuracy": safe_accuracy(sum(values), len(values))}
                for token, values in sorted(by_target.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
            ],
            "accuracy_by_active_value_pair_top": [
                {"active_values": pair, "rows": len(values), "accuracy": safe_accuracy(sum(values), len(values))}
                for pair, values in sorted(by_room_pair.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
            ],
            "predicted_token_top": summarize_counter(by_predicted),
            "most_common_confusions": [
                {"target": target, "predicted": predicted, "rows": count}
                for (target, predicted), count in confusion.most_common(12)
                if target != predicted
            ],
            "prediction_alignment": {
                "predicted_first_value_fraction": safe_accuracy(heuristic_agreement["predicted_first_value"], total),
                "predicted_last_value_fraction": safe_accuracy(heuristic_agreement["predicted_last_value"], total),
                "predicted_majority_value_fraction": safe_accuracy(heuristic_agreement["predicted_majority_value"], total),
                "target_first_value_fraction": safe_accuracy(heuristic_correct["first_value"], total),
                "target_last_value_fraction": safe_accuracy(heuristic_correct["last_value"], total),
                "target_majority_value_fraction": safe_accuracy(heuristic_correct["majority_value"], total),
            },
            "sample_wrong_examples": wrong_examples,
        }

    report = {
        "run_dir": str(run_dir),
        "checkpoint_id": checkpoint_id,
        "dataset_dir": str(dataset_dir),
        "behavior_split_metrics": behavior["split_metrics"],
        "failure_analysis": analyses,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
