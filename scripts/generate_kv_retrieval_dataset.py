#!/usr/bin/env python3
"""Generate a synthetic key-value retrieval dataset for a toy decoder model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


RESERVED_PROMPT_TOKENS = {"<bos>", ";", "Q", "->"}


def read_token_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing token file: {path}")
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"Token file contains zero usable lines: {path}")
    return tokens


def validate_token_list(tokens: Sequence[str], *, role: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for index, token in enumerate(tokens):
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"{role} token at index {index} must be a non-empty string")
        if token != token.strip():
            raise ValueError(f"{role} token at index {index} has surrounding whitespace: {token!r}")
        if any(character.isspace() for character in token):
            raise ValueError(f"{role} token at index {index} must be a single token without whitespace: {token!r}")
        if token in RESERVED_PROMPT_TOKENS:
            raise ValueError(f"{role} token {token!r} conflicts with reserved prompt syntax tokens")
        if token in seen:
            raise ValueError(f"{role} token list contains duplicates: {token!r}")
        normalized.append(token)
        seen.add(token)
    return normalized


def build_vocab(
    num_keys: int,
    num_values: int,
    *,
    key_tokens: Sequence[str] | None = None,
    value_tokens: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    if num_keys <= 0:
        raise ValueError(f"num_keys must be positive, got {num_keys}")
    if num_values <= 0:
        raise ValueError(f"num_values must be positive, got {num_values}")
    if key_tokens is None:
        keys = [f"K{i}" for i in range(num_keys)]
    else:
        keys = validate_token_list(key_tokens, role="key")
        if len(keys) != num_keys:
            raise ValueError(
                f"Expected exactly {num_keys} key tokens, found {len(keys)} in the provided key vocabulary"
            )
    if value_tokens is None:
        values = [f"V{i}" for i in range(num_values)]
    else:
        values = validate_token_list(value_tokens, role="value")
        if len(values) != num_values:
            raise ValueError(
                f"Expected exactly {num_values} value tokens, found {len(values)} in the provided value vocabulary"
            )
    overlap = sorted(set(keys) & set(values))
    if overlap:
        raise ValueError(f"Key and value vocabularies must be disjoint, found overlap: {overlap}")
    return {
        "special": ["<bos>", ";", "Q", "->"],
        "keys": keys,
        "values": values,
    }


def sample_pairs(
    rng: random.Random,
    keys: Sequence[str],
    values: Sequence[str],
    context_pairs: int,
) -> List[Tuple[str, str]]:
    if context_pairs <= 0:
        raise ValueError(f"context_pairs must be positive, got {context_pairs}")
    if context_pairs > len(keys):
        raise ValueError(
            f"context_pairs={context_pairs} exceeds number of keys={len(keys)}"
        )
    if context_pairs > len(values):
        raise ValueError(
            f"context_pairs={context_pairs} exceeds number of values={len(values)}"
        )

    chosen_keys = rng.sample(list(keys), context_pairs)
    chosen_values = rng.sample(list(values), context_pairs)
    pairs = list(zip(chosen_keys, chosen_values))
    rng.shuffle(pairs)
    return pairs


def render_prompt(pairs: Sequence[Tuple[str, str]], query_key: str) -> str:
    pair_tokens = [f"{key} {value}" for key, value in pairs]
    return "<bos> " + " ; ".join(pair_tokens) + f" ; Q {query_key} ->"


def select_query_key(
    rng: random.Random,
    pairs: Sequence[Tuple[str, str]],
    *,
    example_index: int,
    query_slot_policy: str,
) -> str:
    if query_slot_policy == "balanced":
        target_slot = example_index % len(pairs)
        return str(pairs[target_slot][0])
    if query_slot_policy == "random":
        return rng.choice([key for key, _ in pairs])
    raise ValueError(f"Unsupported query_slot_policy {query_slot_policy!r}")


def generate_example(
    example_id: str,
    rng: random.Random,
    keys: Sequence[str],
    values: Sequence[str],
    context_pairs: int,
    split: str,
    example_index: int,
    query_slot_policy: str,
) -> Dict[str, object]:
    pairs = sample_pairs(rng, keys, values, context_pairs)
    mapping = {key: value for key, value in pairs}
    query_key = select_query_key(
        rng,
        pairs,
        example_index=example_index,
        query_slot_policy=query_slot_policy,
    )
    target_value = mapping[query_key]
    prompt = render_prompt(pairs, query_key)

    return {
        "id": example_id,
        "task": "kv_retrieve_3" if context_pairs == 3 else f"kv_retrieve_{context_pairs}",
        "split": split,
        "num_pairs": context_pairs,
        "prompt": prompt,
        "target": target_value,
        "query_key": query_key,
        "context_pairs": [
            {"key": key, "value": value, "pair_index": idx}
            for idx, (key, value) in enumerate(pairs)
        ],
    }


def generate_split(
    rng: random.Random,
    split: str,
    size: int,
    keys: Sequence[str],
    values: Sequence[str],
    context_pairs: int,
    query_slot_policy: str,
    allow_duplicate_prompts: bool,
    global_seen_prompts: set[str],
    max_attempt_multiplier: int,
) -> List[Dict[str, object]]:
    if size < 0:
        raise ValueError(f"size must be non-negative for split={split}, got {size}")
    if max_attempt_multiplier <= 0:
        raise ValueError(f"max_attempt_multiplier must be positive, got {max_attempt_multiplier}")
    rows = []
    local_seen_prompts: set[str] = set()
    max_attempts = max(1, size) * max_attempt_multiplier
    attempts = 0
    while len(rows) < size:
        if attempts >= max_attempts:
            raise ValueError(
                f"Could not generate {size} prompts for split={split!r} after {attempts} attempts. "
                "Reduce the requested split size, increase the vocabulary, or allow duplicate prompts."
            )
        attempts += 1
        example = generate_example(
            example_id=f"{split}_{len(rows):06d}",
            rng=rng,
            keys=keys,
            values=values,
            context_pairs=context_pairs,
            split=split,
            example_index=len(rows),
            query_slot_policy=query_slot_policy,
        )
        prompt = str(example["prompt"])
        if not allow_duplicate_prompts:
            if prompt in local_seen_prompts or prompt in global_seen_prompts:
                continue
            local_seen_prompts.add(prompt)
            global_seen_prompts.add(prompt)
        rows.append(example)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic key-value retrieval dataset."
    )
    parser.add_argument(
        "--outdir",
        default="dataset/phase2/kv_retrieve_3",
        help="Output directory for dataset files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="KV-Retrieve-Balanced",
        help="Dataset metadata name to write into metadata.json.",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--ood-size", type=int, default=500)
    parser.add_argument("--num-keys", type=int, default=8)
    parser.add_argument("--num-values", type=int, default=8)
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help="Optional newline-delimited key vocabulary file. Must contain exactly --num-keys entries.",
    )
    parser.add_argument(
        "--values-file",
        type=Path,
        default=None,
        help="Optional newline-delimited value vocabulary file. Must contain exactly --num-values entries.",
    )
    parser.add_argument(
        "--context-pairs",
        type=int,
        default=3,
        help="Number of key-value pairs in the validation/test in-distribution splits.",
    )
    parser.add_argument(
        "--train-context-pairs",
        type=str,
        default=None,
        help="Comma-separated training pair counts, for example '2,3'. Defaults to the in-distribution context-pair count.",
    )
    parser.add_argument(
        "--ood-context-pairs",
        type=int,
        default=4,
        help="Number of key-value pairs in the OOD split.",
    )
    parser.add_argument(
        "--query-slot-policy",
        choices=["balanced", "random"],
        default="balanced",
        help="How to choose the queried slot within each prompt.",
    )
    parser.add_argument(
        "--allow-duplicate-prompts",
        action="store_true",
        help="Allow duplicate prompt strings across and within splits.",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=200,
        help="Maximum sampling attempts per requested row when duplicate prompts are disallowed.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    key_tokens = None if args.keys_file is None else read_token_file(args.keys_file.expanduser().resolve())
    value_tokens = None if args.values_file is None else read_token_file(args.values_file.expanduser().resolve())
    vocab = build_vocab(
        num_keys=args.num_keys,
        num_values=args.num_values,
        key_tokens=key_tokens,
        value_tokens=value_tokens,
    )
    keys = vocab["keys"]
    values = vocab["values"]
    rng = random.Random(args.seed)
    global_seen_prompts: set[str] = set()

    if args.train_context_pairs is None:
        train_context_pairs = [args.context_pairs]
    else:
        train_context_pairs = []
        for raw_value in args.train_context_pairs.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            train_context_pairs.append(int(raw_value))
    if not train_context_pairs:
        raise ValueError("Expected at least one training context-pair count")

    train_split_rows: dict[str, list[dict[str, object]]] = {}
    for context_pairs in train_context_pairs:
        split_name = "train" if len(train_context_pairs) == 1 and context_pairs == args.context_pairs else f"train_{context_pairs}_pairs"
        train_split_rows[split_name] = generate_split(
            rng=rng,
            split=split_name,
            size=args.train_size,
            keys=keys,
            values=values,
            context_pairs=context_pairs,
            query_slot_policy=args.query_slot_policy,
            allow_duplicate_prompts=args.allow_duplicate_prompts,
            global_seen_prompts=global_seen_prompts,
            max_attempt_multiplier=args.max_attempt_multiplier,
        )
    val_rows = generate_split(
        rng=rng,
        split="val",
        size=args.val_size,
        keys=keys,
        values=values,
        context_pairs=args.context_pairs,
        query_slot_policy=args.query_slot_policy,
        allow_duplicate_prompts=args.allow_duplicate_prompts,
        global_seen_prompts=global_seen_prompts,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )
    test_rows = generate_split(
        rng=rng,
        split="test",
        size=args.test_size,
        keys=keys,
        values=values,
        context_pairs=args.context_pairs,
        query_slot_policy=args.query_slot_policy,
        allow_duplicate_prompts=args.allow_duplicate_prompts,
        global_seen_prompts=global_seen_prompts,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )
    ood_rows = generate_split(
        rng=rng,
        split=f"test_ood_{args.ood_context_pairs}_pairs",
        size=args.ood_size,
        keys=keys,
        values=values,
        context_pairs=args.ood_context_pairs,
        query_slot_policy=args.query_slot_policy,
        allow_duplicate_prompts=args.allow_duplicate_prompts,
        global_seen_prompts=global_seen_prompts,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )

    for split_name, rows in train_split_rows.items():
        write_jsonl(outdir / f"{split_name}.jsonl", rows)
    write_jsonl(outdir / "val.jsonl", val_rows)
    write_jsonl(outdir / "test.jsonl", test_rows)
    write_jsonl(outdir / f"test_ood_{args.ood_context_pairs}_pairs.jsonl", ood_rows)

    split_metadata = {
        split_name: args.train_size
        for split_name in train_split_rows
    }
    split_metadata.update(
        {
            "val": args.val_size,
            "test": args.test_size,
            f"test_ood_{args.ood_context_pairs}_pairs": args.ood_size,
        }
    )
    metadata = {
        "name": args.dataset_name,
        "seed": args.seed,
        "vocabulary": vocab,
        "splits": split_metadata,
        "training_splits": {
            str(context_pairs): split_name
            for context_pairs, split_name in zip(train_context_pairs, train_split_rows.keys(), strict=True)
        },
        "sequence_format": "<bos> key_a value_b ; key_c value_d ; key_e value_f ; Q key_x ->",
        "target": "Single next-token prediction of the queried value token.",
        "generation_rules": {
            "train_context_pairs": train_context_pairs,
            "id_context_pairs": args.context_pairs,
            "values_per_example": args.context_pairs,
            "keys_are_distinct": True,
            "values_are_distinct": True,
            "query_key_sampled_from_context": True,
            "pair_order_shuffled": True,
            "query_slot_policy": args.query_slot_policy,
            "allow_duplicate_prompts": args.allow_duplicate_prompts,
            "max_attempt_multiplier": args.max_attempt_multiplier,
        },
        "ood_rule": {
            "num_pairs": args.ood_context_pairs,
            "description": "Longer-context OOD split with more context pairs.",
        },
        "vocabulary_sources": {
            "keys_file": None if args.keys_file is None else str(args.keys_file.expanduser().resolve()),
            "values_file": None if args.values_file is None else str(args.values_file.expanduser().resolve()),
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
