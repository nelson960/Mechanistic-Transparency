#!/usr/bin/env python3
"""Generate a synthetic key-value retrieval dataset for a toy decoder model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def build_vocab(num_keys: int, num_values: int) -> Dict[str, List[str]]:
    if num_keys <= 0:
        raise ValueError(f"num_keys must be positive, got {num_keys}")
    if num_values <= 0:
        raise ValueError(f"num_values must be positive, got {num_values}")
    return {
        "special": ["<bos>", ";", "Q", "->"],
        "keys": [f"K{i}" for i in range(num_keys)],
        "values": [f"V{i}" for i in range(num_values)],
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


def generate_example(
    example_id: str,
    rng: random.Random,
    keys: Sequence[str],
    values: Sequence[str],
    context_pairs: int,
    split: str,
) -> Dict[str, object]:
    pairs = sample_pairs(rng, keys, values, context_pairs)
    mapping = {key: value for key, value in pairs}
    query_key = rng.choice(list(mapping))
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
) -> List[Dict[str, object]]:
    if size < 0:
        raise ValueError(f"size must be non-negative for split={split}, got {size}")
    rows = []
    for idx in range(size):
        rows.append(
            generate_example(
                example_id=f"{split}_{idx:06d}",
                rng=rng,
                keys=keys,
                values=values,
                context_pairs=context_pairs,
                split=split,
            )
        )
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
        default="dataset/kv_retrieve_3",
        help="Output directory for dataset files.",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--ood-size", type=int, default=500)
    parser.add_argument("--num-keys", type=int, default=8)
    parser.add_argument("--num-values", type=int, default=8)
    parser.add_argument(
        "--context-pairs",
        type=int,
        default=3,
        help="Number of key-value pairs in the in-distribution splits.",
    )
    parser.add_argument(
        "--ood-context-pairs",
        type=int,
        default=4,
        help="Number of key-value pairs in the OOD split.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(num_keys=args.num_keys, num_values=args.num_values)
    keys = vocab["keys"]
    values = vocab["values"]
    rng = random.Random(args.seed)

    train_rows = generate_split(
        rng=rng,
        split="train",
        size=args.train_size,
        keys=keys,
        values=values,
        context_pairs=args.context_pairs,
    )
    val_rows = generate_split(
        rng=rng,
        split="val",
        size=args.val_size,
        keys=keys,
        values=values,
        context_pairs=args.context_pairs,
    )
    test_rows = generate_split(
        rng=rng,
        split="test",
        size=args.test_size,
        keys=keys,
        values=values,
        context_pairs=args.context_pairs,
    )
    ood_rows = generate_split(
        rng=rng,
        split=f"test_ood_{args.ood_context_pairs}_pairs",
        size=args.ood_size,
        keys=keys,
        values=values,
        context_pairs=args.ood_context_pairs,
    )

    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "val.jsonl", val_rows)
    write_jsonl(outdir / "test.jsonl", test_rows)
    write_jsonl(outdir / f"test_ood_{args.ood_context_pairs}_pairs.jsonl", ood_rows)

    metadata = {
        "name": "KV-Retrieve-3",
        "seed": args.seed,
        "vocabulary": vocab,
        "splits": {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
            f"test_ood_{args.ood_context_pairs}_pairs": args.ood_size,
        },
        "sequence_format": "<bos> K_a V_b ; K_c V_d ; K_e V_f ; Q K_x ->",
        "target": "Single next-token prediction of the queried value token.",
        "generation_rules": {
            "keys_per_example": args.context_pairs,
            "values_per_example": args.context_pairs,
            "keys_are_distinct": True,
            "values_are_distinct": True,
            "query_key_sampled_from_context": True,
            "pair_order_shuffled": True,
        },
        "ood_rule": {
            "num_pairs": args.ood_context_pairs,
            "description": "Longer-context OOD split with more context pairs.",
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
