from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kv_retrieve_analysis import load_checkpoint_model, load_dataset_bundle
from scripts.kv_retrieve_features import (
    FEATURE_SITES,
    build_feature_delta_table,
    build_feature_projection_table,
    build_top_feature_examples,
    collect_query_swap_pairs,
    collect_split_activations,
    save_feature_analysis,
    save_sae_checkpoint,
    train_sae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small SAE on a mechanistic activation site.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/kv_retrieve_3/selected_checkpoint.pt"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/kv_retrieve_3"),
    )
    parser.add_argument(
        "--site",
        choices=sorted(FEATURE_SITES.keys()),
        default="block1_final_resid",
    )
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--pair-limit", type=int, default=256)
    parser.add_argument("--hidden-multiplier", type=int, default=8)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebook/outputs/kv_retrieve_feature_basis.json"),
    )
    parser.add_argument(
        "--sae-checkpoint-output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_dataset_bundle(args.dataset_dir)
    device = torch.device("cpu")
    _, model = load_checkpoint_model(args.checkpoint, device=device)
    torch.manual_seed(args.seed)

    train_activations, _ = collect_split_activations(
        model, bundle, split="train", site=args.site, limit=args.train_limit
    )
    val_activations, _ = collect_split_activations(
        model, bundle, split="val", site=args.site, limit=args.val_limit
    )
    test_activations, test_records = collect_split_activations(
        model, bundle, split="test", site=args.site, limit=args.pair_limit
    )

    hidden_dim = train_activations.shape[1] * args.hidden_multiplier
    sae, history_df = train_sae(
        train_activations=train_activations,
        val_activations=val_activations,
        hidden_dim=hidden_dim,
        l1_coeff=args.l1_coeff,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    pair_bundle = collect_query_swap_pairs(
        model, bundle, split="test", site=args.site, limit=args.pair_limit
    )
    feature_delta_df = build_feature_delta_table(
        sae,
        clean_activations=pair_bundle["clean_activations"],
        corrupt_activations=pair_bundle["corrupt_activations"],
    )

    sample_prompt = bundle.raw_splits["test"][0]["prompt"]
    final_position_index = len(sample_prompt.split()) - 1
    projection_df = build_feature_projection_table(
        model=model,
        bundle=bundle,
        sae=sae,
        mean_q_delta=pair_bundle["mean_q_delta"],
        final_position_index=final_position_index,
        site=args.site,
    )

    feature_summary_df = feature_delta_df.merge(projection_df, on="feature_index")
    feature_summary_df["mechanism_score"] = (
        feature_summary_df["mean_delta"] * feature_summary_df["query_alignment"]
    )
    feature_summary_df = feature_summary_df.sort_values(
        ["mechanism_score", "abs_mean_delta"], ascending=[False, False]
    ).reset_index(drop=True)

    top_feature_indices = feature_summary_df.head(5)["feature_index"].tolist()
    top_examples = build_top_feature_examples(
        sae,
        activations=test_activations,
        records=test_records,
        feature_indices=top_feature_indices,
        top_k=5,
    )

    payload = {
        "site": args.site,
        "site_description": FEATURE_SITES[args.site],
        "config": {
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
            "pair_limit": args.pair_limit,
            "hidden_multiplier": args.hidden_multiplier,
            "hidden_dim": hidden_dim,
            "l1_coeff": args.l1_coeff,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "history_tail": history_df.tail(10).to_dict(orient="records"),
        "feature_summary_top10": feature_summary_df.head(10).to_dict(orient="records"),
        "top_examples": top_examples,
        "mean_q_delta_norm": float(pair_bundle["mean_q_delta"].norm().item()),
    }
    save_feature_analysis(args.output, payload)
    if args.sae_checkpoint_output is not None:
        save_sae_checkpoint(
            args.sae_checkpoint_output,
            sae,
            metadata={
                "site": args.site,
                "feature_summary_top10": payload["feature_summary_top10"],
                "config": payload["config"],
                "mean_q_delta_norm": payload["mean_q_delta_norm"],
            },
        )

    print(f"Saved feature analysis to {args.output}")
    if args.sae_checkpoint_output is not None:
        print(f"Saved SAE checkpoint to {args.sae_checkpoint_output}")
    print()
    print("Top candidate features by mechanism_score:")
    display_df = feature_summary_df[
        [
            "feature_index",
            "mean_clean_activation",
            "mean_corrupt_activation",
            "mean_delta",
            "query_alignment",
            "query_cosine",
            "mechanism_score",
            "top_logit_tokens",
        ]
    ].head(10)
    print(display_df.to_string(index=False))
    print()
    print("SAE training tail:")
    print(history_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
