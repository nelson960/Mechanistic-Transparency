#!/usr/bin/env python3
"""Train the tiny decoder on the superposition sparse world dataset with step-level tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from scripts.tiny_transformer_core import TinyDecoderTransformer
from scripts.training_dynamics import (
    _encode_batch,
    append_jsonl,
    build_epoch_batches,
    build_log_spaced_epochs,
    build_step_dynamics_record,
    ensure_run_directory,
    evaluate_next_token_rows,
    save_run_checkpoint,
    snapshot_tracked_parameter_groups,
)


def assert_run_dir_is_empty(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    disallowed_paths = [
        run_dir / "config.json",
        run_dir / "train_history.jsonl",
        run_dir / "checkpoints",
        run_dir / "battery",
        run_dir / "summaries",
    ]
    existing = [path for path in disallowed_paths if path.exists()]
    if existing:
        raise ValueError(
            "Refusing to overwrite an existing run directory with artifacts already present: "
            + ", ".join(str(path) for path in existing)
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"Blank line found in {path} at line {line_number}")
            rows.append(json.loads(text))
    return rows


def load_bundle(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocabulary = metadata.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise ValueError("Dataset metadata must contain a vocabulary object")
    vocab = []
    for key in ["special", "entities", "types", "colors", "positions", "states", "roles", "labels"]:
        values = vocabulary.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Dataset vocabulary is missing non-empty list for {key!r}")
        vocab.extend(str(value) for value in values)
    token_to_id = {token: index for index, token in enumerate(vocab)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    split_names = metadata.get("splits")
    if not isinstance(split_names, dict) or not split_names:
        raise ValueError("Dataset metadata must contain a non-empty splits object")
    raw_splits = {
        split_name: read_jsonl(dataset_dir / f"{split_name}.jsonl")
        for split_name in split_names
    }
    return {
        "metadata": metadata,
        "raw_splits": raw_splits,
        "vocab": vocab,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }


def build_schedule(*, epochs: int, dense_through_epoch: int, log_spaced_epoch_count: int, save_epoch_zero: bool) -> list[int]:
    if dense_through_epoch > epochs:
        raise ValueError("dense_through_epoch cannot exceed epochs")
    scheduled = list(range(0, dense_through_epoch + 1))
    if dense_through_epoch < epochs and log_spaced_epoch_count > 0:
        scheduled.extend(
            build_log_spaced_epochs(
                dense_through_epoch + 1,
                epochs,
                log_spaced_epoch_count,
            )
        )
    epochs_out = sorted(set(scheduled))
    if not save_epoch_zero and 0 in epochs_out:
        epochs_out.remove(0)
    return epochs_out


def train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    token_to_id: dict[str, int],
    device: torch.device,
    batch_size: int,
    epoch: int,
    history_path: Path,
    global_step_start: int,
    batch_seed: int,
) -> dict[str, Any]:
    batches = build_epoch_batches(rows, batch_size=batch_size, seed=batch_seed)
    if not batches:
        raise ValueError("Expected at least one batch")
    model.train()
    total_rows = 0
    total_loss = 0.0
    global_step = global_step_start
    for batch_index, batch_rows in enumerate(batches):
        input_ids, target_ids = _encode_batch(batch_rows, token_to_id, device)
        optimizer.zero_grad()
        logits = model(input_ids)
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, target_ids)
        loss.backward()
        pre_step_snapshot = snapshot_tracked_parameter_groups(model)
        optimizer.step()

        batch_loss = float(loss.item())
        batch_count = len(batch_rows)
        total_rows += batch_count
        total_loss += batch_loss * batch_count
        step_dynamics = build_step_dynamics_record(model, pre_step_snapshot)
        append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "batch_index": batch_index,
                "batch_rows": batch_count,
                "batch_loss": batch_loss,
                "total_grad_norm": step_dynamics["total_grad_norm"],
                "total_param_norm_pre": step_dynamics["total_param_norm_pre"],
                "total_param_norm_post": step_dynamics["total_param_norm_post"],
                "total_update_norm": step_dynamics["total_update_norm"],
                "relative_total_update_norm": step_dynamics["relative_total_update_norm"],
                "parameter_metrics": step_dynamics["parameter_metrics"],
            },
        )
        global_step += 1

    return {
        "epoch": epoch,
        "global_step_end": global_step,
        "train_loss": total_loss / total_rows,
        "train_rows": total_rows,
    }


def evaluate_splits(
    *,
    model: torch.nn.Module,
    bundle: dict[str, Any],
    device: torch.device,
    eval_batch_size: int,
) -> dict[str, float]:
    metrics = {}
    for split_name in ["train", "val", "test"]:
        result = evaluate_next_token_rows(
            model,
            bundle["raw_splits"][split_name],
            token_to_id=bundle["token_to_id"],
            id_to_token=bundle["id_to_token"],
            device=device,
            batch_size=eval_batch_size,
        )
        metrics[f"{split_name}_loss"] = result["loss"]
        metrics[f"{split_name}_accuracy"] = result["accuracy"]
        metrics[f"{split_name}_margin"] = result["margin"]
    ood_split_names = [name for name in bundle["raw_splits"] if name.startswith("test_ood_")]
    if len(ood_split_names) != 1:
        raise ValueError(f"Expected exactly one OOD split, found {ood_split_names}")
    ood_result = evaluate_next_token_rows(
        model,
        bundle["raw_splits"][ood_split_names[0]],
        token_to_id=bundle["token_to_id"],
        id_to_token=bundle["id_to_token"],
        device=device,
        batch_size=eval_batch_size,
    )
    metrics["ood_loss"] = ood_result["loss"]
    metrics["ood_accuracy"] = ood_result["accuracy"]
    metrics["ood_margin"] = ood_result["margin"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny decoder on superposition sparse world.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/phase3/superposition_sparse_world_v1"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=40)
    parser.add_argument("--dense-through-epoch", type=int, default=20)
    parser.add_argument("--log-spaced-epoch-count", type=int, default=24)
    parser.add_argument("--save-epoch-zero", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_dir = args.run_dir.expanduser().resolve()
    assert_run_dir_is_empty(run_dir)
    run_paths = ensure_run_directory(run_dir)
    bundle = load_bundle(args.dataset_dir.expanduser().resolve())

    model = TinyDecoderTransformer(
        vocab_size=len(bundle["vocab"]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config_payload = {
        "benchmark_name": "superposition_sparse_world",
        "dataset_dir": str(args.dataset_dir),
        "model": {
            "vocab_size": len(bundle["vocab"]),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "max_seq_len": args.max_seq_len,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
        },
        "checkpoint_schedule": {
            "dense_through_epoch": args.dense_through_epoch,
            "log_spaced_epoch_count": args.log_spaced_epoch_count,
            "save_epoch_zero": args.save_epoch_zero,
            "save_final": True,
            "best_metric": "val_accuracy",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    scheduled_epochs = set(
        build_schedule(
            epochs=args.epochs,
            dense_through_epoch=args.dense_through_epoch,
            log_spaced_epoch_count=args.log_spaced_epoch_count,
            save_epoch_zero=args.save_epoch_zero,
        )
    )
    best_val: float | None = None
    global_step = 0

    if args.save_epoch_zero:
        epoch_zero_metrics = evaluate_splits(
            model=model,
            bundle=bundle,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )
        save_run_checkpoint(
            run_paths["checkpoints_dir"] / "scheduled_epoch_000.pt",
            model=model,
            config=config_payload["model"],
            token_to_id=bundle["token_to_id"],
            id_to_token=bundle["id_to_token"],
            dataset_metadata=bundle["metadata"],
            seed=args.seed,
            epoch=0,
            global_step=0,
            save_reason="scheduled",
            selected_metrics=epoch_zero_metrics,
            benchmark_name="superposition_sparse_world",
            run_id=run_dir.name,
            train_config=config_payload["training"],
            device_used_for_training=args.device,
        )
        best_val = float(epoch_zero_metrics["val_accuracy"])

    for epoch in range(1, args.epochs + 1):
        train_result = train_epoch(
            model=model,
            optimizer=optimizer,
            rows=bundle["raw_splits"]["train"],
            token_to_id=bundle["token_to_id"],
            device=device,
            batch_size=args.batch_size,
            epoch=epoch,
            history_path=run_paths["history_path"],
            global_step_start=global_step,
            batch_seed=args.seed + epoch,
        )
        global_step = int(train_result["global_step_end"])
        checkpoint_metrics = evaluate_splits(
            model=model,
            bundle=bundle,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )
        checkpoint_metrics["train_batch_loss"] = float(train_result["train_loss"])
        checkpoint_metrics["epoch"] = epoch
        if epoch in scheduled_epochs:
            save_run_checkpoint(
                run_paths["checkpoints_dir"] / f"scheduled_epoch_{epoch:03d}.pt",
                model=model,
                config=config_payload["model"],
                token_to_id=bundle["token_to_id"],
                id_to_token=bundle["id_to_token"],
                dataset_metadata=bundle["metadata"],
                seed=args.seed,
                epoch=epoch,
                global_step=global_step,
                save_reason="scheduled",
                selected_metrics=checkpoint_metrics,
                benchmark_name="superposition_sparse_world",
                run_id=run_dir.name,
                train_config=config_payload["training"],
                device_used_for_training=args.device,
            )
        val_accuracy = float(checkpoint_metrics["val_accuracy"])
        if best_val is None or val_accuracy > best_val:
            best_val = val_accuracy
            save_run_checkpoint(
                run_paths["checkpoints_dir"] / f"best_val_epoch_{epoch:03d}.pt",
                model=model,
                config=config_payload["model"],
                token_to_id=bundle["token_to_id"],
                id_to_token=bundle["id_to_token"],
                dataset_metadata=bundle["metadata"],
                seed=args.seed,
                epoch=epoch,
                global_step=global_step,
                save_reason="best_val",
                selected_metrics=checkpoint_metrics,
                benchmark_name="superposition_sparse_world",
                run_id=run_dir.name,
                train_config=config_payload["training"],
                device_used_for_training=args.device,
            )

    final_metrics = evaluate_splits(
        model=model,
        bundle=bundle,
        device=device,
        eval_batch_size=args.eval_batch_size,
    )
    final_metrics["epoch"] = args.epochs
    save_run_checkpoint(
        run_paths["checkpoints_dir"] / f"final_epoch_{args.epochs:03d}.pt",
        model=model,
        config=config_payload["model"],
        token_to_id=bundle["token_to_id"],
        id_to_token=bundle["id_to_token"],
        dataset_metadata=bundle["metadata"],
        seed=args.seed,
        epoch=args.epochs,
        global_step=global_step,
        save_reason="final",
        selected_metrics=final_metrics,
        benchmark_name="superposition_sparse_world",
        run_id=run_dir.name,
        train_config=config_payload["training"],
        device_used_for_training=args.device,
    )


if __name__ == "__main__":
    main()
