#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.story_text_circuit_benchmark import (
    build_story_bundle,
    build_story_eval_pack,
    run_story_checkpoint_battery,
    save_story_bundle,
    summarize_story_run,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny decoder on a story text file and log circuit-origin artifacts with live tqdm progress."
    )
    parser.add_argument("--text-path", type=Path, default=Path("dataset/phase2/random_story_dataset_v1.txt"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=24, help="Number of text tokens before the target token.")
    parser.add_argument("--ood-context-length", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
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
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--dense-through-epoch", type=int, default=20)
    parser.add_argument("--log-spaced-epoch-count", type=int, default=24)
    parser.add_argument("--save-epoch-zero", action="store_true")
    parser.add_argument("--sweep-base-limit", type=int, default=32)
    parser.add_argument("--train-probe-limit", type=int, default=128)
    parser.add_argument("--sae-train-limit", type=int, default=128)
    parser.add_argument("--sae-val-limit", type=int, default=64)
    parser.add_argument("--sae-hidden-multiplier", type=int, default=2)
    parser.add_argument("--sae-l1-coeff", type=float, default=0.001)
    parser.add_argument("--sae-learning-rate", type=float, default=0.01)
    parser.add_argument("--sae-batch-size", type=int, default=64)
    parser.add_argument("--sae-epochs", type=int, default=5)
    parser.add_argument("--top-features-per-site", type=int, default=5)
    parser.add_argument("--superposition-cosine-threshold", type=float, default=0.2)
    parser.add_argument("--role-top-k", type=int, default=3)
    parser.add_argument("--behavior-birth-val-accuracy", type=float, default=0.9)
    parser.add_argument("--variable-birth-score", type=float, default=0.85)
    parser.add_argument("--variable-family-min-score", type=float, default=0.75)
    parser.add_argument("--operator-birth-score", type=float, default=0.85)
    parser.add_argument("--operator-family-min-score", type=float, default=0.75)
    parser.add_argument("--faithfulness-birth-score", type=float, default=0.85)
    parser.add_argument("--faithfulness-family-min-score", type=float, default=0.75)
    return parser.parse_args()


def assert_run_dir_is_empty(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    disallowed_paths = [
        run_dir / "config.json",
        run_dir / "train_history.jsonl",
        run_dir / "checkpoints",
        run_dir / "battery",
        run_dir / "summaries",
        run_dir / "derived_dataset",
    ]
    existing = [path for path in disallowed_paths if path.exists()]
    if existing:
        raise ValueError(
            "Refusing to overwrite an existing run directory with artifacts already present: "
            + ", ".join(str(path) for path in existing)
        )


def build_schedule(*, epochs: int, dense_through_epoch: int, log_spaced_epoch_count: int, save_epoch_zero: bool) -> list[int]:
    if dense_through_epoch > epochs:
        raise ValueError("dense_through_epoch cannot exceed epochs")
    scheduled = list(range(0, dense_through_epoch + 1))
    if dense_through_epoch < epochs and log_spaced_epoch_count > 0:
        scheduled.extend(build_log_spaced_epochs(dense_through_epoch + 1, epochs, log_spaced_epoch_count))
    result = sorted(set(scheduled))
    if not save_epoch_zero and 0 in result:
        result.remove(0)
    return result


def evaluate_checkpoint_metrics(
    *,
    model: torch.nn.Module,
    bundle,
    device: torch.device,
    eval_batch_size: int,
    train_batch_loss: float | None,
) -> dict[str, float | None]:
    metrics = {
        "train_reference": evaluate_next_token_rows(
            model,
            bundle.raw_splits["train"],
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=eval_batch_size,
        ),
        "val": evaluate_next_token_rows(
            model,
            bundle.raw_splits["val"],
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=eval_batch_size,
        ),
        "test": evaluate_next_token_rows(
            model,
            bundle.raw_splits["test"],
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=eval_batch_size,
        ),
        "ood": evaluate_next_token_rows(
            model,
            bundle.raw_splits["test_ood_longer_context"],
            token_to_id=bundle.token_to_id,
            id_to_token=bundle.id_to_token,
            device=device,
            batch_size=eval_batch_size,
        ),
    }
    return {
        "train_batch_loss": train_batch_loss,
        "train_loss": metrics["train_reference"]["loss"],
        "train_accuracy": metrics["train_reference"]["accuracy"],
        "train_margin": metrics["train_reference"]["margin"],
        "val_loss": metrics["val"]["loss"],
        "val_accuracy": metrics["val"]["accuracy"],
        "val_margin": metrics["val"]["margin"],
        "test_loss": metrics["test"]["loss"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_margin": metrics["test"]["margin"],
        "ood_loss": metrics["ood"]["loss"],
        "ood_accuracy": metrics["ood"]["accuracy"],
        "ood_margin": metrics["ood"]["margin"],
    }


def train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, object]],
    token_to_id: dict[str, int],
    device: torch.device,
    batch_size: int,
    epoch: int,
    history_path: Path,
    global_step_start: int,
    batch_seed: int,
) -> dict[str, float | int]:
    batches = build_epoch_batches(rows, batch_size=batch_size, seed=batch_seed)
    if not batches:
        raise ValueError("Expected at least one batch")
    model.train()
    total_rows = 0
    total_loss = 0.0
    global_step = global_step_start
    batch_bar = tqdm(batches, desc=f"epoch {epoch}", leave=False)
    for batch_index, batch_rows in enumerate(batch_bar):
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
        batch_bar.set_postfix(
            loss=f"{batch_loss:.4f}",
            grad=f"{step_dynamics['total_grad_norm']:.3f}",
            update=f"{step_dynamics['relative_total_update_norm']:.4f}",
        )
        global_step += 1
    batch_bar.close()
    return {
        "epoch": epoch,
        "global_step_end": global_step,
        "train_loss": total_loss / total_rows,
        "train_rows": total_rows,
    }


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    bundle,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    save_reason: str,
    selected_metrics: dict[str, float | None],
) -> None:
    save_run_checkpoint(
        checkpoint_path,
        model=model,
        config={
            "vocab_size": len(bundle.vocab),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "max_seq_len": args.max_seq_len,
        },
        token_to_id=bundle.token_to_id,
        id_to_token=bundle.id_to_token,
        dataset_metadata=bundle.metadata,
        seed=args.seed,
        epoch=epoch,
        global_step=global_step,
        save_reason=save_reason,
        selected_metrics=selected_metrics,
        benchmark_name="story_text_circuit_origins",
        run_id=checkpoint_path.parent.parent.name,
        train_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
        },
        device_used_for_training=args.device,
    )


def main() -> None:
    args = parse_args()
    if args.max_seq_len < args.ood_context_length + 1:
        raise ValueError(
            f"max_seq_len must be at least ood_context_length + 1, got {args.max_seq_len} < {args.ood_context_length + 1}"
        )
    if args.max_seq_len < args.context_length + 1:
        raise ValueError(
            f"max_seq_len must be at least context_length + 1, got {args.max_seq_len} < {args.context_length + 1}"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = args.run_dir.expanduser().resolve()
    assert_run_dir_is_empty(run_dir)
    run_paths = ensure_run_directory(run_dir)
    tqdm.write(f"[run] writing artifacts to {run_dir}")

    bundle = build_story_bundle(
        text_path=args.text_path.expanduser().resolve(),
        context_length=args.context_length,
        ood_context_length=args.ood_context_length,
        stride=args.stride,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    save_story_bundle(bundle, run_dir / "derived_dataset")
    sweep_rows = build_story_eval_pack(bundle, args.sweep_base_limit)
    tqdm.write(
        "[dataset] "
        f"train={len(bundle.raw_splits['train'])} "
        f"val={len(bundle.raw_splits['val'])} "
        f"test={len(bundle.raw_splits['test'])} "
        f"ood={len(bundle.raw_splits['test_ood_longer_context'])} "
        f"vocab={len(bundle.vocab)}"
    )

    model = TinyDecoderTransformer(
        vocab_size=len(bundle.vocab),
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
        "benchmark_name": "story_text_circuit_origins",
        "text_path": str(args.text_path),
        "run_dir": str(run_dir),
        "dataset": {
            "context_length": args.context_length,
            "ood_context_length": args.ood_context_length,
            "stride": args.stride,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "sweep_base_limit": args.sweep_base_limit,
            "derived_dataset_dir": str(run_dir / "derived_dataset"),
        },
        "model": {
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
        "battery": {
            "train_probe_limit": args.train_probe_limit,
            "sae_train_limit": args.sae_train_limit,
            "sae_val_limit": args.sae_val_limit,
            "sae_hidden_multiplier": args.sae_hidden_multiplier,
            "sae_l1_coeff": args.sae_l1_coeff,
            "sae_learning_rate": args.sae_learning_rate,
            "sae_batch_size": args.sae_batch_size,
            "sae_epochs": args.sae_epochs,
            "top_features_per_site": args.top_features_per_site,
            "superposition_cosine_threshold": args.superposition_cosine_threshold,
            "role_top_k": args.role_top_k,
        },
        "summary_thresholds": {
            "behavior_birth_val_accuracy": args.behavior_birth_val_accuracy,
            "variable_birth_score": args.variable_birth_score,
            "variable_family_min_score": args.variable_family_min_score,
            "operator_birth_score": args.operator_birth_score,
            "operator_family_min_score": args.operator_family_min_score,
            "faithfulness_birth_score": args.faithfulness_birth_score,
            "faithfulness_family_min_score": args.faithfulness_family_min_score,
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
    best_val_accuracy: float | None = None
    global_step = 0

    if args.save_epoch_zero:
        tqdm.write("[checkpoint] scheduled_epoch_000: evaluating and running battery")
        epoch_zero_metrics = evaluate_checkpoint_metrics(
            model=model,
            bundle=bundle,
            device=device,
            eval_batch_size=args.eval_batch_size,
            train_batch_loss=None,
        )
        epoch_zero_metrics["epoch"] = 0
        save_checkpoint(
            checkpoint_path=run_paths["checkpoints_dir"] / "scheduled_epoch_000.pt",
            model=model,
            bundle=bundle,
            args=args,
            epoch=0,
            global_step=0,
            save_reason="scheduled",
            selected_metrics=epoch_zero_metrics,
        )
        run_story_checkpoint_battery(
            run_dir=run_dir,
            checkpoint_path=run_paths["checkpoints_dir"] / "scheduled_epoch_000.pt",
            bundle=bundle,
            sweep_rows=sweep_rows,
            train_probe_limit=args.train_probe_limit,
            eval_batch_size=args.eval_batch_size,
            sae_train_limit=args.sae_train_limit,
            sae_val_limit=args.sae_val_limit,
            sae_hidden_multiplier=args.sae_hidden_multiplier,
            sae_l1_coeff=args.sae_l1_coeff,
            sae_learning_rate=args.sae_learning_rate,
            sae_batch_size=args.sae_batch_size,
            sae_epochs=args.sae_epochs,
            sae_seed=args.seed,
            top_features_per_site=args.top_features_per_site,
            superposition_cosine_threshold=args.superposition_cosine_threshold,
            role_top_k=args.role_top_k,
            device=device,
        )
        best_val_accuracy = float(epoch_zero_metrics["val_accuracy"])

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="training", leave=True)
    for epoch in epoch_bar:
        train_result = train_epoch(
            model=model,
            optimizer=optimizer,
            rows=bundle.raw_splits["train"],
            token_to_id=bundle.token_to_id,
            device=device,
            batch_size=args.batch_size,
            epoch=epoch,
            history_path=run_paths["history_path"],
            global_step_start=global_step,
            batch_seed=args.seed + epoch,
        )
        global_step = int(train_result["global_step_end"])
        checkpoint_metrics = evaluate_checkpoint_metrics(
            model=model,
            bundle=bundle,
            device=device,
            eval_batch_size=args.eval_batch_size,
            train_batch_loss=float(train_result["train_loss"]),
        )
        checkpoint_metrics["epoch"] = epoch
        epoch_bar.set_postfix(
            train=f"{checkpoint_metrics['train_loss']:.4f}",
            val=f"{checkpoint_metrics['val_accuracy']:.3f}",
            test=f"{checkpoint_metrics['test_accuracy']:.3f}",
            ood=f"{checkpoint_metrics['ood_accuracy']:.3f}",
        )

        if epoch in scheduled_epochs:
            checkpoint_path = run_paths["checkpoints_dir"] / f"scheduled_epoch_{epoch:03d}.pt"
            tqdm.write(f"[checkpoint] {checkpoint_path.stem}: evaluating and running battery")
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                bundle=bundle,
                args=args,
                epoch=epoch,
                global_step=global_step,
                save_reason="scheduled",
                selected_metrics=checkpoint_metrics,
            )
            run_story_checkpoint_battery(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                bundle=bundle,
                sweep_rows=sweep_rows,
                train_probe_limit=args.train_probe_limit,
                eval_batch_size=args.eval_batch_size,
                sae_train_limit=args.sae_train_limit,
                sae_val_limit=args.sae_val_limit,
                sae_hidden_multiplier=args.sae_hidden_multiplier,
                sae_l1_coeff=args.sae_l1_coeff,
                sae_learning_rate=args.sae_learning_rate,
                sae_batch_size=args.sae_batch_size,
                sae_epochs=args.sae_epochs,
                sae_seed=args.seed,
                top_features_per_site=args.top_features_per_site,
                superposition_cosine_threshold=args.superposition_cosine_threshold,
                role_top_k=args.role_top_k,
                device=device,
            )

        val_accuracy = float(checkpoint_metrics["val_accuracy"])
        if best_val_accuracy is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint_path = run_paths["checkpoints_dir"] / f"best_val_epoch_{epoch:03d}.pt"
            tqdm.write(f"[checkpoint] {best_checkpoint_path.stem}: new best val {best_val_accuracy:.4f}")
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                bundle=bundle,
                args=args,
                epoch=epoch,
                global_step=global_step,
                save_reason="best_val",
                selected_metrics=checkpoint_metrics,
            )
            run_story_checkpoint_battery(
                run_dir=run_dir,
                checkpoint_path=best_checkpoint_path,
                bundle=bundle,
                sweep_rows=sweep_rows,
                train_probe_limit=args.train_probe_limit,
                eval_batch_size=args.eval_batch_size,
                sae_train_limit=args.sae_train_limit,
                sae_val_limit=args.sae_val_limit,
                sae_hidden_multiplier=args.sae_hidden_multiplier,
                sae_l1_coeff=args.sae_l1_coeff,
                sae_learning_rate=args.sae_learning_rate,
                sae_batch_size=args.sae_batch_size,
                sae_epochs=args.sae_epochs,
                sae_seed=args.seed,
                top_features_per_site=args.top_features_per_site,
                superposition_cosine_threshold=args.superposition_cosine_threshold,
                role_top_k=args.role_top_k,
                device=device,
            )
    epoch_bar.close()

    final_metrics = evaluate_checkpoint_metrics(
        model=model,
        bundle=bundle,
        device=device,
        eval_batch_size=args.eval_batch_size,
        train_batch_loss=None,
    )
    final_metrics["epoch"] = args.epochs
    final_checkpoint_path = run_paths["checkpoints_dir"] / f"final_epoch_{args.epochs:03d}.pt"
    tqdm.write(f"[checkpoint] {final_checkpoint_path.stem}: evaluating and running battery")
    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        bundle=bundle,
        args=args,
        epoch=args.epochs,
        global_step=global_step,
        save_reason="final",
        selected_metrics=final_metrics,
    )
    run_story_checkpoint_battery(
        run_dir=run_dir,
        checkpoint_path=final_checkpoint_path,
        bundle=bundle,
        sweep_rows=sweep_rows,
        train_probe_limit=args.train_probe_limit,
        eval_batch_size=args.eval_batch_size,
        sae_train_limit=args.sae_train_limit,
        sae_val_limit=args.sae_val_limit,
        sae_hidden_multiplier=args.sae_hidden_multiplier,
        sae_l1_coeff=args.sae_l1_coeff,
        sae_learning_rate=args.sae_learning_rate,
        sae_batch_size=args.sae_batch_size,
        sae_epochs=args.sae_epochs,
        sae_seed=args.seed,
        top_features_per_site=args.top_features_per_site,
        superposition_cosine_threshold=args.superposition_cosine_threshold,
        role_top_k=args.role_top_k,
        device=device,
    )

    summarize_story_run(
        run_dir=run_dir,
        role_top_k=args.role_top_k,
        behavior_birth_val_accuracy=args.behavior_birth_val_accuracy,
        variable_birth_score=args.variable_birth_score,
        variable_family_min_score=args.variable_family_min_score,
        operator_birth_score=args.operator_birth_score,
        operator_family_min_score=args.operator_family_min_score,
        faithfulness_birth_score=args.faithfulness_birth_score,
        faithfulness_family_min_score=args.faithfulness_family_min_score,
    )
    tqdm.write(f"[done] summaries written under {run_dir / 'summaries'}")


if __name__ == "__main__":
    main()
