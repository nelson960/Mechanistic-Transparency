#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts.device_utils import resolve_training_device
from scripts.kv_benchmark import (
    build_checkpoint_metrics,
    build_kv_model_config,
    instantiate_kv_model,
    load_kv_bundle,
    select_kv_training_rows,
)
from scripts.training_dynamics import (
    build_checkpoint_epoch_schedule,
    ensure_run_directory,
    load_run_manifest,
    resolve_active_training_interventions,
    save_run_manifest,
    save_run_checkpoint,
    train_next_token_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one manifest-defined KV training-dynamics run.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the run manifest JSON file.")
    return parser.parse_args()


def _assert_run_dir_is_empty(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    disallowed_paths = [
        run_dir / "manifest.json",
        run_dir / "train_history.jsonl",
        run_dir / "formation_history.jsonl",
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


def _save_checkpoint(
    *,
    run_dir_paths: dict[str, Path],
    manifest,
    model,
    bundle,
    metrics: dict,
    epoch: int,
    global_step: int,
    save_reason: str,
    device_used_for_training: str,
) -> None:
    checkpoint_path = run_dir_paths["checkpoints_dir"] / f"{save_reason}_epoch_{epoch:03d}.pt"
    save_run_checkpoint(
        checkpoint_path,
        model=model,
        config=build_kv_model_config(manifest, bundle),
        token_to_id=bundle.token_to_id,
        id_to_token=bundle.id_to_token,
        dataset_metadata=bundle.metadata,
        seed=manifest.training.seed,
        epoch=epoch,
        global_step=global_step,
        save_reason=save_reason,
        selected_metrics=metrics,
        benchmark_name=manifest.benchmark.name,
        run_id=manifest.run_id,
        train_config=manifest.training.__dict__,
        device_used_for_training=device_used_for_training,
    )


def main() -> None:
    args = parse_args()
    manifest = load_run_manifest(args.manifest)
    run_dir = manifest.output_path
    _assert_run_dir_is_empty(run_dir)
    run_dir_paths = ensure_run_directory(run_dir)
    save_run_manifest(manifest, run_dir_paths["manifest_path"])
    tqdm.write(f"[run] writing artifacts to {run_dir}")

    torch.manual_seed(manifest.training.seed)
    resolved_device = resolve_training_device(manifest.training.device)
    device = torch.device(resolved_device)
    tqdm.write(f"[device] manifest={manifest.training.device} resolved={resolved_device}")
    bundle = load_kv_bundle(manifest)
    tqdm.write(
        "[dataset] "
        f"train2={len(bundle.raw_splits[manifest.dataset.train_split_by_pairs['2']])} "
        f"train3={len(bundle.raw_splits[manifest.dataset.train_split_by_pairs['3']])} "
        f"val={len(bundle.raw_splits[manifest.dataset.eval_splits['val']])} "
        f"test={len(bundle.raw_splits[manifest.dataset.eval_splits['test']])} "
        f"ood={len(bundle.raw_splits[manifest.dataset.eval_splits['ood']])} "
        f"vocab={len(bundle.vocab)}"
    )
    model = instantiate_kv_model(manifest, bundle, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=manifest.training.learning_rate,
        weight_decay=manifest.training.weight_decay,
    )
    formation_context = None
    if manifest.formation.enabled:
        from research.phase3.scripts.kv_formation_dynamics import build_formation_context

        formation_context = build_formation_context(
            manifest,
            bundle,
            history_path=run_dir_paths["formation_history_path"],
        )

    scheduled_epochs = set(build_checkpoint_epoch_schedule(manifest))
    best_metric_name = manifest.checkpoint_schedule.best_metric
    best_metric_value: float | None = None
    global_step = 0

    reference_train_rows, _reference_stage = select_kv_training_rows(manifest, bundle, 1)
    epoch_zero_metrics = build_checkpoint_metrics(
        manifest,
        model,
        bundle,
        train_rows=reference_train_rows,
        device=device,
        train_batch_loss=None,
    )
    epoch_zero_metrics["epoch"] = 0
    epoch_zero_metrics["active_interventions"] = resolve_active_training_interventions(manifest, 0)
    if manifest.checkpoint_schedule.save_epoch_zero:
        tqdm.write("[checkpoint] scheduled_epoch_000: saving checkpoint")
        _save_checkpoint(
            run_dir_paths=run_dir_paths,
            manifest=manifest,
            model=model,
            bundle=bundle,
            metrics=epoch_zero_metrics,
            epoch=0,
            global_step=global_step,
            save_reason="scheduled",
            device_used_for_training=resolved_device,
        )
    best_metric_value = float(epoch_zero_metrics[best_metric_name])

    epoch_bar = tqdm(range(1, manifest.training.epochs + 1), desc="training", leave=True)
    for epoch in epoch_bar:
        train_rows, curriculum_stage = select_kv_training_rows(manifest, bundle, epoch)
        train_result = train_next_token_epoch(
            model,
            optimizer,
            train_rows,
            manifest=manifest,
            token_to_id=bundle.token_to_id,
            device=device,
            batch_size=manifest.training.batch_size,
            epoch=epoch,
            history_path=run_dir_paths["history_path"],
            formation_context=formation_context,
            bundle=bundle,
            global_step_start=global_step,
            batch_seed=manifest.training.seed + epoch,
            curriculum_stage=curriculum_stage,
            show_progress=True,
        )
        formation_context = train_result.get("formation_context")
        global_step = int(train_result["global_step_end"])

        checkpoint_metrics = build_checkpoint_metrics(
            manifest,
            model,
            bundle,
            train_rows=train_rows,
            device=device,
            train_batch_loss=float(train_result["train_loss"]),
        )
        checkpoint_metrics["epoch"] = epoch
        checkpoint_metrics["curriculum_stage"] = curriculum_stage
        checkpoint_metrics["active_interventions"] = resolve_active_training_interventions(manifest, epoch)
        epoch_bar.set_postfix(
            train=f"{float(checkpoint_metrics['train_loss']):.4f}",
            val=f"{float(checkpoint_metrics['val_accuracy']):.3f}",
            test=f"{float(checkpoint_metrics['test_accuracy']):.3f}",
            ood=f"{float(checkpoint_metrics['ood_accuracy']):.3f}",
        )

        if epoch in scheduled_epochs:
            tqdm.write(f"[checkpoint] scheduled_epoch_{epoch:03d}: saving checkpoint")
            _save_checkpoint(
                run_dir_paths=run_dir_paths,
                manifest=manifest,
                model=model,
                bundle=bundle,
                metrics=checkpoint_metrics,
                epoch=epoch,
                global_step=global_step,
                save_reason="scheduled",
                device_used_for_training=resolved_device,
            )

        metric_value = float(checkpoint_metrics[best_metric_name])
        if best_metric_value is None or metric_value > best_metric_value:
            best_metric_value = metric_value
            tqdm.write(f"[checkpoint] best_val_epoch_{epoch:03d}: new best {best_metric_name}={metric_value:.4f}")
            _save_checkpoint(
                run_dir_paths=run_dir_paths,
                manifest=manifest,
                model=model,
                bundle=bundle,
                metrics=checkpoint_metrics,
                epoch=epoch,
                global_step=global_step,
                save_reason="best_val",
                device_used_for_training=resolved_device,
            )
    epoch_bar.close()

    if manifest.checkpoint_schedule.save_final:
        final_train_rows, final_stage = select_kv_training_rows(manifest, bundle, manifest.training.epochs)
        final_metrics = build_checkpoint_metrics(
            manifest,
            model,
            bundle,
            train_rows=final_train_rows,
            device=device,
            train_batch_loss=None,
        )
        final_metrics["epoch"] = manifest.training.epochs
        final_metrics["curriculum_stage"] = final_stage
        final_metrics["active_interventions"] = resolve_active_training_interventions(
            manifest,
            manifest.training.epochs,
        )
        tqdm.write(f"[checkpoint] final_epoch_{manifest.training.epochs:03d}: saving checkpoint")
        _save_checkpoint(
            run_dir_paths=run_dir_paths,
            manifest=manifest,
            model=model,
            bundle=bundle,
            metrics=final_metrics,
            epoch=manifest.training.epochs,
            global_step=global_step,
            save_reason="final",
            device_used_for_training=resolved_device,
        )
    tqdm.write(f"[done] checkpoints written under {run_dir_paths['checkpoints_dir']}")


if __name__ == "__main__":
    main()
