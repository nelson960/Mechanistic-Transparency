#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from scripts.device_utils import resolve_training_device
from scripts.microlanguage_world_benchmark import (
    build_microlanguage_world_checkpoint_metrics,
    build_microlanguage_world_model_config,
    instantiate_microlanguage_world_model,
    load_microlanguage_world_bundle,
    select_microlanguage_world_training_rows,
    train_microlanguage_world_epoch,
)
from scripts.training_dynamics import (
    build_checkpoint_epoch_schedule,
    ensure_run_directory,
    load_run_manifest,
    resolve_active_training_interventions,
    save_run_checkpoint,
    save_run_manifest,
)


def _clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _load_cloned_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({name: tensor.clone() for name, tensor in state_dict.items()})


def _metric_improved(
    *,
    metric_value: float,
    best_metric_value: float,
    best_metric_name: str,
    min_delta: float,
) -> bool:
    if best_metric_name.endswith("_loss"):
        return metric_value < (best_metric_value - min_delta)
    return metric_value > (best_metric_value + min_delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one manifest-defined microlanguage world run.")
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
        config=build_microlanguage_world_model_config(manifest, bundle),
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
        model_name=getattr(model, "model_name", "tiny_decoder_transformer"),
        device_used_for_training=device_used_for_training,
    )


def main() -> None:
    args = parse_args()
    manifest = load_run_manifest(args.manifest)
    if manifest.benchmark.name != "microlanguage_world_next_token":
        raise ValueError(
            "train_microlanguage_world_run.py requires benchmark.name='microlanguage_world_next_token'"
        )
    run_dir = manifest.output_path
    _assert_run_dir_is_empty(run_dir)
    run_dir_paths = ensure_run_directory(run_dir)
    save_run_manifest(manifest, run_dir_paths["manifest_path"])
    tqdm.write(f"[run] writing artifacts to {run_dir}")

    torch.manual_seed(manifest.training.seed)
    resolved_device = resolve_training_device(manifest.training.device)
    device = torch.device(resolved_device)
    tqdm.write(f"[device] manifest={manifest.training.device} resolved={resolved_device}")
    bundle = load_microlanguage_world_bundle(manifest)
    train_split_name = manifest.dataset.train_split_by_pairs["default"]
    tqdm.write(
        "[dataset] "
        f"train={len(bundle.raw_splits[train_split_name])} "
        f"val={len(bundle.raw_splits[manifest.dataset.eval_splits['val']])} "
        f"test={len(bundle.raw_splits[manifest.dataset.eval_splits['test']])} "
        f"ood={len(bundle.raw_splits[manifest.dataset.eval_splits['ood']])} "
        f"vocab={len(bundle.vocab)}"
    )
    model = instantiate_microlanguage_world_model(manifest, bundle, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=manifest.training.learning_rate,
        weight_decay=manifest.training.weight_decay,
    )

    scheduled_epochs = set(build_checkpoint_epoch_schedule(manifest))
    best_metric_name = manifest.checkpoint_schedule.best_metric
    best_metric_value: float | None = None
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    global_step = 0

    reference_train_rows, reference_stage = select_microlanguage_world_training_rows(manifest, bundle, 1)
    epoch_zero_metrics = build_microlanguage_world_checkpoint_metrics(
        manifest,
        model,
        bundle,
        train_rows=reference_train_rows,
        device=device,
        train_batch_loss=None,
    )
    epoch_zero_metrics["epoch"] = 0
    epoch_zero_metrics["curriculum_stage"] = reference_stage
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
    best_epoch = 0
    best_metrics = dict(epoch_zero_metrics)
    if manifest.early_stopping.restore_best_state:
        best_model_state = _clone_model_state_dict(model)

    epoch_bar = tqdm(range(1, manifest.training.epochs + 1), desc="training", leave=True)
    for epoch in epoch_bar:
        train_rows, curriculum_stage = select_microlanguage_world_training_rows(manifest, bundle, epoch)
        train_result = train_microlanguage_world_epoch(
            model,
            optimizer,
            train_rows,
            manifest=manifest,
            bundle=bundle,
            device=device,
            epoch=epoch,
            history_path=run_dir_paths["history_path"],
            global_step_start=global_step,
            batch_seed=manifest.training.seed + epoch,
            curriculum_stage=curriculum_stage,
            show_progress=True,
        )
        global_step = int(train_result["global_step_end"])

        checkpoint_metrics = build_microlanguage_world_checkpoint_metrics(
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
        if best_metric_value is None:
            raise ValueError("best_metric_value should be initialized before the training loop")
        if _metric_improved(
            metric_value=metric_value,
            best_metric_value=best_metric_value,
            best_metric_name=best_metric_name,
            min_delta=manifest.early_stopping.min_delta,
        ):
            best_metric_value = metric_value
            best_epoch = epoch
            best_metrics = dict(checkpoint_metrics)
            epochs_without_improvement = 0
            if manifest.early_stopping.restore_best_state:
                best_model_state = _clone_model_state_dict(model)
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
        elif manifest.early_stopping.enabled and epoch >= manifest.early_stopping.warmup_epochs:
            epochs_without_improvement += 1
            if epochs_without_improvement >= manifest.early_stopping.patience:
                stop_message = (
                    f"[early-stop] epoch={epoch:03d} "
                    f"best_epoch={best_epoch:03d} "
                    f"metric={best_metric_name} "
                    f"best={best_metric_value:.4f} "
                    f"patience={manifest.early_stopping.patience}"
                )
                tqdm.write(stop_message)
                if manifest.early_stopping.restore_best_state:
                    if best_model_state is None or best_metrics is None or best_epoch is None:
                        raise ValueError("Early stopping requested restore_best_state, but best checkpoint state is missing")
                    _load_cloned_model_state_dict(model, best_model_state)
                    if best_epoch == 0:
                        restored_train_rows, restored_stage = reference_train_rows, reference_stage
                    else:
                        restored_train_rows, restored_stage = select_microlanguage_world_training_rows(
                            manifest,
                            bundle,
                            best_epoch,
                        )
                    restored_metrics = build_microlanguage_world_checkpoint_metrics(
                        manifest,
                        model,
                        bundle,
                        train_rows=restored_train_rows,
                        device=device,
                        train_batch_loss=None,
                    )
                    restored_metrics["epoch"] = epoch
                    restored_metrics["restored_from_epoch"] = best_epoch
                    restored_metrics["curriculum_stage"] = restored_stage
                    restored_metrics["active_interventions"] = resolve_active_training_interventions(manifest, epoch)
                    metrics_to_save = restored_metrics
                else:
                    metrics_to_save = dict(checkpoint_metrics)
                    metrics_to_save["restored_from_epoch"] = None
                _save_checkpoint(
                    run_dir_paths=run_dir_paths,
                    manifest=manifest,
                    model=model,
                    bundle=bundle,
                    metrics=metrics_to_save,
                    epoch=epoch,
                    global_step=global_step,
                    save_reason="early_stop",
                    device_used_for_training=resolved_device,
                )
                early_stopped = True
                break
    epoch_bar.close()

    if manifest.checkpoint_schedule.save_final and not early_stopped:
        final_train_rows, final_stage = select_microlanguage_world_training_rows(
            manifest,
            bundle,
            manifest.training.epochs,
        )
        final_metrics = build_microlanguage_world_checkpoint_metrics(
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
