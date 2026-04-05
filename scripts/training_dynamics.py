from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.tiny_transformer_core import forward_tiny_decoder_with_interventions


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str


@dataclass(frozen=True)
class DatasetConfig:
    dataset_dir: str
    train_split_by_pairs: dict[str, str]
    eval_splits: dict[str, str]
    sweep_base_split: str


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    max_seq_len: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    curriculum: str
    device: str


@dataclass(frozen=True)
class CheckpointScheduleConfig:
    dense_through_epoch: int
    log_spaced_epoch_count: int
    save_epoch_zero: bool
    save_final: bool
    best_metric: str


@dataclass(frozen=True)
class BatteryConfig:
    train_probe_limit: int
    sweep_base_limit: int
    eval_batch_size: int
    role_top_k: int


@dataclass(frozen=True)
class SAETrackingConfig:
    enabled: bool
    sites: list[str]
    train_limit: int
    val_limit: int
    hidden_multiplier: int
    l1_coeff: float
    learning_rate: float
    batch_size: int
    epochs: int
    seed: int
    top_features_per_site: int
    superposition_cosine_threshold: float


@dataclass(frozen=True)
class FormationHeadConfig:
    layer_index: int
    head_index: int


@dataclass(frozen=True)
class FormationConfig:
    enabled: bool
    log_every_steps: int
    eval_pack_size: int
    gradient_family_interval: int
    candidate_mode: str
    candidate_support_head: FormationHeadConfig | None
    candidate_retrieval_head: FormationHeadConfig | None
    candidate_placebo_head: FormationHeadConfig | None
    measure_qk_margin: bool
    measure_ov_lift: bool
    measure_path_gain: bool
    measure_correct_slot_attention: bool
    save_family_gradients: bool
    save_logit_decomposition: bool
    transition_metric_name: str
    transition_metric_delta: float
    transition_boost_steps: int
    q_birth_accuracy: float
    r_birth_accuracy: float
    w_birth_accuracy: float


@dataclass(frozen=True)
class TrainingInterventionConfig:
    name: str
    kind: str
    layer_index: int
    head_index: int | None
    epoch_start: int
    epoch_end: int
    scale: float
    position: str


@dataclass(frozen=True)
class SummaryThresholdsConfig:
    behavior_birth_val_accuracy: float
    operator_birth_score: float
    operator_family_min_score: float
    variable_birth_score: float
    variable_family_min_score: float
    faithfulness_birth_score: float
    faithfulness_family_min_score: float


@dataclass(frozen=True)
class RunManifest:
    benchmark: BenchmarkConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    checkpoint_schedule: CheckpointScheduleConfig
    battery: BatteryConfig
    sae_tracking: SAETrackingConfig
    formation: FormationConfig
    training_interventions: list[TrainingInterventionConfig]
    summary_thresholds: SummaryThresholdsConfig
    output_dir: str

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def run_id(self) -> str:
        return self.output_path.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expect_mapping(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object")
    return value


def _expect_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Manifest field {key!r} must be a positive integer")
    return value


def _expect_non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Manifest field {key!r} must be a non-negative integer")
    return value


def _expect_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Manifest field {key!r} must be numeric")
    return float(value)


def _expect_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string")
    return value


def _expect_string_mapping(payload: dict[str, Any], key: str) -> dict[str, str]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty object")
    converted: dict[str, str] = {}
    for sub_key, sub_value in value.items():
        if not isinstance(sub_key, str) or not isinstance(sub_value, str) or not sub_value.strip():
            raise ValueError(f"Manifest field {key!r} must map strings to non-empty strings")
        converted[sub_key] = sub_value
    return converted


def _expect_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Manifest field {key!r} must be a boolean")
    return value


def _expect_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {key!r} must be a list")
    return value


def _expect_string_list(payload: dict[str, Any], key: str) -> list[str]:
    values = _expect_list(payload, key)
    converted: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manifest field {key!r} must contain only non-empty strings")
        converted.append(value)
    return converted


def _expect_optional_head_config(payload: dict[str, Any], key: str) -> FormationHeadConfig | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be null or an object")
    return FormationHeadConfig(
        layer_index=_expect_non_negative_int(value, "layer_index"),
        head_index=_expect_non_negative_int(value, "head_index"),
    )


def _build_disabled_formation_config() -> FormationConfig:
    return FormationConfig(
        enabled=False,
        log_every_steps=1,
        eval_pack_size=1,
        gradient_family_interval=1,
        candidate_mode="fixed",
        candidate_support_head=None,
        candidate_retrieval_head=None,
        candidate_placebo_head=None,
        measure_qk_margin=False,
        measure_ov_lift=False,
        measure_path_gain=False,
        measure_correct_slot_attention=False,
        save_family_gradients=False,
        save_logit_decomposition=False,
        transition_metric_name="R",
        transition_metric_delta=0.0,
        transition_boost_steps=0,
        q_birth_accuracy=1.0,
        r_birth_accuracy=1.0,
        w_birth_accuracy=1.0,
    )


def build_run_manifest(payload: dict[str, Any]) -> RunManifest:
    benchmark_payload = _expect_mapping(payload, "benchmark")
    dataset_payload = _expect_mapping(payload, "dataset")
    model_payload = _expect_mapping(payload, "model")
    training_payload = _expect_mapping(payload, "training")
    schedule_payload = _expect_mapping(payload, "checkpoint_schedule")
    battery_payload = _expect_mapping(payload, "battery")
    sae_payload = _expect_mapping(payload, "sae_tracking")
    formation_payload = payload.get("formation")
    interventions_payload = _expect_list(payload, "training_interventions")
    thresholds_payload = _expect_mapping(payload, "summary_thresholds")
    output_dir = _expect_string(payload, "output_dir")

    benchmark = BenchmarkConfig(name=_expect_string(benchmark_payload, "name"))
    dataset = DatasetConfig(
        dataset_dir=_expect_string(dataset_payload, "dataset_dir"),
        train_split_by_pairs=_expect_string_mapping(dataset_payload, "train_split_by_pairs"),
        eval_splits=_expect_string_mapping(dataset_payload, "eval_splits"),
        sweep_base_split=_expect_string(dataset_payload, "sweep_base_split"),
    )
    model = ModelConfig(
        d_model=_expect_positive_int(model_payload, "d_model"),
        n_heads=_expect_positive_int(model_payload, "n_heads"),
        d_ff=_expect_positive_int(model_payload, "d_ff"),
        n_layers=_expect_positive_int(model_payload, "n_layers"),
        max_seq_len=_expect_positive_int(model_payload, "max_seq_len"),
    )
    training = TrainingConfig(
        epochs=_expect_positive_int(training_payload, "epochs"),
        batch_size=_expect_positive_int(training_payload, "batch_size"),
        learning_rate=_expect_float(training_payload, "learning_rate"),
        weight_decay=_expect_float(training_payload, "weight_decay"),
        seed=_expect_non_negative_int(training_payload, "seed"),
        curriculum=_expect_string(training_payload, "curriculum"),
        device=_expect_string(training_payload, "device"),
    )
    checkpoint_schedule = CheckpointScheduleConfig(
        dense_through_epoch=_expect_non_negative_int(schedule_payload, "dense_through_epoch"),
        log_spaced_epoch_count=_expect_non_negative_int(schedule_payload, "log_spaced_epoch_count"),
        save_epoch_zero=_expect_bool(schedule_payload, "save_epoch_zero"),
        save_final=_expect_bool(schedule_payload, "save_final"),
        best_metric=_expect_string(schedule_payload, "best_metric"),
    )
    battery = BatteryConfig(
        train_probe_limit=_expect_positive_int(battery_payload, "train_probe_limit"),
        sweep_base_limit=_expect_positive_int(battery_payload, "sweep_base_limit"),
        eval_batch_size=_expect_positive_int(battery_payload, "eval_batch_size"),
        role_top_k=_expect_positive_int(battery_payload, "role_top_k"),
    )
    sae_tracking = SAETrackingConfig(
        enabled=_expect_bool(sae_payload, "enabled"),
        sites=_expect_string_list(sae_payload, "sites"),
        train_limit=_expect_positive_int(sae_payload, "train_limit"),
        val_limit=_expect_positive_int(sae_payload, "val_limit"),
        hidden_multiplier=_expect_positive_int(sae_payload, "hidden_multiplier"),
        l1_coeff=_expect_float(sae_payload, "l1_coeff"),
        learning_rate=_expect_float(sae_payload, "learning_rate"),
        batch_size=_expect_positive_int(sae_payload, "batch_size"),
        epochs=_expect_positive_int(sae_payload, "epochs"),
        seed=_expect_non_negative_int(sae_payload, "seed"),
        top_features_per_site=_expect_positive_int(sae_payload, "top_features_per_site"),
        superposition_cosine_threshold=_expect_float(sae_payload, "superposition_cosine_threshold"),
    )
    if formation_payload is None:
        formation = _build_disabled_formation_config()
    else:
        if not isinstance(formation_payload, dict):
            raise ValueError("Manifest field 'formation' must be an object when provided")
        formation = FormationConfig(
            enabled=_expect_bool(formation_payload, "enabled"),
            log_every_steps=_expect_positive_int(formation_payload, "log_every_steps"),
            eval_pack_size=_expect_positive_int(formation_payload, "eval_pack_size"),
            gradient_family_interval=_expect_positive_int(formation_payload, "gradient_family_interval"),
            candidate_mode=_expect_string(formation_payload, "candidate_mode"),
            candidate_support_head=_expect_optional_head_config(formation_payload, "candidate_support_head"),
            candidate_retrieval_head=_expect_optional_head_config(formation_payload, "candidate_retrieval_head"),
            candidate_placebo_head=_expect_optional_head_config(formation_payload, "candidate_placebo_head"),
            measure_qk_margin=_expect_bool(formation_payload, "measure_qk_margin"),
            measure_ov_lift=_expect_bool(formation_payload, "measure_ov_lift"),
            measure_path_gain=_expect_bool(formation_payload, "measure_path_gain"),
            measure_correct_slot_attention=_expect_bool(formation_payload, "measure_correct_slot_attention"),
            save_family_gradients=_expect_bool(formation_payload, "save_family_gradients"),
            save_logit_decomposition=_expect_bool(formation_payload, "save_logit_decomposition"),
            transition_metric_name=_expect_string(formation_payload, "transition_metric_name"),
            transition_metric_delta=_expect_float(formation_payload, "transition_metric_delta"),
            transition_boost_steps=_expect_non_negative_int(formation_payload, "transition_boost_steps"),
            q_birth_accuracy=_expect_float(formation_payload, "q_birth_accuracy"),
            r_birth_accuracy=_expect_float(formation_payload, "r_birth_accuracy"),
            w_birth_accuracy=_expect_float(formation_payload, "w_birth_accuracy"),
        )
    training_interventions: list[TrainingInterventionConfig] = []
    for index, intervention_payload in enumerate(interventions_payload):
        if not isinstance(intervention_payload, dict):
            raise ValueError(f"Manifest training_interventions[{index}] must be an object")
        head_index = intervention_payload.get("head_index")
        if head_index is not None and (not isinstance(head_index, int) or head_index < 0):
            raise ValueError(
                f"Manifest training_interventions[{index}].head_index must be null or a non-negative integer"
            )
        training_interventions.append(
            TrainingInterventionConfig(
                name=_expect_string(intervention_payload, "name"),
                kind=_expect_string(intervention_payload, "kind"),
                layer_index=_expect_non_negative_int(intervention_payload, "layer_index"),
                head_index=head_index,
                epoch_start=_expect_positive_int(intervention_payload, "epoch_start"),
                epoch_end=_expect_positive_int(intervention_payload, "epoch_end"),
                scale=_expect_float(intervention_payload, "scale"),
                position=_expect_string(intervention_payload, "position"),
            )
        )
    summary_thresholds = SummaryThresholdsConfig(
        behavior_birth_val_accuracy=_expect_float(thresholds_payload, "behavior_birth_val_accuracy"),
        operator_birth_score=_expect_float(thresholds_payload, "operator_birth_score"),
        operator_family_min_score=_expect_float(thresholds_payload, "operator_family_min_score"),
        variable_birth_score=_expect_float(thresholds_payload, "variable_birth_score"),
        variable_family_min_score=_expect_float(thresholds_payload, "variable_family_min_score"),
        faithfulness_birth_score=_expect_float(thresholds_payload, "faithfulness_birth_score"),
        faithfulness_family_min_score=_expect_float(thresholds_payload, "faithfulness_family_min_score"),
    )

    if benchmark.name != "kv_retrieval":
        raise ValueError(f"Unsupported benchmark {benchmark.name!r}; only 'kv_retrieval' is implemented")
    if training.curriculum not in {"on", "off"}:
        raise ValueError("Manifest training.curriculum must be 'on' or 'off'")
    if model.d_model % model.n_heads != 0:
        raise ValueError("Manifest model.d_model must be divisible by model.n_heads")
    if checkpoint_schedule.dense_through_epoch > training.epochs:
        raise ValueError("Manifest checkpoint_schedule.dense_through_epoch exceeds training.epochs")
    if dataset.sweep_base_split not in dataset.eval_splits.values():
        raise ValueError(
            "Manifest dataset.sweep_base_split must match one of the dataset.eval_splits values"
        )
    if "2" not in dataset.train_split_by_pairs or "3" not in dataset.train_split_by_pairs:
        raise ValueError("Manifest dataset.train_split_by_pairs must define both '2' and '3' training splits")
    if sae_tracking.enabled and not sae_tracking.sites:
        raise ValueError("Manifest sae_tracking.sites must be non-empty when sae_tracking.enabled is true")
    if sae_tracking.superposition_cosine_threshold < 0.0 or sae_tracking.superposition_cosine_threshold > 1.0:
        raise ValueError("Manifest sae_tracking.superposition_cosine_threshold must lie in [0, 1]")
    if formation.candidate_mode not in {"fixed", "discovered"}:
        raise ValueError("Manifest formation.candidate_mode must be 'fixed' or 'discovered'")
    if formation.transition_metric_name not in {"Q", "R", "W"}:
        raise ValueError("Manifest formation.transition_metric_name must be one of 'Q', 'R', or 'W'")
    for threshold_name, threshold_value in (
        ("q_birth_accuracy", formation.q_birth_accuracy),
        ("r_birth_accuracy", formation.r_birth_accuracy),
        ("w_birth_accuracy", formation.w_birth_accuracy),
    ):
        if threshold_value < 0.0 or threshold_value > 1.0:
            raise ValueError(f"Manifest formation.{threshold_name} must lie in [0, 1]")
    if formation.transition_metric_delta < 0.0:
        raise ValueError("Manifest formation.transition_metric_delta must be non-negative")
    if formation.enabled and formation.candidate_mode == "fixed":
        if formation.candidate_support_head is None or formation.candidate_retrieval_head is None:
            raise ValueError(
                "Manifest formation fixed mode requires candidate_support_head and candidate_retrieval_head"
            )
    for role_name, candidate in (
        ("candidate_support_head", formation.candidate_support_head),
        ("candidate_retrieval_head", formation.candidate_retrieval_head),
        ("candidate_placebo_head", formation.candidate_placebo_head),
    ):
        if candidate is None:
            continue
        if candidate.layer_index >= model.n_layers:
            raise ValueError(
                f"Manifest formation.{role_name}.layer_index={candidate.layer_index} exceeds model.n_layers={model.n_layers}"
            )
        if candidate.head_index >= model.n_heads:
            raise ValueError(
                f"Manifest formation.{role_name}.head_index={candidate.head_index} exceeds model.n_heads={model.n_heads}"
            )
    for intervention in training_interventions:
        if intervention.kind not in {"head_resid_final_scale", "mlp_out_final_scale"}:
            raise ValueError(
                "Manifest training_interventions kinds must be "
                "'head_resid_final_scale' or 'mlp_out_final_scale'"
            )
        if intervention.position != "final":
            raise ValueError("Manifest training_interventions currently only support position='final'")
        if intervention.epoch_start > intervention.epoch_end:
            raise ValueError(
                f"Training intervention {intervention.name!r} has epoch_start > epoch_end"
            )
        if intervention.epoch_end > training.epochs:
            raise ValueError(
                f"Training intervention {intervention.name!r} ends after training.epochs"
            )
        if intervention.kind == "head_resid_final_scale" and intervention.head_index is None:
            raise ValueError(
                f"Training intervention {intervention.name!r} requires head_index for head_resid_final_scale"
            )
        if intervention.kind == "mlp_out_final_scale" and intervention.head_index is not None:
            raise ValueError(
                f"Training intervention {intervention.name!r} must not set head_index for mlp_out_final_scale"
            )
        if intervention.layer_index >= model.n_layers:
            raise ValueError(
                f"Training intervention {intervention.name!r} targets layer_index={intervention.layer_index}, "
                f"but model.n_layers={model.n_layers}"
            )
        if intervention.head_index is not None and intervention.head_index >= model.n_heads:
            raise ValueError(
                f"Training intervention {intervention.name!r} targets head_index={intervention.head_index}, "
                f"but model.n_heads={model.n_heads}"
            )

    return RunManifest(
        benchmark=benchmark,
        dataset=dataset,
        model=model,
        training=training,
        checkpoint_schedule=checkpoint_schedule,
        battery=battery,
        sae_tracking=sae_tracking,
        formation=formation,
        training_interventions=training_interventions,
        summary_thresholds=summary_thresholds,
        output_dir=output_dir,
    )


def load_run_manifest(path: Path) -> RunManifest:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be an object, got {type(payload).__name__}")
    return build_run_manifest(payload)


def save_run_manifest(manifest: RunManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def ensure_run_directory(run_dir: Path) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    summaries_dir = run_dir / "summaries"
    battery_dir = run_dir / "battery"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    battery_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "summaries_dir": summaries_dir,
        "battery_dir": battery_dir,
        "history_path": run_dir / "train_history.jsonl",
        "formation_history_path": run_dir / "formation_history.jsonl",
        "manifest_path": run_dir / "manifest.json",
    }


def _nearest_available_epoch(ideal: float, available: set[int]) -> int:
    if not available:
        raise ValueError("No epochs remain available while building log-spaced checkpoint schedule")
    return min(available, key=lambda epoch: (abs(epoch - ideal), epoch))


def build_log_spaced_epochs(start_epoch: int, end_epoch: int, count: int) -> list[int]:
    if count == 0:
        return []
    if start_epoch <= 0:
        raise ValueError("Log-spaced checkpoint epochs require start_epoch >= 1")
    if end_epoch < start_epoch:
        raise ValueError("Log-spaced checkpoint epochs require end_epoch >= start_epoch")
    epoch_span = end_epoch - start_epoch + 1
    if count > epoch_span:
        raise ValueError(
            f"Cannot select {count} unique log-spaced epochs from span [{start_epoch}, {end_epoch}]"
        )

    ideals = torch.logspace(
        math.log10(float(start_epoch)),
        math.log10(float(end_epoch)),
        steps=count,
    ).tolist()
    available = set(range(start_epoch, end_epoch + 1))
    chosen: list[int] = []
    for ideal in ideals:
        nearest = _nearest_available_epoch(ideal, available)
        chosen.append(nearest)
        available.remove(nearest)
    return sorted(chosen)


def build_checkpoint_epoch_schedule(manifest: RunManifest) -> list[int]:
    dense_epochs = list(range(0, manifest.checkpoint_schedule.dense_through_epoch + 1))
    log_epochs = build_log_spaced_epochs(
        manifest.checkpoint_schedule.dense_through_epoch + 1,
        manifest.training.epochs,
        manifest.checkpoint_schedule.log_spaced_epoch_count,
    )
    scheduled_epochs = sorted(set(dense_epochs + log_epochs))
    if not manifest.checkpoint_schedule.save_epoch_zero and 0 in scheduled_epochs:
        scheduled_epochs.remove(0)
    return scheduled_epochs


def checkpoint_filename(epoch: int, save_reason: str) -> str:
    safe_reason = save_reason.replace(" ", "_")
    return f"{safe_reason}_epoch_{epoch:03d}.pt"


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")
    checkpoints = sorted(path for path in checkpoints_dir.glob("*.pt") if path.is_file())
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    return checkpoints


def save_run_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
    dataset_metadata: dict[str, Any],
    seed: int,
    epoch: int,
    global_step: int,
    save_reason: str,
    selected_metrics: dict[str, Any],
    benchmark_name: str,
    run_id: str,
    train_config: dict[str, Any],
    model_name: str = "tiny_decoder_transformer",
    device_used_for_training: str | None = None,
) -> None:
    payload = {
        "model_name": model_name,
        "benchmark_name": benchmark_name,
        "run_id": run_id,
        "config": config,
        "state_dict": model.state_dict(),
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "dataset_metadata": dataset_metadata,
        "seed": seed,
        "selected_epoch": epoch,
        "selected_metrics": selected_metrics,
        "global_step": global_step,
        "epoch": epoch,
        "save_reason": save_reason,
        "train_config": train_config,
        "metrics_at_save": selected_metrics,
        "device_used_for_training": device_used_for_training or train_config.get("device"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _flatten_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1).cpu()


def _safe_cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def iter_tracked_parameter_groups(model: torch.nn.Module):
    yield {
        "name": "token_embed.weight",
        "tensor": model.token_embed.weight,
        "grad_source": model.token_embed.weight,
        "layer_index": None,
        "component": "token_embed",
    }
    for layer_index, block in enumerate(model.blocks):
        yield {
            "name": f"block{layer_index + 1}.norm1.weight",
            "tensor": block.norm1.weight,
            "grad_source": block.norm1.weight,
            "layer_index": layer_index,
            "component": "norm1",
        }
        yield {
            "name": f"block{layer_index + 1}.attn.q_proj.weight",
            "tensor": block.attn.q_proj.weight,
            "grad_source": block.attn.q_proj.weight,
            "layer_index": layer_index,
            "component": "q_proj",
        }
        yield {
            "name": f"block{layer_index + 1}.attn.k_proj.weight",
            "tensor": block.attn.k_proj.weight,
            "grad_source": block.attn.k_proj.weight,
            "layer_index": layer_index,
            "component": "k_proj",
        }
        yield {
            "name": f"block{layer_index + 1}.attn.v_proj.weight",
            "tensor": block.attn.v_proj.weight,
            "grad_source": block.attn.v_proj.weight,
            "layer_index": layer_index,
            "component": "v_proj",
        }
        yield {
            "name": f"block{layer_index + 1}.attn.o_proj.weight",
            "tensor": block.attn.o_proj.weight,
            "grad_source": block.attn.o_proj.weight,
            "layer_index": layer_index,
            "component": "o_proj",
        }
        yield {
            "name": f"block{layer_index + 1}.norm2.weight",
            "tensor": block.norm2.weight,
            "grad_source": block.norm2.weight,
            "layer_index": layer_index,
            "component": "norm2",
        }
        yield {
            "name": f"block{layer_index + 1}.mlp.gate_proj.weight",
            "tensor": block.mlp.gate_proj.weight,
            "grad_source": block.mlp.gate_proj.weight,
            "layer_index": layer_index,
            "component": "gate_proj",
        }
        yield {
            "name": f"block{layer_index + 1}.mlp.up_proj.weight",
            "tensor": block.mlp.up_proj.weight,
            "grad_source": block.mlp.up_proj.weight,
            "layer_index": layer_index,
            "component": "up_proj",
        }
        yield {
            "name": f"block{layer_index + 1}.mlp.down_proj.weight",
            "tensor": block.mlp.down_proj.weight,
            "grad_source": block.mlp.down_proj.weight,
            "layer_index": layer_index,
            "component": "down_proj",
        }
        head_dim = block.attn.head_dim
        for head_index in range(block.attn.n_heads):
            head_start = head_index * head_dim
            head_stop = head_start + head_dim
            yield {
                "name": f"block{layer_index + 1}.head{head_index}.q_proj_slice",
                "tensor": block.attn.q_proj.weight[head_start:head_stop, :],
                "grad_source": block.attn.q_proj.weight,
                "grad_row_start": head_start,
                "grad_row_stop": head_stop,
                "layer_index": layer_index,
                "head_index": head_index,
                "component": "q_proj_slice",
            }
            yield {
                "name": f"block{layer_index + 1}.head{head_index}.k_proj_slice",
                "tensor": block.attn.k_proj.weight[head_start:head_stop, :],
                "grad_source": block.attn.k_proj.weight,
                "grad_row_start": head_start,
                "grad_row_stop": head_stop,
                "layer_index": layer_index,
                "head_index": head_index,
                "component": "k_proj_slice",
            }
            yield {
                "name": f"block{layer_index + 1}.head{head_index}.v_proj_slice",
                "tensor": block.attn.v_proj.weight[head_start:head_stop, :],
                "grad_source": block.attn.v_proj.weight,
                "grad_row_start": head_start,
                "grad_row_stop": head_stop,
                "layer_index": layer_index,
                "head_index": head_index,
                "component": "v_proj_slice",
            }
            yield {
                "name": f"block{layer_index + 1}.head{head_index}.o_proj_slice",
                "tensor": block.attn.o_proj.weight[:, head_start:head_stop],
                "grad_source": block.attn.o_proj.weight,
                "grad_col_start": head_start,
                "grad_col_stop": head_stop,
                "layer_index": layer_index,
                "head_index": head_index,
                "component": "o_proj_slice",
            }
    yield {
        "name": "norm_final.weight",
        "tensor": model.norm_final.weight,
        "grad_source": model.norm_final.weight,
        "layer_index": None,
        "component": "norm_final",
    }


def snapshot_tracked_parameter_groups(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        spec["name"]: spec["tensor"].detach().float().cpu().clone()
        for spec in iter_tracked_parameter_groups(model)
    }


def build_step_dynamics_record(
    model: torch.nn.Module,
    pre_step_snapshot: dict[str, torch.Tensor],
) -> dict[str, Any]:
    parameter_metrics: list[dict[str, Any]] = []
    total_grad_sq_norm = 0.0
    total_param_sq_norm_pre = 0.0
    total_param_sq_norm_post = 0.0
    total_update_sq_norm = 0.0

    for spec in iter_tracked_parameter_groups(model):
        name = str(spec["name"])
        tensor = spec["tensor"]
        pre_tensor = pre_step_snapshot[name]
        post_tensor = tensor.detach().float().cpu()
        grad_source = spec.get("grad_source")
        if isinstance(grad_source, torch.nn.Parameter) and grad_source.grad is not None:
            grad_tensor = grad_source.grad.detach().float().cpu()
            if spec.get("grad_row_start") is not None or spec.get("grad_row_stop") is not None:
                grad_tensor = grad_tensor[
                    int(spec.get("grad_row_start") or 0) : int(spec.get("grad_row_stop") or grad_tensor.shape[0]),
                    :,
                ]
            if spec.get("grad_col_start") is not None or spec.get("grad_col_stop") is not None:
                grad_tensor = grad_tensor[
                    :,
                    int(spec.get("grad_col_start") or 0) : int(spec.get("grad_col_stop") or grad_tensor.shape[1]),
                ]
        else:
            grad_tensor = torch.zeros_like(post_tensor)
        update_tensor = post_tensor - pre_tensor

        param_norm_pre = float(pre_tensor.norm().item())
        param_norm_post = float(post_tensor.norm().item())
        grad_norm = float(grad_tensor.norm().item())
        update_norm = float(update_tensor.norm().item())
        total_grad_sq_norm += grad_norm ** 2
        total_param_sq_norm_pre += param_norm_pre ** 2
        total_param_sq_norm_post += param_norm_post ** 2
        total_update_sq_norm += update_norm ** 2

        parameter_metrics.append(
            {
                "name": name,
                "layer_index": spec.get("layer_index"),
                "head_index": spec.get("head_index"),
                "component": spec.get("component"),
                "shape": list(post_tensor.shape),
                "param_norm_pre": param_norm_pre,
                "param_norm_post": param_norm_post,
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "relative_update_norm": (
                    update_norm / param_norm_pre if param_norm_pre > 0.0 else float("nan")
                ),
                "cosine_to_previous": _safe_cosine_similarity(pre_tensor.reshape(-1), post_tensor.reshape(-1)),
            }
        )

    return {
        "parameter_metrics": parameter_metrics,
        "total_grad_norm": math.sqrt(total_grad_sq_norm),
        "total_param_norm_pre": math.sqrt(total_param_sq_norm_pre),
        "total_param_norm_post": math.sqrt(total_param_sq_norm_post),
        "total_update_norm": math.sqrt(total_update_sq_norm),
        "relative_total_update_norm": (
            math.sqrt(total_update_sq_norm) / math.sqrt(total_param_sq_norm_pre)
            if total_param_sq_norm_pre > 0.0
            else float("nan")
        ),
    }


def resolve_active_training_interventions(
    manifest: RunManifest,
    epoch: int,
) -> list[dict[str, object]]:
    active: list[dict[str, object]] = []
    for intervention in manifest.training_interventions:
        if intervention.epoch_start <= epoch <= intervention.epoch_end:
            active.append(
                {
                    "name": intervention.name,
                    "kind": intervention.kind,
                    "layer_index": intervention.layer_index,
                    "head_index": intervention.head_index,
                    "scale": intervention.scale,
                    "position": intervention.position,
                }
            )
    return active


def _group_rows_by_prompt_length(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Training/eval row is missing a prompt: {row!r}")
        grouped.setdefault(len(prompt.split()), []).append(row)
    return grouped


def _encode_batch(rows: list[dict[str, Any]], token_to_id: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    prompts: list[list[int]] = []
    target_ids: list[int] = []
    for row in rows:
        prompt = row.get("prompt")
        target = row.get("target")
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise ValueError(f"Row must contain string prompt/target fields, got {row!r}")
        token_ids = []
        for token in prompt.split():
            if token not in token_to_id:
                raise ValueError(f"Unknown prompt token {token!r} in row {row!r}")
            token_ids.append(token_to_id[token])
        if target not in token_to_id:
            raise ValueError(f"Unknown target token {target!r} in row {row!r}")
        prompts.append(token_ids)
        target_ids.append(token_to_id[target])
    return (
        torch.tensor(prompts, dtype=torch.long, device=device),
        torch.tensor(target_ids, dtype=torch.long, device=device),
    )


def evaluate_next_token_rows(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Expected at least one row when evaluating next-token rows")

    grouped = _group_rows_by_prompt_length(rows)
    total_rows = 0
    total_loss = 0.0
    total_correct = 0
    total_margin = 0.0
    prediction_rows: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for prompt_length in sorted(grouped):
            split_rows = grouped[prompt_length]
            for start in range(0, len(split_rows), batch_size):
                batch_rows = split_rows[start:start + batch_size]
                input_ids, target_ids = _encode_batch(batch_rows, token_to_id, device)
                logits = model(input_ids)
                final_logits = logits[:, -1, :]
                losses = F.cross_entropy(final_logits, target_ids, reduction="none")
                predicted_ids = final_logits.argmax(dim=-1)
                target_logits = final_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                competing_logits = final_logits.clone()
                competing_logits.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
                foil_logits, foil_ids = competing_logits.max(dim=-1)

                total_rows += len(batch_rows)
                total_loss += float(losses.sum().item())
                total_correct += int((predicted_ids == target_ids).sum().item())
                total_margin += float((target_logits - foil_logits).sum().item())

                for row_index, row in enumerate(batch_rows):
                    prediction_rows.append(
                        {
                            "prompt_id": str(row.get("id") or row.get("prompt_id") or row["prompt"]),
                            "predicted_token": id_to_token[int(predicted_ids[row_index].item())],
                            "target_token": row["target"],
                            "foil_token": id_to_token[int(foil_ids[row_index].item())],
                            "margin": float((target_logits[row_index] - foil_logits[row_index]).item()),
                            "correct": bool(predicted_ids[row_index].item() == target_ids[row_index].item()),
                        }
                    )
    return {
        "rows": total_rows,
        "loss": total_loss / total_rows,
        "accuracy": total_correct / total_rows,
        "margin": total_margin / total_rows,
        "predictions": prediction_rows,
    }


def build_epoch_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    grouped = _group_rows_by_prompt_length(rows)
    rng = random.Random(seed)
    lengths = sorted(grouped)
    rng.shuffle(lengths)

    batches: list[list[dict[str, Any]]] = []
    for prompt_length in lengths:
        group_rows = list(grouped[prompt_length])
        rng.shuffle(group_rows)
        for start in range(0, len(group_rows), batch_size):
            batches.append(group_rows[start:start + batch_size])
    rng.shuffle(batches)
    return batches


def train_next_token_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    *,
    manifest: RunManifest,
    token_to_id: dict[str, int],
    device: torch.device,
    batch_size: int,
    epoch: int,
    history_path: Path,
    formation_context: Any | None = None,
    bundle: Any | None = None,
    global_step_start: int,
    batch_seed: int,
    curriculum_stage: str,
    show_progress: bool = False,
) -> dict[str, Any]:
    batches = build_epoch_batches(rows, batch_size=batch_size, seed=batch_seed)
    if not batches:
        raise ValueError("Expected at least one batch in train_next_token_epoch")

    model.train()
    total_rows = 0
    total_loss = 0.0
    global_step = global_step_start
    active_interventions = resolve_active_training_interventions(manifest, epoch)
    batch_iterator = tqdm(batches, desc=f"epoch {epoch}", leave=False) if show_progress else batches
    for batch_index, batch_rows in enumerate(batch_iterator):
        input_ids, target_ids = _encode_batch(batch_rows, token_to_id, device)
        optimizer.zero_grad()
        if active_interventions:
            logits = forward_tiny_decoder_with_interventions(model, input_ids, active_interventions)
        else:
            logits = model(input_ids)
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, target_ids)
        loss.backward()
        pre_step_snapshot = snapshot_tracked_parameter_groups(model)
        optimizer.step()

        batch_loss = float(loss.item())
        batch_rows_count = len(batch_rows)
        total_rows += batch_rows_count
        total_loss += batch_loss * batch_rows_count
        step_dynamics = build_step_dynamics_record(model, pre_step_snapshot)
        if formation_context is not None:
            if bundle is None:
                raise ValueError("Formation logging requires the dataset bundle")
            from research.phase3.scripts.kv_formation_dynamics import maybe_record_formation_step

            formation_context = maybe_record_formation_step(
                context=formation_context,
                manifest=manifest,
                model=model,
                bundle=bundle,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                global_step=global_step,
                curriculum_stage=curriculum_stage,
                pre_step_snapshot=pre_step_snapshot,
            )
        append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "batch_index": batch_index,
                "batch_rows": batch_rows_count,
                "batch_loss": batch_loss,
                "curriculum_stage": curriculum_stage,
                "active_interventions": active_interventions,
                "total_grad_norm": step_dynamics["total_grad_norm"],
                "total_param_norm_pre": step_dynamics["total_param_norm_pre"],
                "total_param_norm_post": step_dynamics["total_param_norm_post"],
                "total_update_norm": step_dynamics["total_update_norm"],
                "relative_total_update_norm": step_dynamics["relative_total_update_norm"],
                "parameter_metrics": step_dynamics["parameter_metrics"],
            },
        )
        if show_progress:
            batch_iterator.set_postfix(
                loss=f"{batch_loss:.4f}",
                grad=f"{step_dynamics['total_grad_norm']:.3f}",
                update=f"{step_dynamics['relative_total_update_norm']:.4f}",
            )
        global_step += 1
    if show_progress:
        batch_iterator.close()
    return {
        "epoch": epoch,
        "global_step_end": global_step,
        "train_loss": total_loss / total_rows,
        "train_rows": total_rows,
        "formation_context": formation_context,
    }


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open() as handle:
        return json.load(handle)


def build_training_intervention_signature(
    interventions: list[TrainingInterventionConfig],
) -> str:
    if not interventions:
        return "none"
    normalized = [
        {
            "name": intervention.name,
            "kind": intervention.kind,
            "layer_index": intervention.layer_index,
            "head_index": intervention.head_index,
            "epoch_start": intervention.epoch_start,
            "epoch_end": intervention.epoch_end,
            "scale": intervention.scale,
            "position": intervention.position,
        }
        for intervention in interventions
    ]
    normalized.sort(
        key=lambda item: (
            str(item["name"]),
            str(item["kind"]),
            int(item["layer_index"]),
            -1 if item["head_index"] is None else int(item["head_index"]),
            int(item["epoch_start"]),
            int(item["epoch_end"]),
            float(item["scale"]),
            str(item["position"]),
        )
    )
    return json.dumps(normalized, sort_keys=True)
