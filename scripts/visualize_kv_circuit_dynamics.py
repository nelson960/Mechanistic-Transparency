#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render artifact-first visualizations for a KV training-dynamics run. "
            "Requires a completed run directory with summaries and checkpoint batteries."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed KV run directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Empty output directory for plots and frames.")
    parser.add_argument(
        "--checkpoint-kind",
        choices=["scheduled", "all"],
        default="scheduled",
        help="Which checkpoint family to visualize in the frame sequence.",
    )
    parser.add_argument("--top-neurons", type=int, default=12, help="Top neurons to show in each frame.")
    parser.add_argument("--top-sites", type=int, default=10, help="Top feature sites to show in each frame.")
    parser.add_argument("--top-drift-sites", type=int, default=8, help="Top tracked sites in the drift plot.")
    parser.add_argument("--top-network-modules", type=int, default=10, help="How many parameter groups to show in the network-change plot.")
    parser.add_argument("--frame-duration-ms", type=int, default=450, help="Frame duration for the exported GIF.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def assert_output_dir_is_empty(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    contents = list(out_dir.iterdir())
    if contents:
        raise ValueError(
            "Refusing to overwrite a non-empty visualization directory: "
            + ", ".join(str(path) for path in contents[:10])
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Unexpected blank line in {path} at line {line_number}")
            rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"JSONL file contains zero rows: {path}")
    return rows


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV artifact: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV artifact contains zero rows: {path}")
    return frame


def collect_required_paths(run_dir: Path) -> None:
    required_paths = [
        run_dir / "train_history.jsonl",
        run_dir / "summaries" / "checkpoint_index.csv",
        run_dir / "summaries" / "neuron_dynamics.csv",
        run_dir / "summaries" / "feature_dynamics.csv",
        run_dir / "summaries" / "superposition_dynamics.csv",
        run_dir / "summaries" / "representation_drift.csv",
        run_dir / "summaries" / "operator_handoffs.csv",
        run_dir / "summaries" / "run_summary.json",
        run_dir / "battery",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")


def load_checkpoint_index(run_dir: Path, checkpoint_kind: str) -> pd.DataFrame:
    checkpoint_index = load_csv(run_dir / "summaries" / "checkpoint_index.csv")
    if checkpoint_kind == "scheduled":
        checkpoint_index = checkpoint_index[checkpoint_index["checkpoint_id"].astype(str).str.startswith("scheduled_epoch_")].copy()
    checkpoint_index = checkpoint_index.sort_values(["epoch", "checkpoint_id"]).reset_index(drop=True)
    if checkpoint_index.empty:
        raise ValueError(f"No checkpoints remain after filtering with checkpoint_kind={checkpoint_kind!r}")
    return checkpoint_index


def collect_variable_names(checkpoint_index: pd.DataFrame) -> list[str]:
    variables: list[str] = []
    for column in checkpoint_index.columns:
        if column.endswith("_site") and not column.endswith("_faithfulness_site"):
            variables.append(column[:-5])
    if not variables:
        raise ValueError("Could not infer tracked variables from checkpoint_index.csv")
    return sorted(variables)


def head_sort_key(head_name: str) -> tuple[int, int]:
    block_part, head_part = head_name.split("_")
    return int(block_part.removeprefix("block")), int(head_part.removeprefix("head"))


def collect_head_names(run_dir: Path, checkpoint_ids: list[str]) -> list[str]:
    head_names = set()
    for checkpoint_id in checkpoint_ids:
        table = load_csv(run_dir / "battery" / checkpoint_id / "operator_scores.csv")
        head_rows = table[table["row_kind"] == "head_score"].copy()
        head_names.update(head_rows["head_name"].astype(str).tolist())
    if not head_names:
        raise ValueError("No head_score rows found across the selected checkpoints")
    return sorted(head_names, key=head_sort_key)


def load_operator_history(run_dir: Path, checkpoint_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for checkpoint_row in checkpoint_index.itertuples(index=False):
        checkpoint_id = str(checkpoint_row.checkpoint_id)
        table = load_csv(run_dir / "battery" / checkpoint_id / "operator_scores.csv")
        head_rows = table[table["row_kind"] == "head_score"].copy()
        head_rows["checkpoint_id"] = checkpoint_id
        head_rows["epoch"] = int(checkpoint_row.epoch)
        rows.append(head_rows)
    return pd.concat(rows, ignore_index=True)


def load_localization_history(run_dir: Path, checkpoint_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for checkpoint_row in checkpoint_index.itertuples(index=False):
        checkpoint_id = str(checkpoint_row.checkpoint_id)
        table = load_csv(run_dir / "battery" / checkpoint_id / "localization.csv")
        table = table[table["row_kind"] == "head_ablation"].copy()
        table["checkpoint_id"] = checkpoint_id
        table["epoch"] = int(checkpoint_row.epoch)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _configure_axis_grid(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def _rolling_series(values: pd.Series) -> pd.Series:
    window = max(1, min(64, len(values) // 20))
    return values.rolling(window=window, min_periods=1).mean()


def plot_training_dynamics(train_history: pd.DataFrame, out_path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    x = train_history["global_step"]

    axes[0].plot(x, train_history["batch_loss"], alpha=0.25, linewidth=0.8, label="batch loss")
    axes[0].plot(x, _rolling_series(train_history["batch_loss"]), linewidth=2.0, label="rolling mean")
    _configure_axis_grid(axes[0], "Per-Step Loss", "cross-entropy")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, train_history["total_grad_norm"], linewidth=1.2, label="grad norm")
    axes[1].plot(x, _rolling_series(train_history["total_grad_norm"]), linewidth=2.0, label="rolling mean")
    _configure_axis_grid(axes[1], "Gradient Norm", "L2 norm")
    axes[1].legend(loc="upper right")

    axes[2].plot(x, train_history["relative_total_update_norm"], linewidth=1.2, label="relative update")
    axes[2].plot(x, _rolling_series(train_history["relative_total_update_norm"]), linewidth=2.0, label="rolling mean")
    _configure_axis_grid(axes[2], "Relative Update Size", "update / param")
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Global Step")

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def explode_parameter_metrics(train_history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for history_row in train_history.itertuples(index=False):
        parameter_metrics = getattr(history_row, "parameter_metrics")
        if not isinstance(parameter_metrics, list):
            raise ValueError("Expected train_history.parameter_metrics to be a list of dict rows")
        for metric in parameter_metrics:
            if not isinstance(metric, dict):
                raise ValueError("Expected each parameter_metrics entry to be a dict")
            layer_index = metric.get("layer_index")
            component = str(metric.get("component"))
            if layer_index is None:
                module_label = component
            else:
                module_label = f"block{int(layer_index) + 1}.{component}"
            rows.append(
                {
                    "epoch": int(history_row.epoch),
                    "global_step": int(history_row.global_step),
                    "module_label": module_label,
                    "component": component,
                    "head_index": metric.get("head_index"),
                    "relative_update_norm": float(metric["relative_update_norm"]),
                    "grad_norm": float(metric["grad_norm"]),
                    "update_norm": float(metric["update_norm"]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Exploded parameter metrics produced zero rows")
    return table


def plot_network_change(parameter_metrics: pd.DataFrame, out_path: Path, dpi: int, top_modules: int) -> None:
    whole_module_metrics = parameter_metrics[parameter_metrics["head_index"].isna()].copy()
    module_epoch = (
        whole_module_metrics.groupby(["epoch", "module_label"], as_index=False)[
            ["relative_update_norm", "grad_norm", "update_norm"]
        ]
        .mean()
    )
    module_rank = (
        module_epoch.groupby("module_label", as_index=False)["relative_update_norm"]
        .max()
        .sort_values("relative_update_norm", ascending=False)
    )
    selected_modules = module_rank["module_label"].head(top_modules).tolist()
    if not selected_modules:
        raise ValueError("Could not select any modules for the network-change plot")

    figure, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    for module_label in selected_modules:
        module_rows = module_epoch[module_epoch["module_label"] == module_label].sort_values("epoch")
        axes[0].plot(module_rows["epoch"], module_rows["relative_update_norm"], linewidth=1.7, label=module_label)
        axes[1].plot(module_rows["epoch"], module_rows["grad_norm"], linewidth=1.7, label=module_label)
        axes[2].plot(module_rows["epoch"], module_rows["update_norm"], linewidth=1.7, label=module_label)

    _configure_axis_grid(axes[0], "Network Change by Module", "relative update norm")
    _configure_axis_grid(axes[1], "Gradient Flow by Module", "grad norm")
    _configure_axis_grid(axes[2], "Absolute Update by Module", "update norm")
    axes[2].set_xlabel("Epoch")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[2].legend(loc="upper right", fontsize=8, ncol=2)

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def plot_behavior_and_variables(
    checkpoint_index: pd.DataFrame,
    variables: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    epochs = checkpoint_index["epoch"]

    axes[0].plot(epochs, checkpoint_index["val_accuracy"], linewidth=2.0, label="val")
    axes[0].plot(epochs, checkpoint_index["test_accuracy"], linewidth=2.0, label="test")
    axes[0].plot(epochs, checkpoint_index["ood_accuracy"], linewidth=2.0, label="ood")
    _configure_axis_grid(axes[0], "Behavior Across Checkpoints", "accuracy")
    axes[0].legend(loc="upper left")

    for variable in variables:
        axes[1].plot(epochs, checkpoint_index[f"{variable}_score"], linewidth=1.8, label=variable)
    _configure_axis_grid(axes[1], "Variable Recovery", "probe score")
    axes[1].legend(loc="upper left", ncol=2, fontsize=9)

    for variable in variables:
        axes[2].plot(
            epochs,
            checkpoint_index[f"{variable}_faithfulness_score"],
            linewidth=1.8,
            label=variable,
        )
    _configure_axis_grid(axes[2], "Variable Faithfulness", "patch score")
    axes[2].legend(loc="upper left", ncol=2, fontsize=9)

    axes[3].plot(epochs, checkpoint_index["routing_score"], linewidth=2.0, label="routing")
    axes[3].plot(epochs, checkpoint_index["copy_score"], linewidth=2.0, label="copy")
    _configure_axis_grid(axes[3], "Operator Scores", "score")
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Epoch")

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def select_focus_heads(checkpoint_index: pd.DataFrame, localization_history: pd.DataFrame) -> list[str]:
    final_epoch = int(checkpoint_index["epoch"].max())
    final_checkpoint_rows = checkpoint_index[checkpoint_index["epoch"] == final_epoch].copy()
    if final_checkpoint_rows.empty:
        raise ValueError("Could not find final checkpoint row")
    final_row = final_checkpoint_rows.iloc[-1]
    head_names = [str(final_row["routing_candidate"]), str(final_row["copy_candidate"])]
    final_localization = localization_history[localization_history["epoch"] == final_epoch].copy()
    top_ablation_heads = (
        final_localization.sort_values("accuracy_drop", ascending=False)["head_name"].astype(str).head(2).tolist()
    )
    focus_heads: list[str] = []
    for head_name in head_names + top_ablation_heads:
        if head_name and head_name not in focus_heads:
            focus_heads.append(head_name)
    return focus_heads


def plot_circuit_emergence(
    checkpoint_index: pd.DataFrame,
    operator_history: pd.DataFrame,
    localization_history: pd.DataFrame,
    variables: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    focus_heads = select_focus_heads(checkpoint_index, localization_history)
    figure, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    epochs = checkpoint_index["epoch"]

    axes[0].plot(epochs, checkpoint_index["val_accuracy"], linewidth=2.2, label="val")
    axes[0].plot(epochs, checkpoint_index["test_accuracy"], linewidth=2.2, label="test")
    axes[0].plot(epochs, checkpoint_index["ood_accuracy"], linewidth=2.2, label="ood")
    _configure_axis_grid(axes[0], "Behavior Emergence", "accuracy")
    axes[0].legend(loc="upper left")

    for head_name in focus_heads:
        head_rows = operator_history[operator_history["head_name"] == head_name].sort_values("epoch")
        axes[1].plot(head_rows["epoch"], head_rows["routing_score"], linewidth=1.8, label=f"{head_name} routing")
        axes[1].plot(
            head_rows["epoch"],
            head_rows["copy_score"],
            linewidth=1.4,
            linestyle="--",
            label=f"{head_name} copy",
        )
    _configure_axis_grid(axes[1], "Candidate Head Scores", "score")
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)

    for head_name in focus_heads:
        head_rows = localization_history[localization_history["head_name"] == head_name].sort_values("epoch")
        axes[2].plot(head_rows["epoch"], head_rows["accuracy_drop"], linewidth=1.8, label=head_name)
    _configure_axis_grid(axes[2], "Ablation Importance of Focus Heads", "accuracy drop")
    axes[2].legend(loc="upper left", fontsize=8, ncol=2)

    for variable in variables:
        axes[3].plot(epochs, checkpoint_index[f"{variable}_score"], linewidth=1.8, label=f"{variable} probe")
        axes[3].plot(
            epochs,
            checkpoint_index[f"{variable}_faithfulness_score"],
            linewidth=1.2,
            linestyle="--",
            label=f"{variable} faithfulness",
        )
    _configure_axis_grid(axes[3], "Variable Formation", "score")
    axes[3].legend(loc="upper left", fontsize=8, ncol=2)
    axes[3].set_xlabel("Epoch")

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def _pivot_history(table: pd.DataFrame, value_column: str, head_names: list[str], epochs: list[int]) -> pd.DataFrame:
    pivot = (
        table.pivot_table(index="head_name", columns="epoch", values=value_column, aggfunc="mean")
        .reindex(index=head_names, columns=epochs)
    )
    return pivot.fillna(float("nan"))


def _draw_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, cmap: str, value_label: str) -> None:
    image = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    x_ticks = list(range(len(matrix.columns)))
    if len(x_ticks) > 12:
        step = max(1, len(x_ticks) // 12)
        x_ticks = x_ticks[::step]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(matrix.columns[index])) for index in x_ticks], rotation=45, ha="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Head")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=value_label)


def plot_operator_and_localization_heatmaps(
    operator_history: pd.DataFrame,
    localization_history: pd.DataFrame,
    head_names: list[str],
    epochs: list[int],
    out_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    routing_matrix = _pivot_history(operator_history, "routing_score", head_names, epochs)
    copy_matrix = _pivot_history(operator_history, "copy_score", head_names, epochs)
    routing_key_matrix = _pivot_history(operator_history, "routing_key_score", head_names, epochs)
    localization_matrix = _pivot_history(localization_history, "accuracy_drop", head_names, epochs)

    _draw_heatmap(axes[0, 0], routing_matrix, "Routing Score by Head", "viridis", "routing score")
    _draw_heatmap(axes[0, 1], copy_matrix, "Copy Score by Head", "magma", "copy score")
    _draw_heatmap(axes[1, 0], routing_key_matrix, "Routing-Key Score by Head", "cividis", "routing-key score")
    _draw_heatmap(axes[1, 1], localization_matrix, "Ablation Accuracy Drop by Head", "plasma", "accuracy drop")

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def select_drift_sites(checkpoint_index: pd.DataFrame, drift_table: pd.DataFrame, max_sites: int) -> list[str]:
    site_counts: dict[str, int] = {}
    for column in checkpoint_index.columns:
        if column.endswith("_site"):
            for value in checkpoint_index[column].dropna().astype(str):
                if not value:
                    continue
                site_counts[value] = site_counts.get(value, 0) + 1
    available_sites = set(drift_table["site"].astype(str))
    selected = [site for site, _ in sorted(site_counts.items(), key=lambda item: (-item[1], item[0])) if site in available_sites]
    selected = selected[:max_sites]
    if not selected:
        raise ValueError("Could not choose any drift sites from checkpoint_index and representation_drift artifacts")
    return selected


def plot_representation_drift(
    checkpoint_index: pd.DataFrame,
    drift_table: pd.DataFrame,
    out_path: Path,
    dpi: int,
    max_sites: int,
) -> None:
    origin_rows = drift_table[drift_table["is_origin_pair"] == True].copy()
    if origin_rows.empty:
        raise ValueError("Representation drift table contains no origin-pair rows")
    selected_sites = select_drift_sites(checkpoint_index, origin_rows, max_sites)
    figure, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    for site in selected_sites:
        site_rows = origin_rows[origin_rows["site"] == site].sort_values("epoch_right")
        axes[0].plot(site_rows["epoch_right"], site_rows["linear_cka"], linewidth=1.8, label=site)
        axes[1].plot(site_rows["epoch_right"], site_rows["relative_frobenius_shift"], linewidth=1.8, label=site)
    _configure_axis_grid(axes[0], "Origin-to-Checkpoint CKA", "linear CKA")
    _configure_axis_grid(axes[1], "Origin-to-Checkpoint Relative Shift", "relative Frobenius shift")
    axes[1].set_xlabel("Epoch")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def plot_neuron_feature_superposition(
    neuron_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    superposition_table: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    layer_top = neuron_table.copy()
    if "layer_name" not in layer_top.columns:
        layer_top = layer_top.copy()
        layer_top["layer_name"] = layer_top["layer_index"].map(lambda idx: f"block{int(idx) + 1}")
    site_summary = feature_table.copy()
    if layer_top.empty:
        raise ValueError("Neuron dynamics table contains zero rows")
    if site_summary.empty:
        raise ValueError("Feature dynamics table contains zero rows")

    figure, axes = plt.subplots(3, 1, figsize=(15, 16), sharex=True)
    neuron_summary = (
        layer_top.groupby(["epoch", "layer_name"], as_index=False)["best_selectivity_score"]
        .mean()
        .sort_values(["layer_name", "epoch"])
    )
    for layer_name, group in neuron_summary.groupby("layer_name"):
        axes[0].plot(group["epoch"], group["best_selectivity_score"], linewidth=1.8, label=layer_name)
    _configure_axis_grid(axes[0], "Top-Neuron Selectivity by Layer", "mean selectivity")
    axes[0].legend(loc="upper left")

    site_summary = site_summary.sort_values(["site", "epoch"])
    for site, group in site_summary.groupby("site"):
        axes[1].plot(group["epoch"], group["top_feature_selectivity_score"], linewidth=1.2, alpha=0.8, label=site)
    _configure_axis_grid(axes[1], "Top SAE Feature Selectivity by Site", "selectivity")
    if site_summary["site"].nunique() <= 12:
        axes[1].legend(loc="upper left", fontsize=8, ncol=2)

    for site, group in superposition_table.groupby("site"):
        axes[2].plot(group["epoch"], group["decoder_overlap_fraction"], linewidth=1.2, alpha=0.8, label=site)
    _configure_axis_grid(axes[2], "Decoder Overlap Fraction by Site", "overlap fraction")
    axes[2].set_xlabel("Epoch")
    if superposition_table["site"].nunique() <= 12:
        axes[2].legend(loc="upper left", fontsize=8, ncol=2)

    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def _barh_with_labels(ax: plt.Axes, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    positions = list(range(len(labels)))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    ax.invert_yaxis()


def render_checkpoint_frames(
    *,
    checkpoint_index: pd.DataFrame,
    operator_history: pd.DataFrame,
    localization_history: pd.DataFrame,
    neuron_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    out_dir: Path,
    variables: list[str],
    top_neurons: int,
    top_sites: int,
    dpi: int,
) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, Any]] = []

    frame_iterator = tqdm(list(checkpoint_index.itertuples(index=False)), desc="render frames", leave=True)
    for frame_index, checkpoint_row in enumerate(frame_iterator):
        checkpoint_id = str(checkpoint_row.checkpoint_id)
        current_epoch = int(checkpoint_row.epoch)
        history_df = checkpoint_index[checkpoint_index["epoch"] <= current_epoch].copy()
        current_operator = operator_history[operator_history["checkpoint_id"] == checkpoint_id].copy()
        current_localization = localization_history[localization_history["checkpoint_id"] == checkpoint_id].copy()
        current_neurons = neuron_table[neuron_table["checkpoint_id"] == checkpoint_id].copy()
        current_features = feature_table[feature_table["checkpoint_id"] == checkpoint_id].copy()

        if current_operator.empty:
            raise ValueError(f"No operator rows found for checkpoint {checkpoint_id}")
        if current_localization.empty:
            raise ValueError(f"No localization rows found for checkpoint {checkpoint_id}")
        if current_neurons.empty:
            raise ValueError(f"No neuron rows found for checkpoint {checkpoint_id}")
        if current_features.empty:
            raise ValueError(f"No feature rows found for checkpoint {checkpoint_id}")
        if "component" not in current_neurons.columns:
            current_neurons = current_neurons.copy()
            current_neurons["component"] = current_neurons.apply(
                lambda row: f"block{int(row['layer_index']) + 1}.mlp.neuron_{int(row['neuron_index'])}",
                axis=1,
            )

        figure, axes = plt.subplots(3, 2, figsize=(18, 16))
        figure.suptitle(f"{checkpoint_id} | epoch {current_epoch}", fontsize=16)

        axes[0, 0].plot(history_df["epoch"], history_df["val_accuracy"], linewidth=2.0, label="val")
        axes[0, 0].plot(history_df["epoch"], history_df["test_accuracy"], linewidth=2.0, label="test")
        axes[0, 0].plot(history_df["epoch"], history_df["ood_accuracy"], linewidth=2.0, label="ood")
        axes[0, 0].axvline(current_epoch, color="black", linewidth=1.0, alpha=0.6)
        _configure_axis_grid(axes[0, 0], "Behavior to Current Checkpoint", "accuracy")
        axes[0, 0].legend(loc="upper left")

        for variable in variables:
            axes[0, 1].plot(history_df["epoch"], history_df[f"{variable}_score"], linewidth=1.5, label=f"{variable} probe")
            axes[0, 1].plot(
                history_df["epoch"],
                history_df[f"{variable}_faithfulness_score"],
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
                label=f"{variable} faithfulness",
            )
        axes[0, 1].axvline(current_epoch, color="black", linewidth=1.0, alpha=0.6)
        _configure_axis_grid(axes[0, 1], "Variable Recovery and Faithfulness", "score")
        axes[0, 1].legend(loc="upper left", fontsize=8, ncol=2)

        operator_rows = current_operator.sort_values(["layer_index", "head_index"]).reset_index(drop=True)
        operator_labels = operator_rows["head_name"].astype(str).tolist()
        routing_values = operator_rows["routing_score"].astype(float).tolist()
        copy_values = operator_rows["copy_score"].astype(float).tolist()
        positions = list(range(len(operator_labels)))
        width = 0.38
        axes[1, 0].bar([position - width / 2 for position in positions], routing_values, width=width, label="routing")
        axes[1, 0].bar([position + width / 2 for position in positions], copy_values, width=width, label="copy")
        axes[1, 0].set_xticks(positions)
        axes[1, 0].set_xticklabels(operator_labels, rotation=25, ha="right")
        axes[1, 0].set_title("Current Head Operator Scores")
        axes[1, 0].set_ylabel("score")
        axes[1, 0].grid(True, axis="y", alpha=0.25, linewidth=0.6)
        axes[1, 0].legend(loc="upper left")

        localization_rows = current_localization.sort_values(["layer_index", "head_index"]).reset_index(drop=True)
        _barh_with_labels(
            axes[1, 1],
            localization_rows["head_name"].astype(str).tolist(),
            localization_rows["accuracy_drop"].astype(float).tolist(),
            "Current Head Ablation Effect",
            "accuracy drop",
        )

        top_neuron_rows = current_neurons.sort_values(
            ["best_selectivity_score", "activation_write_product"],
            ascending=[False, False],
        ).head(top_neurons)
        neuron_labels = [
            f"{row.component} [{row.best_variable}]"
            for row in top_neuron_rows.itertuples(index=False)
        ]
        _barh_with_labels(
            axes[2, 0],
            neuron_labels,
            top_neuron_rows["best_selectivity_score"].astype(float).tolist(),
            "Top Neurons at Current Checkpoint",
            "best selectivity",
        )

        top_feature_rows = current_features.sort_values(
            ["top_feature_selectivity_score", "mean_top_feature_selectivity_score"],
            ascending=[False, False],
        ).head(top_sites)
        feature_labels = [
            f"{row.site} [{row.top_feature_variable}]"
            for row in top_feature_rows.itertuples(index=False)
        ]
        _barh_with_labels(
            axes[2, 1],
            feature_labels,
            top_feature_rows["top_feature_selectivity_score"].astype(float).tolist(),
            "Top SAE Sites at Current Checkpoint",
            "top feature selectivity",
        )

        figure.tight_layout(rect=[0, 0.02, 1, 0.97])
        frame_path = frames_dir / f"frame_{frame_index:04d}_{checkpoint_id}.png"
        figure.savefig(frame_path, dpi=dpi)
        plt.close(figure)

        manifest_rows.append(
            {
                "frame_index": frame_index,
                "checkpoint_id": checkpoint_id,
                "epoch": current_epoch,
                "frame_path": str(frame_path),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(out_dir / "frame_manifest.csv", index=False)


def write_readme(*, out_dir: Path, run_dir: Path, checkpoint_kind: str, checkpoint_count: int) -> None:
    lines = [
        f"Run directory: {run_dir}",
        f"Checkpoint selection: {checkpoint_kind}",
        f"Visualized checkpoints: {checkpoint_count}",
        "",
        "Files:",
        "training_dynamics.png",
        "network_change.png",
        "behavior_and_variables.png",
        "circuit_emergence.png",
        "circuit_emergence.gif",
        "operator_and_localization_heatmaps.png",
        "representation_drift.png",
        "neuron_feature_superposition.png",
        "frames/",
        "frame_manifest.csv",
        "",
        "Interpretation order:",
        "1. training_dynamics.png",
        "2. network_change.png",
        "3. circuit_emergence.gif",
        "4. operator_and_localization_heatmaps.png",
        "5. representation_drift.png",
        "6. neuron_feature_superposition.png",
        "7. frames/",
    ]
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_frame_gif(*, frames_dir: Path, out_path: Path, duration_ms: int) -> None:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise ValueError(f"No frame PNG files found under {frames_dir}")
    from PIL import Image

    images = [Image.open(frame_path).convert("P", palette=Image.ADAPTIVE) for frame_path in frame_paths]
    first_image, remaining_images = images[0], images[1:]
    first_image.save(
        out_path,
        save_all=True,
        append_images=remaining_images,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    collect_required_paths(run_dir)
    assert_output_dir_is_empty(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_index = load_checkpoint_index(run_dir, args.checkpoint_kind)
    variables = collect_variable_names(checkpoint_index)
    checkpoint_ids = checkpoint_index["checkpoint_id"].astype(str).tolist()
    epochs = checkpoint_index["epoch"].astype(int).tolist()

    train_history = pd.DataFrame(read_jsonl(run_dir / "train_history.jsonl"))
    if train_history.empty:
        raise ValueError("train_history.jsonl produced zero rows")
    parameter_metrics = explode_parameter_metrics(train_history)

    operator_history = load_operator_history(run_dir, checkpoint_index)
    localization_history = load_localization_history(run_dir, checkpoint_index)
    neuron_table = load_csv(run_dir / "summaries" / "neuron_dynamics.csv")
    feature_table = load_csv(run_dir / "summaries" / "feature_dynamics.csv")
    superposition_table = load_csv(run_dir / "summaries" / "superposition_dynamics.csv")
    drift_table = load_csv(run_dir / "summaries" / "representation_drift.csv")
    head_names = collect_head_names(run_dir, checkpoint_ids)

    plot_training_dynamics(train_history, out_dir / "training_dynamics.png", args.dpi)
    plot_network_change(parameter_metrics, out_dir / "network_change.png", args.dpi, args.top_network_modules)
    plot_behavior_and_variables(checkpoint_index, variables, out_dir / "behavior_and_variables.png", args.dpi)
    plot_circuit_emergence(
        checkpoint_index,
        operator_history,
        localization_history,
        variables,
        out_dir / "circuit_emergence.png",
        args.dpi,
    )
    plot_operator_and_localization_heatmaps(
        operator_history,
        localization_history,
        head_names,
        epochs,
        out_dir / "operator_and_localization_heatmaps.png",
        args.dpi,
    )
    plot_representation_drift(
        checkpoint_index,
        drift_table,
        out_dir / "representation_drift.png",
        args.dpi,
        args.top_drift_sites,
    )
    plot_neuron_feature_superposition(
        neuron_table,
        feature_table,
        superposition_table,
        out_dir / "neuron_feature_superposition.png",
        args.dpi,
    )
    render_checkpoint_frames(
        checkpoint_index=checkpoint_index,
        operator_history=operator_history,
        localization_history=localization_history,
        neuron_table=neuron_table,
        feature_table=feature_table,
        out_dir=out_dir,
        variables=variables,
        top_neurons=args.top_neurons,
        top_sites=args.top_sites,
        dpi=args.dpi,
    )
    create_frame_gif(
        frames_dir=out_dir / "frames",
        out_path=out_dir / "circuit_emergence.gif",
        duration_ms=args.frame_duration_ms,
    )
    write_readme(
        out_dir=out_dir,
        run_dir=run_dir,
        checkpoint_kind=args.checkpoint_kind,
        checkpoint_count=len(checkpoint_index),
    )
    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
