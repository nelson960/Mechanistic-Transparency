from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_LABELS = {
    "curriculum_on": "Curriculum on",
    "curriculum_off": "Curriculum off",
}

METRIC_LABELS = {
    "val_accuracy": "Val",
    "test_accuracy": "Test",
    "ood_accuracy": "OOD",
    "query_key_score": "Query",
    "matching_slot_score": "Match",
    "selected_value_score": "Selected",
    "routing_score": "Routing",
    "copy_score": "Copy",
}

EMERGENCE_LABELS = {
    "behavior_val_accuracy": "Behavior",
    "operator_copy": "Copy",
    "operator_routing": "Routing",
    "faithfulness_matching_slot": "Match faith.",
}

HEAD_ORDER = ["block1_head0", "block1_head1", "block2_head0", "block2_head1"]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def _condition_paths(matrix_dir: Path) -> dict[str, Path]:
    conditions = {
        "curriculum_on": matrix_dir / "curriculum_on",
        "curriculum_off": matrix_dir / "curriculum_off",
    }
    for condition, path in conditions.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing condition directory for {condition}: {path}")
    return conditions


def _render_performance_figure(matrix_dir: Path, out_path: Path) -> None:
    conditions = _condition_paths(matrix_dir)
    score_metrics = [
        "query_key_score",
        "matching_slot_score",
        "selected_value_score",
        "routing_score",
        "copy_score",
    ]
    acc_metrics = ["val_accuracy", "test_accuracy", "ood_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"curriculum_on": "#B6513D", "curriculum_off": "#1F5A91"}
    width = 0.36

    for ax, metrics, title in [
        (axes[0], acc_metrics, "Held-out behavior"),
        (axes[1], score_metrics, "Internal metric means"),
    ]:
        x = np.arange(len(metrics))
        for offset, condition in zip([-width / 2, width / 2], conditions.keys(), strict=True):
            stability = _load_csv(conditions[condition] / "seed_stability.csv")
            rows = stability.set_index("metric_name").loc[metrics]
            ax.bar(
                x + offset,
                rows["mean"].to_numpy(),
                width,
                yerr=rows["std"].to_numpy(),
                capsize=4,
                color=colors[condition],
                alpha=0.9,
                label=CONDITION_LABELS[condition],
            )
        ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics], rotation=0)
        ax.set_ylim(0.0, 1.08)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Full matrix: grouped outcome comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _seed_from_run_id(run_id: str) -> int:
    match = re.search(r"seed(\d+)", run_id)
    if not match:
        raise ValueError(f"Could not parse seed from run_id: {run_id}")
    return int(match.group(1))


def _render_emergence_figure(matrix_dir: Path, out_path: Path) -> None:
    conditions = _condition_paths(matrix_dir)
    metric_order = list(EMERGENCE_LABELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#E6E6E6")

    max_epoch = 0.0
    matrices = {}
    for condition, path in conditions.items():
        emergence = _load_csv(path / "emergence.csv")
        emergence["seed"] = emergence["run_id"].map(_seed_from_run_id)
        pivot = (
            emergence.pivot(index="seed", columns="metric_name", values="birth_epoch")
            .reindex(index=[0, 1, 2], columns=metric_order)
        )
        matrices[condition] = pivot
        with np.errstate(all="ignore"):
            current_max = np.nanmax(pivot.to_numpy())
        if not math.isnan(current_max):
            max_epoch = max(max_epoch, float(current_max))

    for ax, condition in zip(axes, conditions.keys(), strict=True):
        matrix = matrices[condition]
        im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0, vmax=max_epoch)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xticks(np.arange(len(metric_order)), [EMERGENCE_LABELS[m] for m in metric_order], rotation=20, ha="right")
        ax.set_yticks(np.arange(3), [f"Seed {seed}" for seed in matrix.index])
        for row_index, seed in enumerate(matrix.index):
            for col_index, metric_name in enumerate(metric_order):
                value = matrix.loc[seed, metric_name]
                label = "—" if pd.isna(value) else str(int(value))
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Birth epoch")
    fig.suptitle("Full matrix: emergence timing", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_localization_figure(matrix_dir: Path, out_path: Path) -> None:
    localization_dir = matrix_dir / "localization"
    if not localization_dir.exists():
        raise FileNotFoundError(f"Missing localization directory: {localization_dir}")

    rows = []
    for condition_prefix in ["on", "off"]:
        for seed in [0, 1, 2]:
            path = localization_dir / f"{condition_prefix}_seed{seed}_localization.csv"
            table = _load_csv(path)
            head_rows = table[table["row_kind"] == "head_ablation"].copy()
            head_rows = head_rows.set_index("head_name").reindex(HEAD_ORDER)
            rows.append(
                {
                    "label": f"{condition_prefix}{seed}",
                    **{head: float(head_rows.loc[head, "accuracy_drop"]) for head in HEAD_ORDER},
                }
            )

    matrix = pd.DataFrame(rows).set_index("label")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0.0, vmax=float(np.nanmax(matrix.to_numpy())))
    ax.set_xticks(np.arange(len(HEAD_ORDER)), HEAD_ORDER, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row_index, row_label in enumerate(matrix.index):
        for col_index, head_name in enumerate(HEAD_ORDER):
            value = matrix.loc[row_label, head_name]
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Accuracy drop under head ablation")
    ax.set_title("Full matrix: final localization by seed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mean_role_cosine(role_matching: pd.DataFrame, role_name: str) -> float:
    rows = role_matching[role_matching["role_name"] == role_name]
    return float(rows["score_profile_cosine"].mean())


def _mean_handoffs(operator_handoffs: pd.DataFrame, role_name: str) -> float:
    rows = operator_handoffs[operator_handoffs["role_name"] == role_name].copy()
    changed = rows["candidate_changed"].astype(str).str.lower().eq("true")
    rows["candidate_changed_bool"] = changed
    per_run = rows.groupby("run_id", sort=False)["candidate_changed_bool"].sum()
    return float(per_run.mean())


def _render_stability_figure(matrix_dir: Path, out_path: Path) -> None:
    conditions = _condition_paths(matrix_dir)

    role_data = []
    handoff_data = []
    birth_data = []

    for condition, path in conditions.items():
        role_matching = _load_csv(path / "role_matching.csv")
        operator_handoffs = _load_csv(path / "operator_handoffs.csv")
        summary = _load_json(path / "run_summary.json")

        role_data.append(
            {
                "condition": CONDITION_LABELS[condition],
                "Routing": _mean_role_cosine(role_matching, "routing"),
                "Copy": _mean_role_cosine(role_matching, "copy"),
            }
        )
        handoff_data.append(
            {
                "condition": CONDITION_LABELS[condition],
                "Routing": _mean_handoffs(operator_handoffs, "routing"),
                "Copy": _mean_handoffs(operator_handoffs, "copy"),
            }
        )
        birth_data.append(
            {
                "condition": CONDITION_LABELS[condition],
                "Match faith. births": summary["matching_slot_faithfulness_birth_runs"] / summary["num_runs"],
                "Routing births": summary["routing_birth_runs"] / summary["num_runs"],
                "Routing before copy": summary["routing_before_copy_runs"] / summary["num_runs"],
            }
        )

    role_df = pd.DataFrame(role_data).set_index("condition")
    handoff_df = pd.DataFrame(handoff_data).set_index("condition")
    birth_df = pd.DataFrame(birth_data).set_index("condition")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    colors = ["#B6513D", "#1F5A91"]

    for ax, df, title, ylabel, ylim in [
        (axes[0], role_df, "Role-profile cosine", "Mean cosine", (0.0, 1.02)),
        (axes[1], handoff_df, "Mean operator handoffs", "Candidate changes", (0.0, max(1.0, handoff_df.to_numpy().max() * 1.15))),
        (axes[2], birth_df, "Grouped birth fractions", "Fraction of runs", (0.0, 1.02)),
    ]:
        x = np.arange(len(df.columns))
        width = 0.36
        for idx, condition_label in enumerate(df.index):
            ax.bar(
                x + (idx - 0.5) * width,
                df.loc[condition_label].to_numpy(dtype=float),
                width,
                label=condition_label,
                color=colors[idx],
                alpha=0.9,
            )
        ax.set_xticks(x, list(df.columns), rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Full matrix: stability, turnover, and birth summaries", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render public-facing grouped figures for the textual KV full matrix.")
    parser.add_argument("--matrix-dir", type=Path, required=True, help="Directory containing curriculum_on, curriculum_off, and localization artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for rendered public figures.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _render_performance_figure(args.matrix_dir, args.out_dir / "textual_kv_full_matrix_overview.png")
    _render_emergence_figure(args.matrix_dir, args.out_dir / "textual_kv_full_matrix_emergence.png")
    _render_localization_figure(args.matrix_dir, args.out_dir / "textual_kv_full_matrix_localization.png")
    _render_stability_figure(args.matrix_dir, args.out_dir / "textual_kv_full_matrix_stability.png")


if __name__ == "__main__":
    main()
