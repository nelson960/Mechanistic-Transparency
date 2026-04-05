#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from scripts.kv_benchmark import discover_run_directories
from scripts.training_dynamics import load_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize KV formation-dynamics summaries for baseline and intervention runs."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory containing the formation runs, for example research/phase3/runs/kv_symbolic_formation_v1/baseline",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the rendered PNGs and CSVs will be written.",
    )
    return parser.parse_args()


def _load_run_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    summary_dir = run_dir / "summaries"
    required = {
        "formation": summary_dir / "formation_dynamics.csv",
        "family_gradients": summary_dir / "formation_family_gradients.csv",
        "checkpoint_index": summary_dir / "checkpoint_index.csv",
        "formation_births": summary_dir / "formation_births.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run {run_dir} is missing required formation summaries: {missing}")
    tables = {name: pd.read_csv(path) for name, path in required.items()}
    clamp_path = summary_dir / "clamp_responsiveness.csv"
    if clamp_path.exists():
        try:
            tables["clamp_responsiveness"] = pd.read_csv(clamp_path)
        except EmptyDataError:
            tables["clamp_responsiveness"] = pd.DataFrame()
    else:
        tables["clamp_responsiveness"] = pd.DataFrame()
    return tables


def _epoch_mean_formation(formation_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Q_top_key_accuracy",
        "R_top_slot_accuracy",
        "W_top_written_value_accuracy",
    ]
    grouped = formation_table.groupby("epoch", as_index=False)[columns].mean()
    return grouped.sort_values("epoch").reset_index(drop=True)


def _epoch_checkpoint(checkpoint_index: pd.DataFrame) -> pd.DataFrame:
    columns = ["epoch", "val_accuracy", "test_accuracy", "ood_accuracy", "routing_candidate", "copy_candidate"]
    available = [column for column in columns if column in checkpoint_index.columns]
    return checkpoint_index[available].sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)


def _style_axes(ax: plt.Axes, *, title: str, xlabel: str = "Epoch", ylabel: str = "Score") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)


def _save_baseline_trajectory_plot(
    baseline_runs: list[dict],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metric_specs = [
        ("Q_top_key_accuracy", "Q(t): Query-Support Quality"),
        ("R_top_slot_accuracy", "R(t): Routing Quality"),
        ("W_top_written_value_accuracy", "W(t): Value-Write Quality"),
        ("val_accuracy", "Validation Accuracy"),
    ]
    for axis, (metric_name, title) in zip(axes.flatten(), metric_specs, strict=True):
        mean_rows: list[pd.DataFrame] = []
        for run in baseline_runs:
            if metric_name == "val_accuracy":
                run_table = _epoch_checkpoint(run["checkpoint_index"])[["epoch", "val_accuracy"]].rename(columns={"val_accuracy": metric_name})
            else:
                run_table = _epoch_mean_formation(run["formation"])[["epoch", metric_name]]
            mean_rows.append(run_table.assign(run_id=run["run_id"]))
            axis.plot(run_table["epoch"], run_table[metric_name], alpha=0.35, linewidth=1.5, label=run["run_id"])
        combined = pd.concat(mean_rows, ignore_index=True)
        mean_table = combined.groupby("epoch", as_index=False)[metric_name].mean()
        axis.plot(mean_table["epoch"], mean_table[metric_name], color="black", linewidth=2.4, label="mean")
        _style_axes(axis, title=title)
        if metric_name != "val_accuracy":
            axis.set_ylim(-0.02, 1.02)
        else:
            axis.set_ylim(-0.02, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2 + len(baseline_runs))
    output_path = out_dir / "baseline_qrw_and_behavior.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_family_gradient_plot(
    baseline_runs: list[dict],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for axis, role_name in zip(axes, ["support", "retrieval"], strict=True):
        role_rows: list[pd.DataFrame] = []
        for run in baseline_runs:
            family = run["family_gradients"]
            if family.empty:
                continue
            grouped = family[family["role_name"] == role_name].groupby(["epoch", "family_name"], as_index=False)["grad_norm"].mean()
            grouped["run_id"] = run["run_id"]
            role_rows.append(grouped)
        if not role_rows:
            continue
        combined = pd.concat(role_rows, ignore_index=True)
        for family_name, family_group in combined.groupby("family_name"):
            mean_table = family_group.groupby("epoch", as_index=False)["grad_norm"].mean()
            axis.plot(mean_table["epoch"], mean_table["grad_norm"], linewidth=2.0, label=family_name)
        _style_axes(axis, title=f"{role_name.title()} Family Gradients", ylabel="Gradient Norm")
        axis.legend(loc="upper right")
    output_path = out_dir / "baseline_family_gradients.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _extract_birth_epoch(births: pd.DataFrame, metric_name: str) -> float | None:
    rows = births[births["metric_name"] == metric_name]
    if rows.empty:
        return None
    value = rows["birth_epoch"].iloc[0]
    return None if pd.isna(value) else float(value)


def _save_intervention_plot(
    baseline_runs_by_seed: dict[int, dict],
    intervention_runs: list[dict],
    out_dir: Path,
) -> list[Path]:
    output_paths: list[Path] = []
    for intervention in intervention_runs:
        baseline = baseline_runs_by_seed.get(intervention["seed"])
        if baseline is None:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        comparisons = [
            ("val_accuracy", "Validation Accuracy"),
            ("Q_top_key_accuracy", "Q(t)"),
            ("R_top_slot_accuracy", "R(t)"),
            ("W_top_written_value_accuracy", "W(t)"),
        ]
        for axis, (metric_name, title) in zip(axes.flatten(), comparisons, strict=True):
            if metric_name == "val_accuracy":
                base_table = _epoch_checkpoint(baseline["checkpoint_index"])[["epoch", "val_accuracy"]].rename(columns={"val_accuracy": metric_name})
                intervention_table = _epoch_checkpoint(intervention["checkpoint_index"])[["epoch", "val_accuracy"]].rename(columns={"val_accuracy": metric_name})
            else:
                base_table = _epoch_mean_formation(baseline["formation"])[["epoch", metric_name]]
                intervention_table = _epoch_mean_formation(intervention["formation"])[["epoch", metric_name]]
            axis.plot(base_table["epoch"], base_table[metric_name], label=f"baseline seed {baseline['seed']}", linewidth=2.2)
            axis.plot(intervention_table["epoch"], intervention_table[metric_name], label=intervention["run_id"], linewidth=2.2)
            if metric_name == "val_accuracy":
                birth_metric = "behavior_val_accuracy"
            elif metric_name == "Q_top_key_accuracy":
                birth_metric = "Q"
            elif metric_name == "R_top_slot_accuracy":
                birth_metric = "R"
            else:
                birth_metric = "W"
            baseline_birth = _extract_birth_epoch(baseline["formation_births"] if birth_metric in {"Q", "R", "W"} else baseline["emergence"], birth_metric)
            intervention_birth = _extract_birth_epoch(intervention["formation_births"] if birth_metric in {"Q", "R", "W"} else intervention["emergence"], birth_metric)
            if baseline_birth is not None:
                axis.axvline(baseline_birth, color="tab:blue", linestyle="--", alpha=0.35)
            if intervention_birth is not None:
                axis.axvline(intervention_birth, color="tab:orange", linestyle="--", alpha=0.35)
            _style_axes(axis, title=title)
            axis.set_ylim(-0.02, 1.02)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        output_path = out_dir / f"{intervention['run_id']}_comparison.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def _write_intervention_report(
    baseline_runs_by_seed: dict[int, dict],
    intervention_runs: list[dict],
    out_dir: Path,
) -> Path:
    rows: list[dict] = []
    for intervention in intervention_runs:
        baseline = baseline_runs_by_seed.get(intervention["seed"])
        if baseline is None:
            continue
        def _birth(run: dict, metric_name: str) -> float | None:
            if metric_name in {"Q", "R", "W"}:
                return _extract_birth_epoch(run["formation_births"], metric_name)
            return _extract_birth_epoch(run["emergence"], metric_name)

        rows.append(
            {
                "intervention_run_id": intervention["run_id"],
                "seed": intervention["seed"],
                "matched_baseline_run_id": baseline["run_id"],
                "baseline_behavior_birth": _birth(baseline, "behavior_val_accuracy"),
                "intervention_behavior_birth": _birth(intervention, "behavior_val_accuracy"),
                "baseline_R_birth": _birth(baseline, "R"),
                "intervention_R_birth": _birth(intervention, "R"),
                "baseline_Q_birth": _birth(baseline, "Q"),
                "intervention_Q_birth": _birth(intervention, "Q"),
                "baseline_W_birth": _birth(baseline, "W"),
                "intervention_W_birth": _birth(intervention, "W"),
                "baseline_final_ood": float(_epoch_checkpoint(baseline["checkpoint_index"])["ood_accuracy"].iloc[-1]),
                "intervention_final_ood": float(_epoch_checkpoint(intervention["checkpoint_index"])["ood_accuracy"].iloc[-1]),
            }
        )
    report = pd.DataFrame(rows)
    output_path = out_dir / "intervention_delay_report.csv"
    report.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_directories(target_dir)
    run_entries: list[dict] = []
    for run_dir in run_dirs:
        manifest = load_run_manifest(run_dir / "manifest.json")
        tables = _load_run_tables(run_dir)
        emergence_path = run_dir / "summaries" / "emergence.csv"
        tables["emergence"] = pd.read_csv(emergence_path) if emergence_path.exists() else pd.DataFrame()
        run_entries.append(
            {
                "run_dir": run_dir,
                "run_id": manifest.run_id,
                "seed": int(manifest.training.seed),
                "interventions": manifest.training_interventions,
                **tables,
            }
        )

    baseline_runs = [entry for entry in run_entries if not entry["interventions"]]
    intervention_runs = [entry for entry in run_entries if entry["interventions"]]
    if not baseline_runs:
        raise ValueError(f"No baseline runs found under {target_dir}")
    baseline_runs_by_seed = {entry["seed"]: entry for entry in baseline_runs}

    generated_paths = [
        _save_baseline_trajectory_plot(baseline_runs, out_dir),
        _save_family_gradient_plot(baseline_runs, out_dir),
        _write_intervention_report(baseline_runs_by_seed, intervention_runs, out_dir),
    ]
    generated_paths.extend(_save_intervention_plot(baseline_runs_by_seed, intervention_runs, out_dir))

    manifest_rows = pd.DataFrame(
        {
            "artifact_path": [str(path) for path in generated_paths],
        }
    )
    manifest_rows.to_csv(out_dir / "visual_manifest.csv", index=False)
    print(f"wrote {len(generated_paths)} artifacts to {out_dir}")


if __name__ == "__main__":
    main()
