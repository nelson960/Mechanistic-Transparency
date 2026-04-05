#!/usr/bin/env python3
"""Generate an artifact-first KV training-dynamics notebook."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_PATH = ROOT / "notebook" / "kv_retrieve_algorithm_discovery.ipynb"


def _normalize_cell_source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _normalize_cell_source(text),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _normalize_cell_source(source),
    }


def build_notebook(*, run_dir: Path, checkpoint_id: str, visuals_dir: Path | None = None) -> dict:
    run_dir_str = str(run_dir.resolve())
    checkpoint_id_repr = json.dumps(checkpoint_id)
    visuals_enabled_repr = "True" if visuals_dir is not None else "False"
    cells = [
        markdown_cell(
            """
            # KV Training Dynamics Discovery

            This notebook is artifact-first.

            It does not assume a hardcoded model path or dataset path.
            Instead it reads:

            - `manifest.json`
            - `train_history.jsonl`
            - `summaries/checkpoint_index.csv`
            - `battery/<checkpoint_id>/behavior.json`
            - `battery/<checkpoint_id>/variable_scores.csv`
            - `battery/<checkpoint_id>/variable_faithfulness.csv`
            - `battery/<checkpoint_id>/operator_scores.csv`
            - `battery/<checkpoint_id>/localization.csv`
            - `battery/<checkpoint_id>/weight_metrics.json`
            - `battery/<checkpoint_id>/neuron_scores.csv`
            - `battery/<checkpoint_id>/feature_scores.csv` when `manifest["sae_tracking"]["enabled"]` is true
            - `battery/<checkpoint_id>/superposition_metrics.json` when `manifest["sae_tracking"]["enabled"]` is true

            If any required artifact is missing, the notebook fails immediately.
            """
        ),
        code_cell(
            f"""
            from __future__ import annotations

            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import Image, display

            pd.set_option("display.max_colwidth", 200)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 220)

            RUN_DIR = Path({json.dumps(run_dir_str)})
            CHECKPOINT_ID = {checkpoint_id_repr}
            VISUALS_DIR = Path({json.dumps(str(visuals_dir.resolve()))}) if {visuals_enabled_repr} else None
            MANIFEST_PATH = RUN_DIR / "manifest.json"
            TRAIN_HISTORY_PATH = RUN_DIR / "train_history.jsonl"
            CHECKPOINT_INDEX_PATH = RUN_DIR / "summaries" / "checkpoint_index.csv"
            BATTERY_DIR = RUN_DIR / "battery" / CHECKPOINT_ID
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            REQUIRED_PATHS = [
                MANIFEST_PATH,
                TRAIN_HISTORY_PATH,
                CHECKPOINT_INDEX_PATH,
                BATTERY_DIR / "behavior.json",
                BATTERY_DIR / "variable_scores.csv",
                BATTERY_DIR / "variable_faithfulness.csv",
                BATTERY_DIR / "operator_scores.csv",
                BATTERY_DIR / "localization.csv",
                BATTERY_DIR / "weight_metrics.json",
                BATTERY_DIR / "neuron_scores.csv",
            ]
            if bool(manifest["sae_tracking"]["enabled"]):
                REQUIRED_PATHS.extend(
                    [
                        BATTERY_DIR / "feature_scores.csv",
                        BATTERY_DIR / "superposition_metrics.json",
                    ]
                )
            if VISUALS_DIR is not None:
                REQUIRED_PATHS.extend(
                    [
                        VISUALS_DIR / "training_dynamics.png",
                        VISUALS_DIR / "network_change.png",
                        VISUALS_DIR / "behavior_and_variables.png",
                        VISUALS_DIR / "circuit_emergence.png",
                        VISUALS_DIR / "circuit_emergence.gif",
                        VISUALS_DIR / "operator_and_localization_heatmaps.png",
                        VISUALS_DIR / "representation_drift.png",
                        VISUALS_DIR / "neuron_feature_superposition.png",
                        VISUALS_DIR / "frame_manifest.csv",
                    ]
                )
            for required_path in REQUIRED_PATHS:
                if not required_path.exists():
                    raise FileNotFoundError(f"Missing required artifact: {{required_path}}")

            train_history = pd.read_json(TRAIN_HISTORY_PATH, lines=True)
            checkpoint_index = pd.read_csv(CHECKPOINT_INDEX_PATH)
            behavior = json.loads((BATTERY_DIR / "behavior.json").read_text(encoding="utf-8"))
            variable_scores = pd.read_csv(BATTERY_DIR / "variable_scores.csv")
            variable_faithfulness = pd.read_csv(BATTERY_DIR / "variable_faithfulness.csv")
            operator_scores = pd.read_csv(BATTERY_DIR / "operator_scores.csv")
            localization_scores = pd.read_csv(BATTERY_DIR / "localization.csv")
            weight_metrics = json.loads((BATTERY_DIR / "weight_metrics.json").read_text(encoding="utf-8"))
            neuron_scores = pd.read_csv(BATTERY_DIR / "neuron_scores.csv")
            feature_scores = None
            superposition_metrics = None
            frame_manifest = None
            if bool(manifest["sae_tracking"]["enabled"]):
                feature_scores = pd.read_csv(BATTERY_DIR / "feature_scores.csv")
                superposition_metrics = pd.read_json(BATTERY_DIR / "superposition_metrics.json")
            if VISUALS_DIR is not None:
                frame_manifest = pd.read_csv(VISUALS_DIR / "frame_manifest.csv")
            """
        ),
        markdown_cell(
            """
            ## Run Manifest

            The manifest is the source of truth for:

            - dataset location
            - model width/depth/head count
            - curriculum mode
            - checkpoint schedule
            - battery limits
            - birth thresholds
            """
        ),
        code_cell(
            """
            manifest
            """
        ),
        markdown_cell(
            """
            ## Checkpoint Index

            This table tracks the saved checkpoints and the main birth-relevant metrics across time.
            """
        ),
        code_cell(
            """
            checkpoint_index
            """
        ),
        markdown_cell(
            """
            ## Per-Step Dynamics

            This is the cheap-every-step telemetry:

            - batch loss
            - total grad norm
            - total update norm
            - per-matrix and per-head-slice parameter metrics
            """
        ),
        code_cell(
            """
            train_history[[
                "epoch",
                "global_step",
                "batch_index",
                "batch_loss",
                "curriculum_stage",
                "total_grad_norm",
                "total_update_norm",
                "relative_total_update_norm",
            ]]
            """
        ),
        code_cell(
            """
            train_history.iloc[0]["parameter_metrics"][:12]
            """
        ),
        markdown_cell(
            """
            ## Behavior

            Split-level metrics plus controlled prompt-family breakdown.
            """
        ),
        code_cell(
            """
            pd.DataFrame(
                [
                    {"split": split_name, **metrics}
                    for split_name, metrics in behavior["split_metrics"].items()
                ]
            )
            """
        ),
        code_cell(
            """
            pd.DataFrame(behavior["family_breakdown"]).sort_values(["accuracy", "margin"], ascending=[False, False])
            """
        ),
        markdown_cell(
            """
            ## Variable Recovery

            The summary rows report the best site for each variable, along with pooled and family-min scores.
            """
        ),
        code_cell(
            """
            variable_scores.query("row_kind == 'summary'")
            """
        ),
        code_cell(
            """
            variable_scores.query("row_kind == 'probe'").sort_values(
                ["variable", "eval_accuracy", "eval_margin_over_chance"],
                ascending=[True, False, False],
            ).head(24)
            """
        ),
        markdown_cell(
            """
            ## Variable Faithfulness

            These rows test whether patching the best site for a variable from one sweep example into another
            causes the answer to follow the symbolic intervention.
            """
        ),
        code_cell(
            """
            variable_faithfulness.query("row_kind == 'summary'")
            """
        ),
        code_cell(
            """
            variable_faithfulness.query("row_kind == 'family_summary'")
            """
        ),
        markdown_cell(
            """
            ## Operator Scores

            Every head is scored for routing-like and copy-like behavior.
            The candidate rows identify the selected routing and copy heads at this checkpoint.
            """
        ),
        code_cell(
            """
            operator_scores.query("row_kind == 'candidate'")
            """
        ),
        code_cell(
            """
            operator_scores.query("row_kind == 'head_score'").sort_values(
                ["routing_score", "copy_score"],
                ascending=[False, False],
            ).head(24)
            """
        ),
        code_cell(
            """
            operator_scores.query("row_kind != 'head_score'")
            """
        ),
        markdown_cell(
            """
            ## Localization

            This combines per-head final-position ablations with any path-patching rows available for the selected candidate pair.
            """
        ),
        code_cell(
            """
            localization_scores
            """
        ),
        markdown_cell(
            """
            ## Weight Metrics

            These metrics summarize whole-matrix and per-head slice geometry at the selected checkpoint.
            """
        ),
        code_cell(
            """
            pd.DataFrame(weight_metrics["matrices"]).head(20)
            """
        ),
        code_cell(
            """
            pd.DataFrame(weight_metrics["head_slices"]).head(20)
            """
        ),
        markdown_cell(
            """
            ## Neuron Tracking

            These rows show which MLP neurons are becoming selective for query key, matching slot, or selected value.
            """
        ),
        code_cell(
            """
            neuron_scores.query("row_kind == 'layer_top'")
            """
        ),
        code_cell(
            """
            neuron_scores.query("row_kind == 'neuron'").sort_values(
                ["layer_index", "best_selectivity_score", "activation_write_product"],
                ascending=[True, False, False],
            ).head(30)
            """
        ),
        markdown_cell(
            """
            ## SAE Feature Tracking

            When SAE tracking is enabled in the manifest, this section shows checkpoint-local sparse feature summaries
            over the configured dynamic sites.
            """
        ),
        code_cell(
            """
            if feature_scores is None:
                "SAE tracking disabled in manifest"
            else:
                feature_scores.query("row_kind == 'site_summary'")
            """
        ),
        code_cell(
            """
            if feature_scores is None:
                "SAE tracking disabled in manifest"
            else:
                feature_scores.query("row_kind == 'feature'").sort_values(
                    ["site", "best_selectivity_score", "mean_abs_activation"],
                    ascending=[True, False, False],
                ).head(30)
            """
        ),
        markdown_cell(
            """
            ## Superposition Metrics

            These metrics summarize feature density, decoder overlap, and decoder rank geometry for each tracked site.
            """
        ),
        code_cell(
            """
            if superposition_metrics is None:
                "SAE tracking disabled in manifest"
            else:
                superposition_metrics
            """
        ),
    ]

    if visuals_dir is not None:
        plot_files = [
            "training_dynamics.png",
            "network_change.png",
            "behavior_and_variables.png",
            "circuit_emergence.png",
            "operator_and_localization_heatmaps.png",
            "representation_drift.png",
            "neuron_feature_superposition.png",
        ]
        plot_files_repr = json.dumps(plot_files)
        cells.extend(
            [
                markdown_cell(
                    """
                    ## Visualization Bundle

                    These plots come from the artifact-first KV visualizer.
                    They show:

                    - optimization and update dynamics
                    - network-level parameter change
                    - behavior and variable formation
                    - head-level circuit emergence
                    - head heatmaps and ablation importance
                    - representation drift
                    - neuron, feature, and superposition summaries
                    """
                ),
                code_cell(
                    """
                    print(VISUALS_DIR / "circuit_emergence.gif")
                    display(Image(filename=str(VISUALS_DIR / "circuit_emergence.gif")))
                    """
                ),
                code_cell(
                    f"""
                    PLOT_FILES = {plot_files_repr}
                    for plot_name in PLOT_FILES:
                        plot_path = VISUALS_DIR / plot_name
                        print(plot_path)
                        display(Image(filename=str(plot_path)))
                    """
                ),
                markdown_cell(
                    """
                    ## Frame Sequence

                    The frame manifest indexes the epoch-by-epoch circuit snapshots.
                    """
                ),
                code_cell(
                    """
                    frame_manifest
                    """
                ),
                code_cell(
                    """
                    key_indices = sorted(set([0, len(frame_manifest) // 2, len(frame_manifest) - 1]))
                    for frame_index in key_indices:
                        row = frame_manifest.iloc[int(frame_index)]
                        print(f"frame {int(row['frame_index'])} | {row['checkpoint_id']} | epoch {int(row['epoch'])}")
                        display(Image(filename=str(row["frame_path"])))
                    """
                ),
                markdown_cell(
                    """
                    ## All Frames

                    Run the next cell to display every scheduled-checkpoint frame inside the notebook.
                    """
                ),
                code_cell(
                    """
                    for row in frame_manifest.itertuples(index=False):
                        print(f"frame {int(row.frame_index)} | {row.checkpoint_id} | epoch {int(row.epoch)}")
                        display(Image(filename=str(row.frame_path)))
                    """
                ),
            ]
        )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an artifact-first KV training-dynamics notebook.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory that contains manifest.json.")
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        required=True,
        help="Checkpoint stem to open from the run battery directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Output notebook path.",
    )
    parser.add_argument(
        "--visuals-dir",
        type=Path,
        default=None,
        help="Optional visualization directory produced by visualize_kv_circuit_dynamics.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = build_notebook(run_dir=args.run_dir, checkpoint_id=args.checkpoint_id, visuals_dir=args.visuals_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
    print(f"Wrote notebook to {args.out}")


if __name__ == "__main__":
    main()
