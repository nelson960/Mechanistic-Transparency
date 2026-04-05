#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.phase3.scripts.kv_formation_dynamics import load_formation_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a fixed-head formation intervention manifest from a completed discovered-candidate baseline run."
    )
    parser.add_argument("--baseline-run-dir", type=Path, required=True, help="Path to a completed baseline run directory.")
    parser.add_argument(
        "--intervention-role",
        choices=["support", "retrieval", "placebo"],
        required=True,
        help="Which discovered role to damp during training.",
    )
    parser.add_argument("--epoch-start", type=int, required=True, help="First epoch at which the damp intervention is active.")
    parser.add_argument("--epoch-end", type=int, required=True, help="Last epoch at which the damp intervention is active.")
    parser.add_argument("--scale", type=float, default=0.0, help="Residual-output scale to apply during the intervention.")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Where to write the derived intervention manifest JSON.",
    )
    parser.add_argument(
        "--output-run-dir",
        type=str,
        default=None,
        help="Optional output_dir for the derived run. Defaults to a sibling run directory with an intervention suffix.",
    )
    return parser.parse_args()


def _latest_candidate(history_rows: list[dict], key: str) -> dict:
    for row in reversed(history_rows):
        candidate = row.get(key)
        if candidate is not None:
            return candidate
    raise ValueError(f"Could not find a resolved {key} entry in the formation history")


def main() -> None:
    args = parse_args()
    baseline_run_dir = args.baseline_run_dir.expanduser().resolve()
    manifest_path = baseline_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing baseline manifest: {manifest_path}")
    history_rows = load_formation_history(baseline_run_dir / "formation_history.jsonl")
    if not history_rows:
        raise ValueError(f"Formation history is empty for baseline run {baseline_run_dir}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    formation_payload = payload.get("formation")
    if not isinstance(formation_payload, dict) or not bool(formation_payload.get("enabled")):
        raise ValueError("Baseline manifest must include an enabled formation section")

    support_head = _latest_candidate(history_rows, "candidate_support_head")
    retrieval_head = _latest_candidate(history_rows, "candidate_retrieval_head")
    placebo_head = _latest_candidate(history_rows, "candidate_placebo_head")
    resolved_heads = {
        "support": support_head,
        "retrieval": retrieval_head,
        "placebo": placebo_head,
    }
    target_head = resolved_heads[args.intervention_role]
    if target_head is None:
        raise ValueError(f"Could not derive a {args.intervention_role} candidate from the baseline run")

    formation_payload["candidate_mode"] = "fixed"
    formation_payload["candidate_support_head"] = {
        "layer_index": int(support_head["layer_index"]),
        "head_index": int(support_head["head_index"]),
    }
    formation_payload["candidate_retrieval_head"] = {
        "layer_index": int(retrieval_head["layer_index"]),
        "head_index": int(retrieval_head["head_index"]),
    }
    formation_payload["candidate_placebo_head"] = (
        None
        if placebo_head is None
        else {
            "layer_index": int(placebo_head["layer_index"]),
            "head_index": int(placebo_head["head_index"]),
        }
    )

    intervention_name = f"early_{args.intervention_role}_damp"
    payload["training_interventions"] = [
        {
            "name": intervention_name,
            "kind": "head_resid_final_scale",
            "layer_index": int(target_head["layer_index"]),
            "head_index": int(target_head["head_index"]),
            "epoch_start": args.epoch_start,
            "epoch_end": args.epoch_end,
            "scale": float(args.scale),
            "position": "final",
        }
    ]

    baseline_output_dir = str(payload["output_dir"])
    default_run_dir_path = baseline_run_dir.parent / f"{baseline_run_dir.name}__{intervention_name}"
    try:
        default_output_run_dir = str(default_run_dir_path.relative_to(Path.cwd()))
    except ValueError:
        default_output_run_dir = str(default_run_dir_path)
    derived_run_dir = (
        args.output_run_dir
        if args.output_run_dir is not None
        else default_output_run_dir
    )
    payload["output_dir"] = derived_run_dir
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {args.output_manifest}")
    print(f"baseline_output_dir={baseline_output_dir}")
    print(f"derived_output_dir={derived_run_dir}")
    print(
        f"intervention_target=block{int(target_head['layer_index']) + 1}_head{int(target_head['head_index'])}"
    )


if __name__ == "__main__":
    main()
