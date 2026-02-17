#!/usr/bin/env python3
"""One-shot pipeline: prompt -> model internals -> single viewer payload JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {i} in {path}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def normalize_source(src: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "position": safe_int(src.get("position"), -1),
        "token": str(src.get("token", "")),
        "token_id": src.get("token_id"),
        "score": safe_float(src.get("score")),
        "abs_score": safe_float(src.get("abs_score")),
    }


def normalize_component(comp: Dict[str, Any]) -> Dict[str, Any]:
    rank_val = comp.get("rank")
    if rank_val is None:
        rank_val = comp.get("component_rank_by_abs")
    return {
        "label": str(comp.get("label", comp.get("component_label", ""))),
        "kind": str(comp.get("kind", comp.get("component_kind", ""))),
        "layer": safe_int(comp.get("layer"), -1),
        "rank": safe_int(rank_val, 0),
        "score": safe_float(comp.get("score", comp.get("component_score", 0.0))) or 0.0,
        "abs_score": safe_float(comp.get("abs_score", comp.get("component_abs_score", 0.0)))
        or 0.0,
        "ablation_drop": safe_float(comp.get("ablation_drop")),
        "ablation_margin": safe_float(comp.get("ablation_margin")),
        "sources": [
            normalize_source(src)
            for src in (comp.get("sources") or comp.get("source_token_attributions") or [])
            if isinstance(src, dict)
        ],
    }


def build_top_paths(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paths: List[Dict[str, Any]] = []
    for comp in components:
        if comp.get("kind") != "attn_out":
            continue
        comp_score = float(comp.get("score") or 0.0)
        if comp_score == 0.0:
            continue
        for src in comp.get("sources", []):
            src_score = safe_float(src.get("score"))
            if src_score is None:
                continue
            if comp_score * src_score <= 0.0:
                continue
            paths.append(
                {
                    "source_position": safe_int(src.get("position"), -1),
                    "source_token": str(src.get("token", "")),
                    "source_score": float(src_score),
                    "component_label": str(comp.get("label", "")),
                    "component_kind": str(comp.get("kind", "")),
                    "layer": safe_int(comp.get("layer"), -1),
                    "component_score": comp_score,
                    "path_strength": abs(float(src_score)),
                }
            )
    paths.sort(key=lambda item: float(item["path_strength"]), reverse=True)
    return paths[:3]


def prepare_tracker_payload(
    rows: List[Dict[str, Any]],
    max_source_tokens: int,
    max_components: int,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        prompt_id = str(row.get("prompt_id", "unknown"))
        if "response_token_index" in row:
            idx = safe_int(row.get("response_token_index"), 0)
        elif "decision_index" in row:
            idx = safe_int(row.get("decision_index"), 0)
        else:
            idx = 0
        grouped[f"{prompt_id}::t{idx}"].append(row)

    prompts: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        items_sorted = sorted(
            items,
            key=lambda r: (
                safe_int(r.get("prompt_index"), 0),
                safe_int(r.get("response_token_index", r.get("decision_index", 0)), 0),
                safe_int(r.get("component_rank_by_abs"), 10**9),
            ),
        )
        first = items_sorted[0]

        prompt_id = str(first.get("prompt_id", "unknown"))
        decision_index = safe_int(
            first.get("response_token_index", first.get("decision_index", 0)),
            0,
        )
        response_token = str(first.get("response_token", first.get("target_token", "")))
        response_text = str(first.get("generated_response", first.get("response_text", "")))
        prompt_plus_response = str(first.get("prompt_plus_response", ""))
        if not prompt_plus_response and response_text:
            prompt_plus_response = str(first.get("prompt", "")) + response_text

        components: List[Dict[str, Any]] = []
        for row in items_sorted[: max(1, max_components)]:
            src_raw = row.get("source_token_attributions") or []
            norm = normalize_component(
                {
                    "component_label": row.get("component_label"),
                    "component_kind": row.get("component_kind"),
                    "layer": row.get("layer"),
                    "component_rank_by_abs": row.get("component_rank_by_abs"),
                    "component_score": row.get("component_score"),
                    "component_abs_score": row.get("component_abs_score"),
                    "ablation_drop": row.get("ablation_drop"),
                    "ablation_margin": row.get("ablation_margin"),
                    "source_token_attributions": src_raw[: max(1, max_source_tokens)],
                }
            )
            components.append(norm)

        prompts.append(
            {
                "group_key": key,
                "prompt_id": prompt_id,
                "display_id": f"{prompt_id}#t{decision_index}",
                "prompt_index": safe_int(first.get("prompt_index"), 0),
                "decision_index": decision_index,
                "task": str(first.get("task", "unknown")),
                "prompt": str(first.get("prompt", "")),
                "response_token": response_token,
                "response_text": response_text,
                "prompt_plus_response": prompt_plus_response,
                "target_token": str(first.get("target_token", "")),
                "foil_token": str(first.get("foil_token", "")),
                "clean_margin": safe_float(first.get("clean_margin")),
                "pred_margin": safe_float(first.get("pred_margin_from_components")),
                "reconstruction_error": safe_float(first.get("prompt_reconstruction_error")),
                "components": components,
                "top_paths": build_top_paths(components),
            }
        )

    prompts.sort(
        key=lambda x: (
            safe_int(x.get("prompt_index"), 0),
            safe_int(x.get("decision_index"), 0),
            str(x.get("prompt_id", "")),
        )
    )
    return {"num_prompts": len(prompts), "prompts": prompts}


def augment_graph_with_residual_stream(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out_nodes = [dict(n) for n in nodes]
    out_edges = [dict(e) for e in edges]

    node_by_id: Dict[str, Dict[str, Any]] = {
        str(n.get("id")): n for n in out_nodes if n.get("id") is not None
    }

    def add_node(node: Dict[str, Any]) -> None:
        node_id = str(node.get("id", ""))
        if not node_id or node_id in node_by_id:
            return
        node_by_id[node_id] = node
        out_nodes.append(node)

    def add_edge(source: str, target: str, kind: str, weight: float = 1.0) -> None:
        if source not in node_by_id or target not in node_by_id:
            return
        for edge in out_edges:
            if (
                str(edge.get("source")) == source
                and str(edge.get("target")) == target
                and str(edge.get("kind")) == kind
            ):
                return
        out_edges.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "weight": weight,
            }
        )

    layers = sorted(
        {
            safe_int(n.get("layer"), -1)
            for n in out_nodes
            if safe_int(n.get("layer"), -1) >= 0
        }
    )
    if not layers:
        return out_nodes, out_edges

    def get_anchor(layer: int) -> Optional[Dict[str, Any]]:
        candidates = [f"L{layer}_ln1", f"L{layer}_attn", f"L{layer}_ln2", f"L{layer}_mlp"]
        for node_id in candidates:
            node = node_by_id.get(node_id)
            if node is not None:
                return node
        return None

    for layer in layers:
        anchor = get_anchor(layer)
        x = float(anchor.get("x", layer)) if anchor else float(layer)
        add_node(
            {
                "id": f"L{layer}_resid",
                "kind": "resid",
                "label": f"L{layer}_resid",
                "x": x,
                "y": 0.95,
                "z": 0.0,
                "size": 140.0,
                "strength": None,
                "color": "#F4D35E",
                "layer": layer,
            }
        )

    for i, layer in enumerate(layers):
        resid_id = f"L{layer}_resid"
        if i == 0 and "embed" in node_by_id:
            add_edge("embed", resid_id, "residual_bus", 1.0)
        if i > 0:
            add_edge(f"L{layers[i - 1]}_resid", resid_id, "residual_bus", 1.0)

        if f"L{layer}_attn" in node_by_id:
            add_edge(f"L{layer}_attn", resid_id, "residual_write", 1.0)
        if f"L{layer}_mlp" in node_by_id:
            add_edge(f"L{layer}_mlp", resid_id, "residual_write", 1.0)
        if f"L{layer}_ln1" in node_by_id:
            add_edge(resid_id, f"L{layer}_ln1", "residual_read", 1.0)
        if f"L{layer}_ln2" in node_by_id:
            add_edge(resid_id, f"L{layer}_ln2", "residual_read", 1.0)

    last_resid = f"L{layers[-1]}_resid"
    if last_resid in node_by_id and "unembed" in node_by_id:
        add_edge(last_resid, "unembed", "residual_bus", 1.0)

    return out_nodes, out_edges


def mean_of(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def build_summary(graph: Dict[str, Any], tracker: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    prompts = tracker.get("prompts", [])

    node_kind_counts = Counter(str(node.get("kind", "unknown")) for node in nodes)
    edge_kind_counts = Counter(str(edge.get("kind", "unknown")) for edge in edges)

    margins = [safe_float(p.get("clean_margin")) for p in prompts]
    pred_margins = [safe_float(p.get("pred_margin")) for p in prompts]
    recon = [safe_float(p.get("reconstruction_error")) for p in prompts]

    component_count = 0
    component_kind_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    for prompt in prompts:
        for comp in prompt.get("components", []):
            component_count += 1
            kind = str(comp.get("kind", "unknown"))
            layer = safe_int(comp.get("layer"), -1)
            component_kind_counts[kind] += 1
            if layer >= 0:
                layer_counts[layer] += 1

    top_layers = [
        {"layer": layer, "count": count}
        for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_kind_counts": dict(sorted(node_kind_counts.items())),
            "edge_kind_counts": dict(sorted(edge_kind_counts.items())),
        },
        "tracker": {
            "num_prompts": safe_int(tracker.get("num_prompts"), len(prompts)),
            "component_count": component_count,
            "component_kind_counts": dict(sorted(component_kind_counts.items())),
            "mean_clean_margin": mean_of(margins),
            "mean_pred_margin": mean_of(pred_margins),
            "mean_reconstruction_error": mean_of(recon),
            "top_component_layers": top_layers,
        },
    }


def resolve_python_runner(project_root: Path, python_bin: Optional[str]) -> str:
    if python_bin:
        return python_bin
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_step(cmd: List[str], label: str, verbose: bool = False) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose and proc.stdout:
        print(proc.stdout.strip())
    if proc.returncode != 0:
        err = proc.stderr.strip() if proc.stderr else ""
        out = proc.stdout.strip() if proc.stdout else ""
        msg = f"{label} failed (exit={proc.returncode})"
        if out:
            msg += f"\nstdout:\n{out}"
        if err:
            msg += f"\nstderr:\n{err}"
        raise RuntimeError(msg)


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def build_pipeline_payload(
    project_root: Path,
    prompt: str,
    prompt_id: str,
    task: str,
    model: str,
    device: str,
    detail_level: str,
    mlp_top_neurons: int,
    top_source_tokens: int,
    max_components: int,
    augment_residual: bool,
    python_runner: str,
    workdir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    scripts_dir = project_root / "scripts"

    prompts_file = workdir / "single_prompt.jsonl"
    tracker_out = workdir / "tracker"
    map_out = workdir / "map"
    tracker_out.mkdir(parents=True, exist_ok=True)
    map_out.mkdir(parents=True, exist_ok=True)

    prompts_file.write_text(
        json.dumps({"id": prompt_id, "task": task, "prompt": prompt}, ensure_ascii=True) + "\n"
    )

    tracker_cmd = [
        python_runner,
        str(scripts_dir / "run_component_tracker.py"),
        "--model",
        model,
        "--prompts",
        str(prompts_file),
        "--outdir",
        str(tracker_out),
        "--device",
        device,
        "--max-prompts",
        "1",
        "--top-source-tokens",
        str(max(1, top_source_tokens)),
    ]
    print("[pipeline] running component tracker...")
    run_step(tracker_cmd, "component tracker", verbose=verbose)

    tracker_rows_file = tracker_out / "component_tracker.jsonl"
    if not tracker_rows_file.exists():
        raise FileNotFoundError(f"Expected tracker output missing: {tracker_rows_file}")

    map_cmd = [
        python_runner,
        str(scripts_dir / "visualize_3d_model_map.py"),
        "--model",
        model,
        "--device",
        device,
        "--tracker-file",
        str(tracker_rows_file),
        "--outdir",
        str(map_out),
        "--detail-level",
        detail_level,
        "--mlp-top-neurons",
        str(max(1, mlp_top_neurons)),
    ]
    print("[pipeline] building model graph...")
    run_step(map_cmd, "3d model map", verbose=verbose)

    nodes_file = map_out / "model_graph_nodes.json"
    edges_file = map_out / "model_graph_edges.json"
    if not nodes_file.exists() or not edges_file.exists():
        raise FileNotFoundError("Graph output missing from visualize_3d_model_map.py")

    nodes_raw = read_json(nodes_file)
    edges_raw = read_json(edges_file)
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise ValueError("model graph files are not arrays")

    if augment_residual:
        nodes, edges = augment_graph_with_residual_stream(nodes_raw, edges_raw)
    else:
        nodes, edges = nodes_raw, edges_raw

    rows = read_jsonl(tracker_rows_file)
    tracker_payload = prepare_tracker_payload(
        rows,
        max_source_tokens=max(1, top_source_tokens),
        max_components=max(1, max_components),
    )

    graph_payload = {"nodes": nodes, "edges": edges}
    summary_payload = build_summary(graph_payload, tracker_payload)

    payload: Dict[str, Any] = {
        "meta": {
            "mode": "single_prompt_pipeline",
            "model": model,
            "device": device,
            "prompt_id": prompt_id,
            "task": task,
            "prompt": prompt,
            "detail_level": detail_level,
            "mlp_top_neurons": int(max(1, mlp_top_neurons)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "graph": graph_payload,
        "tracker": tracker_payload,
        "summary": summary_payload,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tracker+graph pipeline from one prompt and emit one JSON payload."
    )
    parser.add_argument("--prompt", required=True, help="Prompt text to analyze.")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--prompt-id", default="prompt_001")
    parser.add_argument("--task", default="custom")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--detail-level",
        choices=["basic", "expanded", "maximal"],
        default="maximal",
    )
    parser.add_argument("--mlp-top-neurons", type=int, default=64)
    parser.add_argument("--top-source-tokens", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=80)
    parser.add_argument(
        "--out",
        default="outputs/viewer_payload.json",
        help="Single output JSON payload path.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable to run sub-scripts (defaults to .venv/bin/python if present).",
    )
    parser.add_argument(
        "--no-augment-residual",
        action="store_true",
        help="Disable residual-stream node/edge augmentation.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate tracker/map files (saved next to output).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout of internal pipeline steps.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    python_runner = resolve_python_runner(project_root, args.python_bin)

    if args.keep_intermediate:
        workdir = Path(
            tempfile.mkdtemp(prefix="viewer_pipeline_", dir=str(out_path.parent))
        )
        cleanup = False
    else:
        workdir = Path(tempfile.mkdtemp(prefix="viewer_pipeline_", dir=str(out_path.parent)))
        cleanup = True

    try:
        payload = build_pipeline_payload(
            project_root=project_root,
            prompt=args.prompt,
            prompt_id=args.prompt_id,
            task=args.task,
            model=args.model,
            device=args.device,
            detail_level=args.detail_level,
            mlp_top_neurons=args.mlp_top_neurons,
            top_source_tokens=args.top_source_tokens,
            max_components=args.max_components,
            augment_residual=not args.no_augment_residual,
            python_runner=python_runner,
            workdir=workdir,
            verbose=args.verbose,
        )
        write_payload(out_path, payload)
        print(f"[payload] wrote {out_path}")
        if not cleanup:
            print(f"[payload] intermediate files kept at {workdir}")
    finally:
        if cleanup and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
