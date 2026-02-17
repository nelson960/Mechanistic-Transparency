#!/usr/bin/env python3
"""Create a 3D structural map of a transformer with detailed internal components."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

# Keep plotting cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def robust_norm(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    return np.linalg.norm(arr.astype(np.float64), axis=axis)


def aggregate_layer_strengths(rows: List[Dict[str, Any]], n_layers: int) -> Tuple[np.ndarray, np.ndarray]:
    attn_vals = [[] for _ in range(n_layers)]
    mlp_vals = [[] for _ in range(n_layers)]

    for row in rows:
        layer = int(row.get("layer", -1))
        kind = row.get("component_kind", "")
        score = float(row.get("component_abs_score", 0.0))
        if not (0 <= layer < n_layers):
            continue
        if kind == "attn_out":
            attn_vals[layer].append(score)
        elif kind == "mlp_out":
            mlp_vals[layer].append(score)

    attn = np.array([float(np.mean(v)) if v else 0.0 for v in attn_vals], dtype=float)
    mlp = np.array([float(np.mean(v)) if v else 0.0 for v in mlp_vals], dtype=float)
    return attn, mlp


def scale_values(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(values, (low + high) * 0.5)
    out = (values - vmin) / (vmax - vmin)
    return low + out * (high - low)


def value_to_size(value: float, values: np.ndarray, low: float, high: float) -> float:
    if values.size == 0:
        return float((low + high) * 0.5)
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return float((low + high) * 0.5)
    t = (float(value) - vmin) / (vmax - vmin)
    return float(low + t * (high - low))


def layernorm_strengths(model: HookedTransformer, n_layers: int) -> Tuple[np.ndarray, np.ndarray]:
    ln1 = np.zeros(n_layers, dtype=float)
    ln2 = np.zeros(n_layers, dtype=float)
    for layer in range(n_layers):
        ln1_w = getattr(getattr(model.blocks[layer], "ln1"), "w", None)
        ln2_w = getattr(getattr(model.blocks[layer], "ln2"), "w", None)
        ln1[layer] = float(torch.norm(ln1_w).item()) if ln1_w is not None else 1.0
        ln2[layer] = float(torch.norm(ln2_w).item()) if ln2_w is not None else 1.0
    return ln1, ln2


def build_strength_pack(
    model: HookedTransformer,
    tracker_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n_layers = int(model.cfg.n_layers)

    wq = model.W_Q.detach().float().cpu().numpy()  # [L, H, d_model, d_head]
    wk = model.W_K.detach().float().cpu().numpy()
    wv = model.W_V.detach().float().cpu().numpy()
    wo = model.W_O.detach().float().cpu().numpy()  # [L, H, d_head, d_model]
    win = model.W_in.detach().float().cpu().numpy()  # [L, d_model, d_mlp]
    wout = model.W_out.detach().float().cpu().numpy()  # [L, d_mlp, d_model]

    head_q = robust_norm(wq.reshape(wq.shape[0], wq.shape[1], -1), axis=-1)
    head_k = robust_norm(wk.reshape(wk.shape[0], wk.shape[1], -1), axis=-1)
    head_v = robust_norm(wv.reshape(wv.shape[0], wv.shape[1], -1), axis=-1)
    head_o = robust_norm(wo.reshape(wo.shape[0], wo.shape[1], -1), axis=-1)

    attn_from_tracker, mlp_from_tracker = aggregate_layer_strengths(tracker_rows, n_layers=n_layers)

    attn_fallback = np.mean(head_o, axis=1)
    mlp_fallback = robust_norm(wout.reshape(wout.shape[0], -1), axis=-1)

    if float(np.max(attn_from_tracker)) <= 0.0:
        layer_attn = attn_fallback
    else:
        layer_attn = attn_from_tracker

    if float(np.max(mlp_from_tracker)) <= 0.0:
        layer_mlp = mlp_fallback
    else:
        layer_mlp = mlp_from_tracker

    ln1, ln2 = layernorm_strengths(model, n_layers=n_layers)
    mlp_in_strength = robust_norm(win.reshape(win.shape[0], -1), axis=-1)
    mlp_out_strength = robust_norm(wout.reshape(wout.shape[0], -1), axis=-1)
    mlp_act_strength = np.sqrt(np.maximum(1e-12, mlp_in_strength * mlp_out_strength))

    neuron_in = robust_norm(win, axis=1)  # [L, d_mlp]
    neuron_out = robust_norm(wout, axis=2)  # [L, d_mlp]
    neuron_score = np.sqrt(np.maximum(1e-12, neuron_in * neuron_out))

    return {
        "layer_attn": layer_attn,
        "layer_mlp": layer_mlp,
        "ln1": ln1,
        "ln2": ln2,
        "head_q": head_q,
        "head_k": head_k,
        "head_v": head_v,
        "head_o": head_o,
        "mlp_in": mlp_in_strength,
        "mlp_act": mlp_act_strength,
        "mlp_out": mlp_out_strength,
        "neuron_score": neuron_score,
        "neuron_in": neuron_in,
        "neuron_out": neuron_out,
    }


def add_node(
    nodes: List[Dict[str, Any]],
    node_id: str,
    kind: str,
    label: str,
    x: float,
    y: float,
    z: float,
    size: float,
    strength: Optional[float],
    color: str,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    node: Dict[str, Any] = {
        "id": node_id,
        "kind": kind,
        "label": label,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "size": float(size),
        "strength": None if strength is None else float(strength),
        "color": color,
    }
    if layer is not None:
        node["layer"] = int(layer)
    if head is not None:
        node["head"] = int(head)
    if extra:
        node.update(extra)
    nodes.append(node)


def add_edge(
    edges: List[Dict[str, Any]],
    source: str,
    target: str,
    kind: str,
    weight: float,
) -> None:
    edges.append(
        {
            "source": source,
            "target": target,
            "kind": kind,
            "weight": float(weight),
        }
    )


def build_graph(
    model: HookedTransformer,
    strengths: Dict[str, Any],
    include_heads: bool,
    detail_level: str,
    head_ring_radius: float,
    head_part_radius: float,
    mlp_top_neurons: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    layer_attn = np.asarray(strengths["layer_attn"], dtype=float)
    layer_mlp = np.asarray(strengths["layer_mlp"], dtype=float)
    ln1 = np.asarray(strengths["ln1"], dtype=float)
    ln2 = np.asarray(strengths["ln2"], dtype=float)

    head_q = np.asarray(strengths["head_q"], dtype=float)
    head_k = np.asarray(strengths["head_k"], dtype=float)
    head_v = np.asarray(strengths["head_v"], dtype=float)
    head_o = np.asarray(strengths["head_o"], dtype=float)

    mlp_in = np.asarray(strengths["mlp_in"], dtype=float)
    mlp_act = np.asarray(strengths["mlp_act"], dtype=float)
    mlp_out = np.asarray(strengths["mlp_out"], dtype=float)
    neuron_score = np.asarray(strengths["neuron_score"], dtype=float)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Size bands.
    attn_size_arr = scale_values(layer_attn, low=160, high=360)
    mlp_size_arr = scale_values(layer_mlp, low=160, high=360)
    ln1_size_arr = scale_values(ln1, low=90, high=180)
    ln2_size_arr = scale_values(ln2, low=90, high=180)

    head_size_arr = scale_values(head_o.reshape(-1), low=24, high=120).reshape(head_o.shape)
    q_size_arr = scale_values(head_q.reshape(-1), low=12, high=46).reshape(head_q.shape)
    k_size_arr = scale_values(head_k.reshape(-1), low=12, high=46).reshape(head_k.shape)
    v_size_arr = scale_values(head_v.reshape(-1), low=12, high=46).reshape(head_v.shape)
    o_size_arr = scale_values(head_o.reshape(-1), low=12, high=46).reshape(head_o.shape)

    mlp_in_size = scale_values(mlp_in, low=90, high=190)
    mlp_act_size = scale_values(mlp_act, low=90, high=190)
    mlp_out_size = scale_values(mlp_out, low=90, high=190)

    add_node(
        nodes,
        node_id="embed",
        kind="embed",
        label="embed",
        x=-1.0,
        y=0.0,
        z=0.0,
        size=380,
        strength=None,
        color="#D9822B",
    )
    add_node(
        nodes,
        node_id="unembed",
        kind="unembed",
        label="unembed",
        x=float(n_layers),
        y=0.0,
        z=0.0,
        size=380,
        strength=None,
        color="#D9822B",
    )

    prev_backbone = "embed"

    for layer in range(n_layers):
        x = float(layer)
        ln1_id = f"L{layer}_ln1"
        attn_id = f"L{layer}_attn"
        ln2_id = f"L{layer}_ln2"
        mlp_id = f"L{layer}_mlp"

        y_ln1 = 3.0
        y_attn = 1.8
        y_ln2 = 0.2
        y_mlp = -1.5

        add_node(
            nodes,
            ln1_id,
            "ln1",
            ln1_id,
            x,
            y_ln1,
            0.0,
            size=float(ln1_size_arr[layer]),
            strength=float(ln1[layer]),
            color="#4C78A8",
            layer=layer,
        )
        add_node(
            nodes,
            attn_id,
            "attn",
            attn_id,
            x,
            y_attn,
            0.0,
            size=float(attn_size_arr[layer]),
            strength=float(layer_attn[layer]),
            color="#2F6B8A",
            layer=layer,
        )
        add_node(
            nodes,
            ln2_id,
            "ln2",
            ln2_id,
            x,
            y_ln2,
            0.0,
            size=float(ln2_size_arr[layer]),
            strength=float(ln2[layer]),
            color="#4C78A8",
            layer=layer,
        )
        add_node(
            nodes,
            mlp_id,
            "mlp",
            mlp_id,
            x,
            y_mlp,
            0.0,
            size=float(mlp_size_arr[layer]),
            strength=float(layer_mlp[layer]),
            color="#3A9D5D",
            layer=layer,
        )

        add_edge(edges, prev_backbone, ln1_id, "backbone", 1.0)
        add_edge(edges, ln1_id, attn_id, "ln_to_attn", float(ln1[layer]))
        add_edge(edges, attn_id, ln2_id, "attn_to_ln", float(layer_attn[layer]))
        add_edge(edges, ln2_id, mlp_id, "ln_to_mlp", float(ln2[layer]))

        # MLP sub-operators.
        if detail_level in {"expanded", "maximal"}:
            fc_in_id = f"L{layer}_mlp_fc_in"
            act_id = f"L{layer}_mlp_act"
            fc_out_id = f"L{layer}_mlp_fc_out"

            add_node(
                nodes,
                fc_in_id,
                "mlp_fc_in",
                fc_in_id,
                x,
                -2.45,
                -0.65,
                size=float(mlp_in_size[layer]),
                strength=float(mlp_in[layer]),
                color="#4E9A57",
                layer=layer,
            )
            add_node(
                nodes,
                act_id,
                "mlp_act",
                act_id,
                x,
                -3.15,
                0.0,
                size=float(mlp_act_size[layer]),
                strength=float(mlp_act[layer]),
                color="#78C17D",
                layer=layer,
            )
            add_node(
                nodes,
                fc_out_id,
                "mlp_fc_out",
                fc_out_id,
                x,
                -2.45,
                0.65,
                size=float(mlp_out_size[layer]),
                strength=float(mlp_out[layer]),
                color="#4E9A57",
                layer=layer,
            )

            add_edge(edges, ln2_id, fc_in_id, "mlp_flow", float(ln2[layer]))
            add_edge(edges, fc_in_id, act_id, "mlp_flow", float(mlp_in[layer]))
            add_edge(edges, act_id, fc_out_id, "mlp_flow", float(mlp_act[layer]))
            add_edge(edges, fc_out_id, mlp_id, "mlp_flow", float(mlp_out[layer]))

            if detail_level == "maximal" and mlp_top_neurons > 0:
                scores = neuron_score[layer]
                topk = min(int(mlp_top_neurons), int(scores.shape[0]))
                top_idx = np.argpartition(scores, -topk)[-topk:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                top_vals = scores[top_idx]

                ring_radius = 1.45
                for idx_i, (neuron_idx, neuron_val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
                    theta = (2.0 * math.pi * idx_i) / float(max(1, len(top_idx)))
                    ny = -3.15 + ring_radius * math.cos(theta)
                    nz = ring_radius * math.sin(theta)
                    nid = f"L{layer}_N{int(neuron_idx)}"
                    nsize = value_to_size(float(neuron_val), top_vals, low=11.0, high=50.0)

                    add_node(
                        nodes,
                        nid,
                        "neuron",
                        nid,
                        x,
                        ny,
                        nz,
                        size=nsize,
                        strength=float(neuron_val),
                        color="#9CCB9F",
                        layer=layer,
                        extra={"neuron_index": int(neuron_idx)},
                    )
                    add_edge(edges, fc_in_id, nid, "neuron_path", float(neuron_val))
                    add_edge(edges, nid, fc_out_id, "neuron_path", float(neuron_val))

        if include_heads:
            for head in range(n_heads):
                theta = (2.0 * math.pi * head) / float(n_heads)
                hy = y_attn + head_ring_radius * math.cos(theta)
                hz = head_ring_radius * math.sin(theta)
                hid = f"L{layer}_H{head}"

                add_node(
                    nodes,
                    hid,
                    "head",
                    hid,
                    x,
                    hy,
                    hz,
                    size=float(head_size_arr[layer, head]),
                    strength=float(head_o[layer, head]),
                    color="#8F5AA8",
                    layer=layer,
                    head=head,
                )
                add_edge(edges, hid, attn_id, "head_to_attn", float(head_o[layer, head]))

                if detail_level in {"expanded", "maximal"}:
                    qid = f"L{layer}_H{head}_Q"
                    kid = f"L{layer}_H{head}_K"
                    vid = f"L{layer}_H{head}_V"
                    oid = f"L{layer}_H{head}_O"

                    add_node(
                        nodes,
                        qid,
                        "head_q",
                        qid,
                        x,
                        hy + head_part_radius,
                        hz,
                        size=float(q_size_arr[layer, head]),
                        strength=float(head_q[layer, head]),
                        color="#B07CC6",
                        layer=layer,
                        head=head,
                    )
                    add_node(
                        nodes,
                        kid,
                        "head_k",
                        kid,
                        x,
                        hy - head_part_radius,
                        hz,
                        size=float(k_size_arr[layer, head]),
                        strength=float(head_k[layer, head]),
                        color="#B07CC6",
                        layer=layer,
                        head=head,
                    )
                    add_node(
                        nodes,
                        vid,
                        "head_v",
                        vid,
                        x,
                        hy,
                        hz + head_part_radius,
                        size=float(v_size_arr[layer, head]),
                        strength=float(head_v[layer, head]),
                        color="#B07CC6",
                        layer=layer,
                        head=head,
                    )
                    add_node(
                        nodes,
                        oid,
                        "head_o",
                        oid,
                        x,
                        hy,
                        hz - head_part_radius,
                        size=float(o_size_arr[layer, head]),
                        strength=float(head_o[layer, head]),
                        color="#B07CC6",
                        layer=layer,
                        head=head,
                    )

                    add_edge(edges, ln1_id, qid, "qkv_read", float(head_q[layer, head]))
                    add_edge(edges, ln1_id, kid, "qkv_read", float(head_k[layer, head]))
                    add_edge(edges, ln1_id, vid, "qkv_read", float(head_v[layer, head]))
                    add_edge(edges, qid, hid, "qkv_to_head", float(head_q[layer, head]))
                    add_edge(edges, kid, hid, "qkv_to_head", float(head_k[layer, head]))
                    add_edge(edges, vid, hid, "qkv_to_head", float(head_v[layer, head]))
                    add_edge(edges, hid, oid, "head_to_o", float(head_o[layer, head]))
                    add_edge(edges, oid, attn_id, "o_to_attn", float(head_o[layer, head]))

        prev_backbone = mlp_id

    add_edge(edges, prev_backbone, "unembed", "backbone", 1.0)
    return nodes, edges


def node_lookup(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in nodes}


def draw_edges(ax: Any, nodes_by_id: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    for edge in edges:
        src = nodes_by_id.get(edge["source"])
        dst = nodes_by_id.get(edge["target"])
        if src is None or dst is None:
            continue

        kind = edge.get("kind", "other")
        weight = float(edge.get("weight", 1.0))

        if kind == "head_to_attn":
            color = "#8F5AA8"
            alpha = 0.16
            lw = min(1.9, 0.35 + 0.03 * math.log1p(max(0.0, weight)))
        elif kind in {"qkv_read", "qkv_to_head", "head_to_o", "o_to_attn"}:
            color = "#A783BE"
            alpha = 0.12
            lw = 0.45
        elif kind in {"mlp_flow", "neuron_path"}:
            color = "#5C9C65"
            alpha = 0.12 if kind == "neuron_path" else 0.24
            lw = 0.35 if kind == "neuron_path" else 0.9
        elif kind in {"ln_to_attn", "attn_to_ln", "ln_to_mlp"}:
            color = "#4C78A8"
            alpha = 0.28
            lw = 0.95
        elif kind == "backbone":
            color = "#1F1F1F"
            alpha = 0.46
            lw = 1.35
        else:
            color = "#6C7A89"
            alpha = 0.2
            lw = 0.7

        ax.plot(
            [float(src["x"]), float(dst["x"])],
            [float(src["y"]), float(dst["y"])],
            [float(src["z"]), float(dst["z"])],
            color=color,
            alpha=alpha,
            linewidth=lw,
        )


def render_figure(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    outpath: Path,
    elev: float,
    azim: float,
    title: str,
    annotate_every_layer: int,
) -> None:
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)

    nodes_by_id = node_lookup(nodes)
    draw_edges(ax, nodes_by_id, edges)

    kinds = sorted({n["kind"] for n in nodes})
    for kind in kinds:
        bucket = [n for n in nodes if n["kind"] == kind]
        if not bucket:
            continue
        x = np.array([float(n["x"]) for n in bucket], dtype=float)
        y = np.array([float(n["y"]) for n in bucket], dtype=float)
        z = np.array([float(n["z"]) for n in bucket], dtype=float)
        s = np.array([float(n["size"]) for n in bucket], dtype=float)
        c = bucket[0].get("color", "#888")
        alpha = 0.85
        if kind in {"neuron", "head_q", "head_k", "head_v", "head_o"}:
            alpha = 0.55
        elif kind in {"head"}:
            alpha = 0.65
        ax.scatter(x, y, z, s=s, c=c, alpha=alpha, depthshade=True, edgecolors="none")

    for node in nodes:
        if node["kind"] in {"embed", "unembed"}:
            ax.text(node["x"], node["y"], node["z"], f" {node['label']}", fontsize=9)
        elif node["kind"] in {"attn", "mlp", "ln1", "ln2"}:
            layer_num = int(node.get("layer", -1))
            if layer_num >= 0 and layer_num % max(1, annotate_every_layer) == 0:
                ax.text(node["x"], node["y"], node["z"], f" {node['label']}", fontsize=7)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Module Axis")
    ax.set_zlabel("Subcomponent Axis")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_guide(
    outdir: Path,
    model_name: str,
    n_layers: int,
    n_heads: int,
    detail_level: str,
    mlp_top_neurons: int,
    tracker_file: Optional[Path],
    node_count: int,
    edge_count: int,
) -> None:
    lines = [
        "# 3D Model Map Guide",
        "",
        f"- Model: `{model_name}`",
        f"- Layers: `{n_layers}`",
        f"- Heads per layer: `{n_heads}`",
        f"- Detail level: `{detail_level}`",
        f"- Top MLP neurons per layer: `{mlp_top_neurons}`",
        f"- Node count: `{node_count}`",
        f"- Edge count: `{edge_count}`",
        "",
        "## Node Types",
        "- `embed`, `unembed`: endpoints.",
        "- `ln1`, `attn`, `ln2`, `mlp`: main block flow per layer.",
        "- `head`: per-head aggregation unit.",
        "- `head_q`, `head_k`, `head_v`, `head_o`: per-head Q/K/V/O internals.",
        "- `mlp_fc_in`, `mlp_act`, `mlp_fc_out`: MLP operator stages.",
        "- `neuron`: top-scoring MLP neurons (maximal detail mode).",
        "",
        "## Sizing",
        "- Attention/MLP node size: tracker mean-abs contribution (or weight fallback).",
        "- Head sizes: `W_O` norms.",
        "- Q/K/V/O sizes: corresponding matrix norms (`W_Q`, `W_K`, `W_V`, `W_O`).",
        "- Neuron sizes: geometric mean of input/output neuron weight norms.",
    ]
    if tracker_file is not None:
        lines.append(f"- Tracker file used: `{tracker_file}`")
    else:
        lines.append("- Tracker file used: none (weight-only sizing fallback).")

    lines.extend(
        [
            "",
            "## Files",
            "- `model_map_3d_view1.png`: default perspective.",
            "- `model_map_3d_view2.png`: alternate angle.",
            "- `model_graph_nodes.json`: node table (coordinates, strengths, metadata).",
            "- `model_graph_edges.json`: edge table (typed links + weights).",
            "",
        ]
    )

    (outdir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 3D transformer structure map.")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tracker-file",
        default="outputs/component_tracker_gpt2_small/component_tracker.jsonl",
        help="Optional component tracker JSONL for attn/mlp sizing.",
    )
    parser.add_argument("--outdir", default="outputs/model_map_3d_gpt2_small")
    parser.add_argument("--no-heads", action="store_true", help="Hide per-head nodes.")
    parser.add_argument(
        "--detail-level",
        choices=["basic", "expanded", "maximal"],
        default="maximal",
        help="basic: layer blocks only; expanded: add QKVO + MLP operators; maximal: plus top neurons.",
    )
    parser.add_argument("--head-ring-radius", type=float, default=1.15)
    parser.add_argument("--head-part-radius", type=float, default=0.34)
    parser.add_argument("--mlp-top-neurons", type=int, default=16)
    parser.add_argument("--annotate-every-layer", type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"[3d-map] loading model={args.model} device={device}")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    tracker_path = Path(args.tracker_file)
    tracker_rows = read_jsonl(tracker_path)

    strengths = build_strength_pack(model=model, tracker_rows=tracker_rows)

    nodes, edges = build_graph(
        model=model,
        strengths=strengths,
        include_heads=not args.no_heads,
        detail_level=args.detail_level,
        head_ring_radius=float(args.head_ring_radius),
        head_part_radius=float(args.head_part_radius),
        mlp_top_neurons=max(0, int(args.mlp_top_neurons)),
    )

    (outdir / "model_graph_nodes.json").write_text(json.dumps(nodes, indent=2))
    (outdir / "model_graph_edges.json").write_text(json.dumps(edges, indent=2))

    title = (
        f"{args.model} 3D Internal Map "
        f"({n_layers}L, {n_heads}H, detail={args.detail_level})"
    )
    render_figure(
        nodes=nodes,
        edges=edges,
        outpath=outdir / "model_map_3d_view1.png",
        elev=20,
        azim=-62,
        title=title,
        annotate_every_layer=max(1, int(args.annotate_every_layer)),
    )
    render_figure(
        nodes=nodes,
        edges=edges,
        outpath=outdir / "model_map_3d_view2.png",
        elev=12,
        azim=24,
        title=title,
        annotate_every_layer=max(1, int(args.annotate_every_layer)),
    )

    write_guide(
        outdir=outdir,
        model_name=args.model,
        n_layers=n_layers,
        n_heads=n_heads,
        detail_level=args.detail_level,
        mlp_top_neurons=max(0, int(args.mlp_top_neurons)) if args.detail_level == "maximal" else 0,
        tracker_file=tracker_path if tracker_rows else None,
        node_count=len(nodes),
        edge_count=len(edges),
    )

    print(f"[3d-map] nodes={len(nodes)} edges={len(edges)}")
    print(f"[3d-map] wrote {outdir / 'model_map_3d_view1.png'}")
    print(f"[3d-map] wrote {outdir / 'model_map_3d_view2.png'}")
    print(f"[3d-map] wrote {outdir / 'model_graph_nodes.json'}")
    print(f"[3d-map] wrote {outdir / 'model_graph_edges.json'}")
    print(f"[3d-map] wrote {outdir / 'README.md'}")


if __name__ == "__main__":
    main()
