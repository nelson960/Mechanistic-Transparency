#!/usr/bin/env python3
"""Track prompt-level component behavior with source traces and ablation effects."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompts(path: Path, max_prompts: int) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(
                {
                    "id": row.get("id", f"row_{len(prompts)}"),
                    "task": row.get("task", "unknown"),
                    "prompt": row["prompt"],
                }
            )
    else:
        raw = json.loads(path.read_text())
        for idx, row in enumerate(raw):
            prompts.append(
                {
                    "id": row.get("id", f"row_{idx}"),
                    "task": row.get("task", "unknown"),
                    "prompt": row["prompt"],
                }
            )

    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def clean_token_text(text: str) -> str:
    return text.replace("\n", "\\n")


def token_str(model: HookedTransformer, token_id: int) -> str:
    return clean_token_text(model.tokenizer.decode([token_id]))


def parse_component_label(label: str) -> Tuple[str, int]:
    if label == "embed":
        return "embed", -1
    if label == "pos_embed":
        return "pos_embed", -1
    match = re.match(r"^(\d+)_(attn_out|mlp_out)$", label)
    if not match:
        return "unknown", -1
    return match.group(2), int(match.group(1))


def ensure_next_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    return logits[-1]


def ensure_component_matrix(stack: torch.Tensor) -> torch.Tensor:
    if stack.ndim == 2:
        return stack
    if stack.ndim == 3 and stack.shape[1] == 1:
        return stack[:, 0, :]
    raise ValueError(f"Unexpected residual stack shape: {tuple(stack.shape)}")


def build_zero_last_token_hook() -> Any:
    def hook_fn(value: torch.Tensor, hook: Any) -> torch.Tensor:
        out = value.clone()
        out[:, -1, :] = 0.0
        return out

    return hook_fn


def component_hook_name(kind: str, layer: int) -> Optional[str]:
    if kind == "embed":
        return "hook_embed"
    if kind == "pos_embed":
        return "hook_pos_embed"
    if kind == "attn_out" and layer >= 0:
        return f"blocks.{layer}.hook_attn_out"
    if kind == "mlp_out" and layer >= 0:
        return f"blocks.{layer}.hook_mlp_out"
    return None


def component_activation(
    cache: Any,
    label: str,
    kind: str,
    layer: int,
    position: int,
) -> torch.Tensor:
    if kind == "embed":
        return cache["hook_embed"][0, position, :]
    if kind == "pos_embed":
        return cache["hook_pos_embed"][0, position, :]
    if kind == "attn_out":
        return cache[f"blocks.{layer}.hook_attn_out"][0, position, :]
    if kind == "mlp_out":
        return cache[f"blocks.{layer}.hook_mlp_out"][0, position, :]
    raise ValueError(f"Unsupported component label={label} kind={kind}")


def attention_source_contributions(
    model: HookedTransformer,
    cache: Any,
    layer: int,
    dest_pos: int,
    direction: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, dest_pos, :]
    value = cache[f"blocks.{layer}.attn.hook_v"]
    wo = model.W_O[layer]

    head_source_write = torch.einsum("bshd,hdm->bshm", value, wo)
    weighted_source_write = head_source_write * pattern.permute(0, 2, 1).unsqueeze(-1)
    source_vectors = weighted_source_write.sum(dim=2)

    if hasattr(model, "b_O") and model.b_O is not None:
        bias = model.b_O[layer].to(source_vectors.dtype).unsqueeze(0).unsqueeze(0)
    else:
        bias = torch.zeros(
            (1, 1, source_vectors.shape[-1]),
            dtype=source_vectors.dtype,
            device=source_vectors.device,
        )

    stack = torch.cat([source_vectors, bias], dim=1)
    stack = stack.permute(1, 0, 2).unsqueeze(2)
    scaled_stack = cache.apply_ln_to_stack(
        stack,
        layer=model.cfg.n_layers,
        pos_slice=dest_pos,
        has_batch_dim=True,
    )
    scaled = scaled_stack[:, 0, 0, :]
    margin_direction = direction.to(dtype=scaled.dtype, device=scaled.device)
    contrib = torch.matmul(scaled, margin_direction)
    return contrib[:-1], float(contrib[-1].item())


def margin_from_logits(logits: torch.Tensor, target_id: int, foil_id: int) -> float:
    return float((logits[target_id] - logits[foil_id]).item())


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(x.norm().item() * y.norm().item())
    if denom <= 0.0:
        return None
    return float((x * y).sum().item() / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track component-level activations and interventions for prompt decisions."
    )
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--prompts", default="data/pilot_prompts.jsonl")
    parser.add_argument("--outdir", default="outputs/component_tracker_gpt2_small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--top-source-tokens", type=int, default=8)
    parser.add_argument(
        "--also-write-json",
        action="store_true",
        help="Also write component_tracker.json (array form).",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path, args.max_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")

    device = choose_device(args.device)
    print(f"[tracker] loading model={args.model} device={device}")
    model = HookedTransformer.from_pretrained(args.model, device=device)

    all_rows: List[Dict[str, Any]] = []
    prompt_summaries: List[Dict[str, Any]] = []

    with torch.no_grad():
        for prompt_idx, row in enumerate(prompts):
            prompt_id = row["id"]
            task = row["task"]
            prompt = row["prompt"]

            tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
            token_ids = [int(x) for x in tokens[0].tolist()]
            token_texts = [token_str(model, token_id) for token_id in token_ids]
            decision_pos = int(tokens.shape[-1] - 1)

            logits, cache = model.run_with_cache(tokens, return_type="logits")
            next_logits = ensure_next_logits(logits)
            top2_vals, top2_ids = torch.topk(next_logits, k=2)
            target_id = int(top2_ids[0].item())
            foil_id = int(top2_ids[1].item())
            clean_margin = float((top2_vals[0] - top2_vals[1]).item())
            response_token_text = token_str(model, target_id)
            prompt_plus_response = f"{prompt}{response_token_text}"

            direction = model.W_U[:, target_id] - model.W_U[:, foil_id]
            bias_diff = 0.0
            if hasattr(model, "b_U") and model.b_U is not None:
                bias_diff = float((model.b_U[target_id] - model.b_U[foil_id]).item())

            resid_stack, labels = cache.decompose_resid(
                layer=model.cfg.n_layers,
                mlp_input=False,
                mode="all",
                apply_ln=True,
                pos_slice=decision_pos,
                incl_embeds=True,
                return_labels=True,
            )
            components = ensure_component_matrix(resid_stack)
            scores = torch.matmul(components, direction)
            abs_total = float(scores.abs().sum().item())

            pred_margin = float(bias_diff + scores.sum().item())
            recon_error = abs(clean_margin - pred_margin) / max(1.0, abs(clean_margin))

            sorted_indices = sorted(
                range(len(labels)),
                key=lambda idx: abs(float(scores[idx].item())),
                reverse=True,
            )

            prompt_rows: List[Dict[str, Any]] = []
            for rank, component_index in enumerate(sorted_indices, start=1):
                label = labels[component_index]
                kind, layer = parse_component_label(label)
                score = float(scores[component_index].item())
                abs_score = abs(score)
                abs_fraction = float(abs_score / abs_total) if abs_total > 0 else 0.0

                act_vec = component_activation(
                    cache=cache,
                    label=label,
                    kind=kind,
                    layer=layer,
                    position=decision_pos,
                )
                activation_l2 = float(act_vec.norm().item())

                source_top: List[Dict[str, Any]] = []
                source_bias: Optional[float] = None
                source_sum: Optional[float] = None
                source_error: Optional[float] = None

                if kind == "attn_out":
                    src_scores, source_bias = attention_source_contributions(
                        model=model,
                        cache=cache,
                        layer=layer,
                        dest_pos=decision_pos,
                        direction=direction,
                    )
                    src_sum_no_bias = float(src_scores.sum().item())
                    source_sum = float(src_sum_no_bias + source_bias)
                    source_error = float(score - source_sum)

                    src_abs_total = float(src_scores.abs().sum().item())
                    src_top_idx = sorted(
                        range(len(token_ids)),
                        key=lambda pos: abs(float(src_scores[pos].item())),
                        reverse=True,
                    )[: args.top_source_tokens]
                    for pos in src_top_idx:
                        src_score = float(src_scores[pos].item())
                        src_abs = abs(src_score)
                        source_top.append(
                            {
                                "position": int(pos),
                                "token_id": int(token_ids[pos]),
                                "token": token_texts[pos],
                                "score": src_score,
                                "abs_score": float(src_abs),
                                "abs_fraction": float(src_abs / src_abs_total)
                                if src_abs_total > 0
                                else 0.0,
                            }
                        )

                hook_name = component_hook_name(kind=kind, layer=layer)
                ablation_margin: Optional[float] = None
                ablation_drop: Optional[float] = None
                ablation_relative_drop: Optional[float] = None
                ablation_top1_after_id: Optional[int] = None
                ablation_top1_after_token: Optional[str] = None

                if hook_name is not None:
                    hooks = [(hook_name, build_zero_last_token_hook())]
                    ablated_logits = model.run_with_hooks(
                        tokens,
                        return_type="logits",
                        fwd_hooks=hooks,
                    )
                    next_ablated = ensure_next_logits(ablated_logits)
                    ablation_margin = margin_from_logits(next_ablated, target_id, foil_id)
                    ablation_drop = float(clean_margin - ablation_margin)
                    ablation_relative_drop = float(
                        ablation_drop / max(1.0, abs(clean_margin))
                    )
                    ablation_top1_after_id = int(torch.argmax(next_ablated).item())
                    ablation_top1_after_token = token_str(model, ablation_top1_after_id)

                component_row: Dict[str, Any] = {
                    "model": args.model,
                    "device": device,
                    "prompt_index": int(prompt_idx),
                    "prompt_id": prompt_id,
                    "task": task,
                    "prompt": prompt,
                    "response_token_index": 0,
                    "response_token_id": target_id,
                    "response_token": response_token_text,
                    "response_text": response_token_text,
                    "generated_response": response_token_text,
                    "prompt_plus_response": prompt_plus_response,
                    "prompt_token_count": int(tokens.shape[-1]),
                    "decision_position": decision_pos,
                    "target_token_id": target_id,
                    "target_token": token_str(model, target_id),
                    "foil_token_id": foil_id,
                    "foil_token": token_str(model, foil_id),
                    "clean_margin": clean_margin,
                    "pred_margin_from_components": pred_margin,
                    "prompt_reconstruction_error": recon_error,
                    "component_rank_by_abs": int(rank),
                    "component_label": label,
                    "component_kind": kind,
                    "layer": int(layer),
                    "component_score": score,
                    "component_abs_score": float(abs_score),
                    "component_abs_fraction": abs_fraction,
                    "component_activation_l2": activation_l2,
                    "ablation_margin": ablation_margin,
                    "ablation_drop": ablation_drop,
                    "ablation_relative_drop": ablation_relative_drop,
                    "ablation_top1_after_id": ablation_top1_after_id,
                    "ablation_top1_after_token": ablation_top1_after_token,
                    "source_token_attributions": source_top,
                    "source_bias_contribution": source_bias,
                    "source_contribution_sum": source_sum,
                    "source_reconstruction_error": source_error,
                }
                prompt_rows.append(component_row)

            prompt_rows_sorted = sorted(
                prompt_rows,
                key=lambda x: x["component_rank_by_abs"],
            )
            all_rows.extend(prompt_rows_sorted)

            top_component = prompt_rows_sorted[0]
            prompt_summaries.append(
                {
                    "prompt_id": prompt_id,
                    "task": task,
                    "clean_margin": clean_margin,
                    "pred_margin_from_components": pred_margin,
                    "reconstruction_error": recon_error,
                    "top_component_label": top_component["component_label"],
                    "top_component_score": top_component["component_score"],
                    "top_component_ablation_drop": top_component["ablation_drop"],
                }
            )
            print(
                f"[tracker] {prompt_id}: target={token_str(model, target_id)!r} "
                f"margin={clean_margin:.4f} recon_err={recon_error:.4f} "
                f"rows={len(prompt_rows_sorted)}"
            )

    tracker_path = outdir / "component_tracker.jsonl"
    with tracker_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    tracker_json_path = outdir / "component_tracker.json"
    if args.also_write_json:
        tracker_json_path.write_text(json.dumps(all_rows, indent=2))

    ablatable_rows = [row for row in all_rows if row["ablation_drop"] is not None]
    pred_vals = [float(row["component_score"]) for row in ablatable_rows]
    obs_vals = [float(row["ablation_drop"]) for row in ablatable_rows]
    abs_pred_vals = [abs(x) for x in pred_vals]
    abs_obs_vals = [abs(x) for x in obs_vals]
    support_rows = [row for row in ablatable_rows if row["component_score"] > 0]

    recon_errors = [float(item["reconstruction_error"]) for item in prompt_summaries]
    summary: Dict[str, Any] = {
        "model": args.model,
        "device": device,
        "prompt_file": str(prompts_path),
        "num_prompts": len(prompt_summaries),
        "num_component_rows": len(all_rows),
        "num_ablatable_rows": len(ablatable_rows),
        "mean_prompt_reconstruction_error": float(statistics.fmean(recon_errors))
        if recon_errors
        else None,
        "median_prompt_reconstruction_error": float(statistics.median(recon_errors))
        if recon_errors
        else None,
        "max_prompt_reconstruction_error": float(max(recon_errors))
        if recon_errors
        else None,
        "pearson_score_vs_ablation_drop": pearson(pred_vals, obs_vals),
        "pearson_abs_score_vs_abs_ablation_drop": pearson(abs_pred_vals, abs_obs_vals),
        "mean_abs_score": float(statistics.fmean(abs_pred_vals))
        if abs_pred_vals
        else None,
        "mean_abs_ablation_drop": float(statistics.fmean(abs_obs_vals))
        if abs_obs_vals
        else None,
        "mean_abs_error_score_vs_ablation": float(
            statistics.fmean(abs(p - o) for p, o in zip(pred_vals, obs_vals))
        )
        if pred_vals
        else None,
        "support_only": {
            "count": len(support_rows),
            "pearson": pearson(
                [float(r["component_score"]) for r in support_rows],
                [float(r["ablation_drop"]) for r in support_rows],
            ),
        },
        "prompt_summaries": prompt_summaries,
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[tracker] wrote {tracker_path}")
    if args.also_write_json:
        print(f"[tracker] wrote {tracker_json_path}")
    print(f"[tracker] wrote {summary_path}")


if __name__ == "__main__":
    main()
