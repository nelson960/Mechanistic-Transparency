from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.reshape_as(x)


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(positions, frequencies)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        rope_cos, rope_sin = build_rope_cache(max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, capture: bool = False) -> tuple[torch.Tensor, dict | None]:
        _, seq_len, _ = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        cos = self.rope_cos[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        pattern = scores.softmax(dim=-1)
        head_out = torch.matmul(pattern, v)
        merged = self.merge_heads(head_out)
        out = self.o_proj(merged)

        if not capture:
            return out, None

        cache = {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "scores": scores.detach().cpu(),
            "pattern": pattern.detach().cpu(),
            "head_out": head_out.detach().cpu(),
            "merged": merged.detach().cpu(),
            "out": out.detach().cpu(),
        }
        return out, cache


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, capture: bool = False) -> tuple[torch.Tensor, dict | None] | torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = F.silu(gate) * up
        out = self.down_proj(activated)
        if not capture:
            return out
        cache = {
            "gate": gate.detach().cpu(),
            "up": up.detach().cpu(),
            "activated": activated.detach().cpu(),
            "out": out.detach().cpu(),
        }
        return out, cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, capture: bool = False) -> tuple[torch.Tensor, dict | None]:
        attn_in = self.norm1(x)
        attn_out, attn_cache = self.attn(attn_in, capture=capture)
        resid_after_attn = x + attn_out
        mlp_in = self.norm2(resid_after_attn)
        if capture:
            mlp_out, mlp_cache = self.mlp(mlp_in, capture=True)
        else:
            mlp_out = self.mlp(mlp_in)
            mlp_cache = None
        resid_after_mlp = resid_after_attn + mlp_out

        if not capture:
            return resid_after_mlp, None

        cache = {
            "resid_in": x.detach().cpu(),
            "attn_in": attn_in.detach().cpu(),
            "attention": attn_cache,
            "resid_after_attn": resid_after_attn.detach().cpu(),
            "mlp_in": mlp_in.detach().cpu(),
            "mlp": mlp_cache,
            "resid_after_mlp": resid_after_mlp.detach().cpu(),
        }
        return resid_after_mlp, cache


class TinyDecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, max_seq_len) for _ in range(n_layers)]
        )
        self.norm_final = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor, return_cache: bool = False) -> tuple[torch.Tensor, dict] | torch.Tensor:
        x = self.token_embed(input_ids)
        if not return_cache:
            for block in self.blocks:
                x, _ = block(x, capture=False)
            x = self.norm_final(x)
            return x @ self.token_embed.weight.T

        cache = {
            "token_embed": x.detach().cpu(),
            "blocks": [],
        }
        for block in self.blocks:
            x, block_cache = block(x, capture=True)
            cache["blocks"].append(block_cache)
        final_hidden = self.norm_final(x)
        logits = final_hidden @ self.token_embed.weight.T
        cache["final_hidden"] = final_hidden.detach().cpu()
        cache["logits"] = logits.detach().cpu()
        return logits, cache


def load_tiny_decoder_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, TinyDecoderTransformer]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TinyDecoderTransformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def forward_tiny_decoder_with_interventions(
    model: TinyDecoderTransformer,
    input_ids: torch.Tensor,
    interventions: list[dict[str, object]],
    *,
    return_cache: bool = False,
) -> tuple[torch.Tensor, dict] | torch.Tensor:
    resid = model.token_embed(input_ids)
    cache: dict[str, object] | None = None
    if return_cache:
        cache = {
            "token_embed": resid.detach().cpu(),
            "blocks": [],
        }
    for layer_index, block in enumerate(model.blocks):
        attn_in = block.norm1(resid)
        attn = block.attn
        _, seq_len, _ = attn_in.shape
        q = attn.split_heads(attn.q_proj(attn_in))
        k = attn.split_heads(attn.k_proj(attn_in))
        v = attn.split_heads(attn.v_proj(attn_in))

        cos = attn.rope_cos[:seq_len].to(device=attn_in.device, dtype=attn_in.dtype).unsqueeze(0).unsqueeze(0)
        sin = attn.rope_sin[:seq_len].to(device=attn_in.device, dtype=attn_in.dtype).unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_in.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        pattern = scores.softmax(dim=-1)
        head_out = torch.matmul(pattern, v)

        for intervention in interventions:
            if int(intervention["layer_index"]) != layer_index:
                continue
            if str(intervention["kind"]) == "head_resid_final_scale":
                head_index = int(intervention["head_index"])
                if head_index < 0 or head_index >= head_out.shape[1]:
                    raise ValueError(
                        f"Invalid intervention head index {head_index} for layer {layer_index + 1}"
                    )
                head_out = head_out.clone()
                head_out[:, head_index, -1, :] = head_out[:, head_index, -1, :] * float(intervention["scale"])

        attn_out = attn.o_proj(attn.merge_heads(head_out))
        resid_after_attn = resid + attn_out

        mlp_in = block.norm2(resid_after_attn)
        mlp_out = block.mlp(mlp_in)
        for intervention in interventions:
            if int(intervention["layer_index"]) != layer_index:
                continue
            if str(intervention["kind"]) == "mlp_out_final_scale":
                mlp_out = mlp_out.clone()
                mlp_out[:, -1, :] = mlp_out[:, -1, :] * float(intervention["scale"])

        resid = resid_after_attn + mlp_out
        if return_cache and cache is not None:
            cache["blocks"].append(
                {
                    "resid_in": resid.detach().cpu() if False else None,
                    "attn_in": attn_in.detach().cpu(),
                    "attention": {
                        "q": q.detach().cpu(),
                        "k": k.detach().cpu(),
                        "v": v.detach().cpu(),
                        "scores": scores.detach().cpu(),
                        "pattern": pattern.detach().cpu(),
                        "head_out": head_out.detach().cpu(),
                        "merged": attn.merge_heads(head_out).detach().cpu(),
                        "out": attn_out.detach().cpu(),
                    },
                    "resid_after_attn": resid_after_attn.detach().cpu(),
                    "mlp_in": mlp_in.detach().cpu(),
                    "mlp": {
                        "out": mlp_out.detach().cpu(),
                    },
                    "resid_after_mlp": resid.detach().cpu(),
                }
            )

    final_hidden = model.norm_final(resid)
    logits = final_hidden @ model.token_embed.weight.T
    if not return_cache:
        return logits
    if cache is None:
        raise ValueError("Expected an intervention cache dictionary when return_cache is enabled")
    cache["final_hidden"] = final_hidden.detach().cpu()
    cache["logits"] = logits.detach().cpu()
    for block_cache_index, block_cache in enumerate(cache["blocks"]):
        if block_cache.get("resid_in") is None:
            if block_cache_index == 0:
                block_cache["resid_in"] = cache["token_embed"]
            else:
                block_cache["resid_in"] = cache["blocks"][block_cache_index - 1]["resid_after_mlp"]
    return logits, cache
