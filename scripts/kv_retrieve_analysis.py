from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DatasetBundle:
    metadata: dict
    raw_splits: dict[str, list[dict]]
    vocab: list[str]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    value_tokens: list[str]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    rows: list[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Blank line found in {path} at line {line_number}")
            rows.append(json.loads(line))
    return rows


def load_dataset_bundle(dataset_dir: Path) -> DatasetBundle:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")

    with metadata_path.open() as handle:
        metadata = json.load(handle)

    vocab = (
        metadata["vocabulary"]["special"]
        + metadata["vocabulary"]["keys"]
        + metadata["vocabulary"]["values"]
    )
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    split_names = metadata.get("splits")
    if not isinstance(split_names, dict) or not split_names:
        raise ValueError(f"Dataset metadata must contain a non-empty 'splits' object: {metadata_path}")
    raw_splits = {
        name: read_jsonl(dataset_dir / f"{name}.jsonl")
        for name in split_names
    }

    return DatasetBundle(
        metadata=metadata,
        raw_splits=raw_splits,
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        value_tokens=metadata["vocabulary"]["values"],
    )


def encode_prompt(prompt: str, token_to_id: dict[str, int]) -> torch.Tensor:
    tokens = prompt.split()
    unknown_tokens = [token for token in tokens if token not in token_to_id]
    if unknown_tokens:
        raise ValueError(f"Unknown tokens in prompt {prompt!r}: {unknown_tokens}")
    return torch.tensor([token_to_id[token] for token in tokens], dtype=torch.long)


def decode_token(token_id: int, id_to_token: dict[int, str]) -> str:
    return id_to_token[int(token_id)]


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


class KVRetrievalTransformer(nn.Module):
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


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, KVRetrievalTransformer]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = KVRetrievalTransformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def run_prompt(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    device: torch.device,
    expected_target: str | None = None,
    return_cache: bool = False,
) -> tuple[dict, dict | None]:
    encoded_prompt = encode_prompt(prompt, bundle.token_to_id).unsqueeze(0).to(device)
    with torch.no_grad():
        if return_cache:
            logits, cache = model(encoded_prompt, return_cache=True)
        else:
            logits = model(encoded_prompt)
            cache = None
        final_logits = logits[0, -1].detach().cpu()

    predicted_id = int(final_logits.argmax().item())
    predicted_token = decode_token(predicted_id, bundle.id_to_token)
    result = {
        "prompt": prompt,
        "predicted_token": predicted_token,
    }

    if expected_target is not None:
        if expected_target not in bundle.token_to_id:
            raise ValueError(f"Unknown target token: {expected_target}")
        expected_id = bundle.token_to_id[expected_target]
        competing_logits = final_logits.clone()
        competing_logits[expected_id] = float("-inf")
        foil_id = int(competing_logits.argmax().item())
        result.update(
            {
                "target_token": expected_target,
                "foil_token": decode_token(foil_id, bundle.id_to_token),
                "target_logit": float(final_logits[expected_id].item()),
                "foil_logit": float(final_logits[foil_id].item()),
                "margin": float(final_logits[expected_id].item() - final_logits[foil_id].item()),
                "correct": predicted_token == expected_target,
            }
        )

    top_k = torch.topk(final_logits, k=5)
    result["top_5_tokens"] = [
        decode_token(token_id, bundle.id_to_token) for token_id in top_k.indices.tolist()
    ]
    result["top_5_logits"] = [float(value) for value in top_k.values.tolist()]
    return result, cache


def build_attention_tables(prompt: str, cache: dict) -> list[tuple[str, pd.DataFrame]]:
    prompt_tokens = prompt.split()
    tables: list[tuple[str, pd.DataFrame]] = []
    for layer_index, block_cache in enumerate(cache["blocks"], start=1):
        final_position_attention = block_cache["attention"]["pattern"][0, :, -1, :]
        attention_df = pd.DataFrame(
            final_position_attention.T.detach().cpu().numpy(),
            index=prompt_tokens,
            columns=[f"L{layer_index}H{head}" for head in range(final_position_attention.shape[0])],
        )
        tables.append((f"Final-position attention, layer {layer_index}", attention_df))
    return tables


def make_query_swap_prompt(row: dict) -> tuple[str, str]:
    alternate_pairs = [pair for pair in row["context_pairs"] if pair["key"] != row["query_key"]]
    if not alternate_pairs:
        raise ValueError("Could not construct a corrupted prompt because the example has no alternate context key.")
    alternate_pair = alternate_pairs[0]
    corrupt_tokens = row["prompt"].split()
    if corrupt_tokens[-3] != "Q" or corrupt_tokens[-1] != "->":
        raise ValueError(f"Unexpected prompt format for query swap: {row['prompt']}")
    corrupt_tokens[-2] = alternate_pair["key"]
    return " ".join(corrupt_tokens), alternate_pair["value"]


def forward_with_head_ablation(
    model: nn.Module,
    input_ids: torch.Tensor,
    ablation: dict | None = None,
) -> torch.Tensor:
    resid = model.token_embed(input_ids)
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

        if ablation is not None and ablation["layer_index"] == layer_index:
            head_index = ablation["head_index"]
            if head_index < 0 or head_index >= head_out.shape[1]:
                raise ValueError(f"Invalid head index {head_index} for layer {layer_index + 1}")
            head_out = head_out.clone()
            head_out[:, head_index, -1, :] = 0.0

        attn_out = attn.o_proj(attn.merge_heads(head_out))
        resid = resid + attn_out
        resid = resid + block.mlp(block.norm2(resid))

    final_hidden = model.norm_final(resid)
    return final_hidden @ model.token_embed.weight.T


def score_rows_with_optional_ablation(
    model: nn.Module,
    bundle: DatasetBundle,
    rows: list[dict],
    device: torch.device,
    ablation: dict | None = None,
) -> pd.DataFrame:
    if not rows:
        raise ValueError("Expected at least one row to score.")

    prompt_lengths = {len(row["prompt"].split()) for row in rows}
    if len(prompt_lengths) != 1:
        raise ValueError(f"Expected a fixed prompt length within the batch, found {sorted(prompt_lengths)}")

    input_ids = torch.stack([encode_prompt(row["prompt"], bundle.token_to_id) for row in rows]).to(device)
    target_ids = torch.tensor([bundle.token_to_id[row["target"]] for row in rows], dtype=torch.long)

    with torch.no_grad():
        logits = forward_with_head_ablation(model, input_ids, ablation=ablation)
        final_logits = logits[:, -1, :].detach().cpu()

    predicted_ids = final_logits.argmax(dim=-1)
    target_logits = final_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    competing_logits = final_logits.clone()
    competing_logits.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    foil_logits, foil_ids = competing_logits.max(dim=-1)

    records = []
    for row_index, row in enumerate(rows):
        predicted_token = decode_token(int(predicted_ids[row_index].item()), bundle.id_to_token)
        target_token = row["target"]
        records.append(
            {
                "prompt": row["prompt"],
                "query_key": row["query_key"],
                "predicted_token": predicted_token,
                "target_token": target_token,
                "foil_token": decode_token(int(foil_ids[row_index].item()), bundle.id_to_token),
                "target_logit": float(target_logits[row_index].item()),
                "foil_logit": float(foil_logits[row_index].item()),
                "margin": float((target_logits[row_index] - foil_logits[row_index]).item()),
                "correct": predicted_token == target_token,
            }
        )

    return pd.DataFrame(records)


def summarize_logits_against_target(
    final_logits: torch.Tensor,
    bundle: DatasetBundle,
    target_token: str,
    top_k: int = 5,
) -> dict:
    if target_token not in bundle.token_to_id:
        raise ValueError(f"Unknown target token: {target_token}")

    target_id = bundle.token_to_id[target_token]
    predicted_id = int(final_logits.argmax().item())
    competing_logits = final_logits.clone()
    competing_logits[target_id] = float("-inf")
    foil_id = int(competing_logits.argmax().item())
    top_logits = torch.topk(final_logits, k=top_k)

    return {
        "predicted_token": decode_token(predicted_id, bundle.id_to_token),
        "target_token": target_token,
        "foil_token": decode_token(foil_id, bundle.id_to_token),
        "target_logit": float(final_logits[target_id].item()),
        "foil_logit": float(final_logits[foil_id].item()),
        "margin": float(final_logits[target_id].item() - final_logits[foil_id].item()),
        "correct": decode_token(predicted_id, bundle.id_to_token) == target_token,
        "top_tokens": [decode_token(token_id, bundle.id_to_token) for token_id in top_logits.indices.tolist()],
        "top_logits": [float(value) for value in top_logits.values.tolist()],
    }


def forward_with_activation_patch(
    model: nn.Module,
    input_ids: torch.Tensor,
    clean_cache: dict,
    patch: dict | None = None,
) -> torch.Tensor:
    resid = model.token_embed(input_ids)

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

        if patch is not None and patch["kind"] == "head_out" and patch["layer_index"] == layer_index:
            head_index = patch["head_index"]
            if head_index < 0 or head_index >= head_out.shape[1]:
                raise ValueError(f"Invalid head index {head_index} for layer {layer_index + 1}")
            clean_head_out = clean_cache["blocks"][layer_index]["attention"]["head_out"].to(
                device=head_out.device,
                dtype=head_out.dtype,
            )
            head_out = head_out.clone()
            head_out[:, head_index, -1, :] = clean_head_out[:, head_index, -1, :]

        attn_out = attn.o_proj(attn.merge_heads(head_out))
        resid = resid + attn_out

        mlp_out = block.mlp(block.norm2(resid))
        if patch is not None and patch["kind"] == "mlp_out" and patch["layer_index"] == layer_index:
            clean_mlp_out = clean_cache["blocks"][layer_index]["mlp"]["out"].to(
                device=mlp_out.device,
                dtype=mlp_out.dtype,
            )
            mlp_out = mlp_out.clone()
            mlp_out[:, -1, :] = clean_mlp_out[:, -1, :]

        resid = resid + mlp_out

        if patch is not None and patch["kind"] == "resid_after_block" and patch["layer_index"] == layer_index:
            clean_resid = clean_cache["blocks"][layer_index]["resid_after_mlp"].to(
                device=resid.device,
                dtype=resid.dtype,
            )
            resid = resid.clone()
            resid[:, -1, :] = clean_resid[:, -1, :]

    final_hidden = model.norm_final(resid)
    return final_hidden @ model.token_embed.weight.T


def score_patched_prompt(
    model: nn.Module,
    bundle: DatasetBundle,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_target: str,
    device: torch.device,
    patch: dict | None = None,
    clean_cache: dict | None = None,
) -> dict:
    clean_ids = encode_prompt(clean_prompt, bundle.token_to_id).unsqueeze(0).to(device)
    corrupt_ids = encode_prompt(corrupt_prompt, bundle.token_to_id).unsqueeze(0).to(device)

    with torch.no_grad():
        if clean_cache is None:
            _, clean_cache = model(clean_ids, return_cache=True)
        logits = forward_with_activation_patch(model, corrupt_ids, clean_cache, patch=patch)
        final_logits = logits[0, -1].detach().cpu()

    result = summarize_logits_against_target(final_logits, bundle, clean_target)
    result["patch"] = "none" if patch is None else patch
    return result


def build_qk_table(
    prompt: str,
    cache: dict,
    layer_index: int,
    head_index: int,
    query_position: int = -1,
) -> pd.DataFrame:
    prompt_tokens = prompt.split()
    scores = cache["blocks"][layer_index]["attention"]["scores"][0, head_index, query_position, :].detach().cpu()
    pattern = cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, query_position, :].detach().cpu()

    records = []
    for position, token in enumerate(prompt_tokens):
        records.append(
            {
                "position": position,
                "token": token,
                "qk_score": float(scores[position].item()),
                "attention_weight": float(pattern[position].item()),
            }
        )
    return pd.DataFrame(records)


def ov_source_logits(
    model: nn.Module,
    cache: dict,
    layer_index: int,
    head_index: int,
    source_position: int,
) -> torch.Tensor:
    block = model.blocks[layer_index]
    attn = block.attn
    head_dim = attn.head_dim
    head_start = head_index * head_dim
    head_stop = head_start + head_dim

    source_state = cache["blocks"][layer_index]["attn_in"][0, source_position].to(
        device=attn.v_proj.weight.device,
        dtype=attn.v_proj.weight.dtype,
    )
    value_weight = attn.v_proj.weight[head_start:head_stop, :]
    output_weight = attn.o_proj.weight[:, head_start:head_stop]

    head_value = torch.matmul(value_weight, source_state)
    head_write = torch.matmul(output_weight, head_value)
    return torch.matmul(head_write, model.token_embed.weight.T).detach().cpu()


def build_ov_topk_table(
    model: nn.Module,
    bundle: DatasetBundle,
    cache: dict,
    layer_index: int,
    head_index: int,
    source_position: int,
    top_k: int = 5,
) -> pd.DataFrame:
    logits = ov_source_logits(model, cache, layer_index, head_index, source_position)
    top_logits = torch.topk(logits, k=top_k)
    return pd.DataFrame(
        {
            "token": [decode_token(token_id, bundle.id_to_token) for token_id in top_logits.indices.tolist()],
            "logit": [float(value) for value in top_logits.values.tolist()],
        }
    )


def head_residual_contribution(
    model: nn.Module,
    cache: dict,
    layer_index: int,
    head_index: int,
) -> torch.Tensor:
    block = model.blocks[layer_index]
    attn = block.attn
    if head_index < 0 or head_index >= attn.n_heads:
        raise ValueError(f"Invalid head index {head_index} for layer {layer_index + 1}")

    head_dim = attn.head_dim
    head_start = head_index * head_dim
    head_stop = head_start + head_dim
    head_out = cache["blocks"][layer_index]["attention"]["head_out"][:, head_index, :, :].to(
        device=attn.o_proj.weight.device,
        dtype=attn.o_proj.weight.dtype,
    )
    output_weight = attn.o_proj.weight[:, head_start:head_stop]
    return F.linear(head_out, output_weight)


def head_value_mix_to_write(
    model: nn.Module,
    layer_index: int,
    head_index: int,
    value_mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = model.blocks[layer_index]
    attn = block.attn
    if head_index < 0 or head_index >= attn.n_heads:
        raise ValueError(f"Invalid head index {head_index} for layer {layer_index + 1}")
    if value_mix.ndim != 1 or value_mix.shape[0] != attn.head_dim:
        raise ValueError(
            "Expected value_mix to match the head dimension: "
            f"expected {(attn.head_dim,)}, got {tuple(value_mix.shape)}"
        )

    head_start = head_index * attn.head_dim
    head_stop = head_start + attn.head_dim
    output_weight = attn.o_proj.weight[:, head_start:head_stop]
    head_write = F.linear(
        value_mix.to(device=output_weight.device, dtype=output_weight.dtype),
        output_weight,
    ).detach().cpu()
    logits = torch.matmul(head_write, model.token_embed.weight.T.detach().cpu())
    return head_write, logits


def build_head_source_write_table(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    attention_pattern: torch.Tensor,
    value_vectors: torch.Tensor,
    layer_index: int,
    head_index: int,
    target_token: str,
    foil_token: str,
) -> pd.DataFrame:
    prompt_tokens = prompt.split()
    if attention_pattern.ndim != 1:
        raise ValueError(f"Expected attention_pattern to be rank 1, got {tuple(attention_pattern.shape)}")
    if value_vectors.ndim != 2:
        raise ValueError(f"Expected value_vectors to be rank 2, got {tuple(value_vectors.shape)}")
    if attention_pattern.shape[0] != value_vectors.shape[0]:
        raise ValueError(
            "Attention/value length mismatch: "
            f"{attention_pattern.shape[0]} vs {value_vectors.shape[0]}"
        )
    if len(prompt_tokens) != attention_pattern.shape[0]:
        raise ValueError(
            f"Prompt length {len(prompt_tokens)} does not match attention length {attention_pattern.shape[0]}"
        )
    if target_token not in bundle.token_to_id:
        raise ValueError(f"Unknown target token: {target_token}")
    if foil_token not in bundle.token_to_id:
        raise ValueError(f"Unknown foil token: {foil_token}")

    target_id = bundle.token_to_id[target_token]
    foil_id = bundle.token_to_id[foil_token]
    rows = []
    for position, token in enumerate(prompt_tokens):
        source_value_mix = attention_pattern[position] * value_vectors[position]
        source_write, source_logits = head_value_mix_to_write(
            model=model,
            layer_index=layer_index,
            head_index=head_index,
            value_mix=source_value_mix.detach().cpu(),
        )
        rows.append(
            {
                "position": position,
                "token": token,
                "attention_weight": float(attention_pattern[position].item()),
                "value_norm": float(value_vectors[position].norm().item()),
                "mix_component_norm": float(source_value_mix.norm().item()),
                "write_norm": float(source_write.norm().item()),
                "target_logit": float(source_logits[target_id].item()),
                "foil_logit": float(source_logits[foil_id].item()),
                "target_minus_foil": float((source_logits[target_id] - source_logits[foil_id]).item()),
            }
        )
    return pd.DataFrame(rows)


def source_component_tensor(
    model: nn.Module,
    cache: dict,
    source_patch: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_layer_index = source_patch["layer_index"]
    if source_layer_index < 0 or source_layer_index >= len(model.blocks):
        raise ValueError(f"Invalid source layer index: {source_layer_index}")

    kind = source_patch["kind"]
    if kind == "resid_after_block":
        source_tensor = cache["blocks"][source_layer_index]["resid_after_mlp"]
    elif kind == "head_resid":
        source_tensor = head_residual_contribution(
            model,
            cache,
            layer_index=source_layer_index,
            head_index=source_patch["head_index"],
        )
    elif kind == "mlp_out":
        source_tensor = cache["blocks"][source_layer_index]["mlp"]["out"]
    else:
        raise ValueError(f"Unsupported source patch kind: {kind}")

    return source_tensor.to(device=device, dtype=dtype)


def forward_with_path_patch(
    model: nn.Module,
    input_ids: torch.Tensor,
    clean_cache: dict,
    corrupt_cache: dict,
    source_patch: dict,
    destination: dict,
) -> torch.Tensor:
    destination_layer_index = destination["layer_index"]
    destination_head_index = destination["head_index"]
    source_layer_index = source_patch["layer_index"]

    if destination_layer_index <= 0:
        raise ValueError("Path patching requires a destination head that is not in the first layer.")
    if source_layer_index != destination_layer_index - 1:
        raise ValueError(
            "This path patch helper only supports sources from the immediately previous layer "
            f"(got source layer {source_layer_index + 1} and destination layer {destination_layer_index + 1})."
        )

    resid = model.token_embed(input_ids)
    if destination_layer_index > 0:
        resid = corrupt_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
            device=resid.device,
            dtype=resid.dtype,
        )

    clean_source = source_component_tensor(model, clean_cache, source_patch, resid.device, resid.dtype)
    corrupt_source = source_component_tensor(model, corrupt_cache, source_patch, resid.device, resid.dtype)
    source_positions = source_patch.get("source_positions")
    if source_positions is not None:
        invalid_positions = [
            position
            for position in source_positions
            if position < 0 or position >= resid.shape[1]
        ]
        if invalid_positions:
            raise ValueError(f"Invalid source positions for path patching: {invalid_positions}")

    if source_patch["kind"] == "resid_after_block":
        if source_positions is None:
            patched_resid_for_destination = clean_source
        else:
            patched_resid_for_destination = resid.clone()
            for position in source_positions:
                patched_resid_for_destination[:, position, :] = clean_source[:, position, :]
    else:
        if source_positions is None:
            patched_resid_for_destination = resid - corrupt_source + clean_source
        else:
            source_delta = clean_source - corrupt_source
            position_mask = torch.zeros_like(source_delta)
            for position in source_positions:
                position_mask[:, position, :] = 1.0
            patched_resid_for_destination = resid + (source_delta * position_mask)

    block = model.blocks[destination_layer_index]
    attn = block.attn

    corrupt_head_out = corrupt_cache["blocks"][destination_layer_index]["attention"]["head_out"].to(
        device=resid.device,
        dtype=resid.dtype,
    )

    attn_in = block.norm1(patched_resid_for_destination)
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
    patched_head_out = torch.matmul(pattern, v)

    if destination_head_index < 0 or destination_head_index >= patched_head_out.shape[1]:
        raise ValueError(
            f"Invalid destination head index {destination_head_index} for layer {destination_layer_index + 1}"
        )

    mixed_head_out = corrupt_head_out.clone()
    mixed_head_out[:, destination_head_index, :, :] = patched_head_out[:, destination_head_index, :, :]

    attn_out = attn.o_proj(attn.merge_heads(mixed_head_out))
    resid = resid + attn_out
    resid = resid + block.mlp(block.norm2(resid))

    for layer_index in range(destination_layer_index + 1, len(model.blocks)):
        resid, _ = model.blocks[layer_index](resid, capture=False)

    final_hidden = model.norm_final(resid)
    return final_hidden @ model.token_embed.weight.T


def score_path_patched_prompt(
    model: nn.Module,
    bundle: DatasetBundle,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_target: str,
    device: torch.device,
    source_patch: dict,
    destination: dict,
    clean_cache: dict | None = None,
    corrupt_cache: dict | None = None,
) -> dict:
    clean_ids = encode_prompt(clean_prompt, bundle.token_to_id).unsqueeze(0).to(device)
    corrupt_ids = encode_prompt(corrupt_prompt, bundle.token_to_id).unsqueeze(0).to(device)

    with torch.no_grad():
        if clean_cache is None:
            _, clean_cache = model(clean_ids, return_cache=True)
        if corrupt_cache is None:
            _, corrupt_cache = model(corrupt_ids, return_cache=True)
        logits = forward_with_path_patch(
            model,
            corrupt_ids,
            clean_cache,
            corrupt_cache,
            source_patch=source_patch,
            destination=destination,
        )
        final_logits = logits[0, -1].detach().cpu()

    result = summarize_logits_against_target(final_logits, bundle, clean_target)
    result["source_patch"] = source_patch
    result["destination"] = destination
    return result


def forward_with_qkv_patch(
    model: nn.Module,
    input_ids: torch.Tensor,
    clean_cache: dict,
    corrupt_cache: dict,
    destination: dict,
    components: list[str],
) -> torch.Tensor:
    destination_layer_index = destination["layer_index"]
    destination_head_index = destination["head_index"]
    component_set = set(components)

    if not component_set:
        raise ValueError("Expected at least one Q/K/V component to patch.")
    invalid_components = sorted(component_set - {"q", "k", "v"})
    if invalid_components:
        raise ValueError(f"Invalid Q/K/V patch components: {invalid_components}")
    if destination_layer_index < 0 or destination_layer_index >= len(model.blocks):
        raise ValueError(f"Invalid destination layer index {destination_layer_index}")

    if destination_layer_index == 0:
        resid = model.token_embed(input_ids)
        clean_resid = clean_cache["token_embed"].to(
            device=resid.device,
            dtype=resid.dtype,
        )
    else:
        resid = corrupt_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
            device=model.token_embed.weight.device,
            dtype=model.token_embed.weight.dtype,
        )
        clean_resid = clean_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
            device=resid.device,
            dtype=resid.dtype,
        )

    block = model.blocks[destination_layer_index]
    attn = block.attn
    final_position_index = resid.shape[1] - 1

    corrupt_attn_in = block.norm1(resid)
    clean_attn_in = block.norm1(clean_resid)

    q = attn.split_heads(attn.q_proj(corrupt_attn_in))
    k = attn.split_heads(attn.k_proj(corrupt_attn_in))
    v = attn.split_heads(attn.v_proj(corrupt_attn_in))
    clean_q = attn.split_heads(attn.q_proj(clean_attn_in))
    clean_k = attn.split_heads(attn.k_proj(clean_attn_in))
    clean_v = attn.split_heads(attn.v_proj(clean_attn_in))

    if destination_head_index < 0 or destination_head_index >= q.shape[1]:
        raise ValueError(
            f"Invalid destination head index {destination_head_index} for layer {destination_layer_index + 1}"
        )

    if "q" in component_set:
        q = q.clone()
        q[:, destination_head_index, final_position_index, :] = clean_q[
            :, destination_head_index, final_position_index, :
        ]
    if "k" in component_set:
        k = k.clone()
        k[:, destination_head_index, final_position_index, :] = clean_k[
            :, destination_head_index, final_position_index, :
        ]
    if "v" in component_set:
        v = v.clone()
        v[:, destination_head_index, final_position_index, :] = clean_v[
            :, destination_head_index, final_position_index, :
        ]

    _, seq_len, _ = corrupt_attn_in.shape
    cos = attn.rope_cos[:seq_len].to(device=corrupt_attn_in.device, dtype=corrupt_attn_in.dtype).unsqueeze(0).unsqueeze(0)
    sin = attn.rope_sin[:seq_len].to(device=corrupt_attn_in.device, dtype=corrupt_attn_in.dtype).unsqueeze(0).unsqueeze(0)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=corrupt_attn_in.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    pattern = scores.softmax(dim=-1)
    patched_head_out = torch.matmul(pattern, v)

    mixed_head_out = corrupt_cache["blocks"][destination_layer_index]["attention"]["head_out"].to(
        device=resid.device,
        dtype=resid.dtype,
    ).clone()
    mixed_head_out[:, destination_head_index, :, :] = patched_head_out[:, destination_head_index, :, :]

    attn_out = attn.o_proj(attn.merge_heads(mixed_head_out))
    resid = resid + attn_out
    resid = resid + block.mlp(block.norm2(resid))

    for layer_index in range(destination_layer_index + 1, len(model.blocks)):
        resid, _ = model.blocks[layer_index](resid, capture=False)

    final_hidden = model.norm_final(resid)
    return final_hidden @ model.token_embed.weight.T


def score_qkv_patched_prompt(
    model: nn.Module,
    bundle: DatasetBundle,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_target: str,
    device: torch.device,
    destination: dict,
    components: list[str],
    clean_cache: dict | None = None,
    corrupt_cache: dict | None = None,
) -> dict:
    clean_ids = encode_prompt(clean_prompt, bundle.token_to_id).unsqueeze(0).to(device)
    corrupt_ids = encode_prompt(corrupt_prompt, bundle.token_to_id).unsqueeze(0).to(device)

    with torch.no_grad():
        if clean_cache is None:
            _, clean_cache = model(clean_ids, return_cache=True)
        if corrupt_cache is None:
            _, corrupt_cache = model(corrupt_ids, return_cache=True)
        logits = forward_with_qkv_patch(
            model,
            corrupt_ids,
            clean_cache,
            corrupt_cache,
            destination=destination,
            components=components,
        )
        final_logits = logits[0, -1].detach().cpu()

    result = summarize_logits_against_target(final_logits, bundle, clean_target)
    result["destination"] = destination
    result["components"] = sorted(component_set) if (component_set := set(components)) else []
    return result


def build_head_source_contribution_table(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    cache: dict,
    layer_index: int,
    head_index: int,
    target_token: str,
    foil_token: str,
    query_position: int = -1,
) -> pd.DataFrame:
    if target_token not in bundle.token_to_id:
        raise ValueError(f"Unknown target token: {target_token}")
    if foil_token not in bundle.token_to_id:
        raise ValueError(f"Unknown foil token: {foil_token}")

    prompt_tokens = prompt.split()
    attention = cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, query_position, :].detach().cpu()
    target_id = bundle.token_to_id[target_token]
    foil_id = bundle.token_to_id[foil_token]

    rows = []
    for source_position, source_token in enumerate(prompt_tokens):
        source_logits = ov_source_logits(model, cache, layer_index, head_index, source_position)
        top_token_id = int(source_logits.argmax().item())
        ov_target = float(source_logits[target_id].item())
        ov_foil = float(source_logits[foil_id].item())
        ov_margin = ov_target - ov_foil
        attention_weight = float(attention[source_position].item())
        rows.append(
            {
                "source_position": source_position,
                "source_token": source_token,
                "attention_weight": attention_weight,
                "top_written_token": decode_token(top_token_id, bundle.id_to_token),
                "top_written_logit": float(source_logits[top_token_id].item()),
                "ov_target_logit": ov_target,
                "ov_foil_logit": ov_foil,
                "ov_target_minus_foil": ov_margin,
                "weighted_target_minus_foil": attention_weight * ov_margin,
            }
        )

    return pd.DataFrame(rows)


def build_qkv_patched_attention_table(
    model: nn.Module,
    prompt: str,
    clean_cache: dict,
    corrupt_cache: dict,
    destination: dict,
    components: list[str],
    query_position: int = -1,
) -> pd.DataFrame:
    destination_layer_index = destination["layer_index"]
    destination_head_index = destination["head_index"]
    component_set = set(components)

    if not component_set:
        raise ValueError("Expected at least one Q/K/V component to patch.")
    invalid_components = sorted(component_set - {"q", "k", "v"})
    if invalid_components:
        raise ValueError(f"Invalid Q/K/V patch components: {invalid_components}")
    prompt_tokens = prompt.split()
    if destination_layer_index < 0 or destination_layer_index >= len(model.blocks):
        raise ValueError(f"Invalid destination layer index {destination_layer_index}")
    if destination_layer_index == 0:
        corrupt_resid = corrupt_cache["token_embed"].to(
            device=model.token_embed.weight.device,
            dtype=model.token_embed.weight.dtype,
        )
        clean_resid = clean_cache["token_embed"].to(
            device=corrupt_resid.device,
            dtype=corrupt_resid.dtype,
        )
    else:
        corrupt_resid = corrupt_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
            device=model.token_embed.weight.device,
            dtype=model.token_embed.weight.dtype,
        )
        clean_resid = clean_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
            device=corrupt_resid.device,
            dtype=corrupt_resid.dtype,
        )

    block = model.blocks[destination_layer_index]
    attn = block.attn
    final_position_index = corrupt_resid.shape[1] - 1

    corrupt_attn_in = block.norm1(corrupt_resid)
    clean_attn_in = block.norm1(clean_resid)

    q = attn.split_heads(attn.q_proj(corrupt_attn_in))
    k = attn.split_heads(attn.k_proj(corrupt_attn_in))
    v = attn.split_heads(attn.v_proj(corrupt_attn_in))
    clean_q = attn.split_heads(attn.q_proj(clean_attn_in))
    clean_k = attn.split_heads(attn.k_proj(clean_attn_in))
    clean_v = attn.split_heads(attn.v_proj(clean_attn_in))

    if "q" in component_set:
        q = q.clone()
        q[:, destination_head_index, final_position_index, :] = clean_q[
            :, destination_head_index, final_position_index, :
        ]
    if "k" in component_set:
        k = k.clone()
        k[:, destination_head_index, final_position_index, :] = clean_k[
            :, destination_head_index, final_position_index, :
        ]
    if "v" in component_set:
        v = v.clone()
        v[:, destination_head_index, final_position_index, :] = clean_v[
            :, destination_head_index, final_position_index, :
        ]

    _, seq_len, _ = corrupt_attn_in.shape
    cos = attn.rope_cos[:seq_len].to(device=corrupt_attn_in.device, dtype=corrupt_attn_in.dtype).unsqueeze(0).unsqueeze(0)
    sin = attn.rope_sin[:seq_len].to(device=corrupt_attn_in.device, dtype=corrupt_attn_in.dtype).unsqueeze(0).unsqueeze(0)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=corrupt_attn_in.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    pattern = scores.softmax(dim=-1)

    head_scores = scores[0, destination_head_index, query_position, :].detach().cpu()
    head_pattern = pattern[0, destination_head_index, query_position, :].detach().cpu()
    rows = []
    for source_position, source_token in enumerate(prompt_tokens):
        rows.append(
            {
                "source_position": source_position,
                "source_token": source_token,
                "qk_score": float(head_scores[source_position].item()),
                "attention_weight": float(head_pattern[source_position].item()),
            }
        )

    return pd.DataFrame(rows)


def compute_path_patched_head_details(
    model: nn.Module,
    clean_cache: dict,
    corrupt_cache: dict,
    source_patch: dict,
    destination: dict,
) -> dict:
    destination_layer_index = destination["layer_index"]
    destination_head_index = destination["head_index"]
    source_layer_index = source_patch["layer_index"]

    if destination_layer_index <= 0:
        raise ValueError("Path patching requires a destination head that is not in the first layer.")
    if source_layer_index != destination_layer_index - 1:
        raise ValueError(
            "This helper only supports sources from the immediately previous layer "
            f"(got source layer {source_layer_index + 1} and destination layer {destination_layer_index + 1})."
        )

    resid = corrupt_cache["blocks"][destination_layer_index - 1]["resid_after_mlp"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    clean_source = source_component_tensor(model, clean_cache, source_patch, resid.device, resid.dtype)
    corrupt_source = source_component_tensor(model, corrupt_cache, source_patch, resid.device, resid.dtype)
    source_positions = source_patch.get("source_positions")
    if source_positions is not None:
        invalid_positions = [
            position
            for position in source_positions
            if position < 0 or position >= resid.shape[1]
        ]
        if invalid_positions:
            raise ValueError(f"Invalid source positions for path patching: {invalid_positions}")

    if source_patch["kind"] == "resid_after_block":
        if source_positions is None:
            patched_resid_for_destination = clean_source
        else:
            patched_resid_for_destination = resid.clone()
            for position in source_positions:
                patched_resid_for_destination[:, position, :] = clean_source[:, position, :]
    else:
        if source_positions is None:
            patched_resid_for_destination = resid - corrupt_source + clean_source
        else:
            source_delta = clean_source - corrupt_source
            position_mask = torch.zeros_like(source_delta)
            for position in source_positions:
                position_mask[:, position, :] = 1.0
            patched_resid_for_destination = resid + (source_delta * position_mask)

    block = model.blocks[destination_layer_index]
    attn = block.attn
    attn_in = block.norm1(patched_resid_for_destination)
    q = attn.split_heads(attn.q_proj(attn_in))
    k = attn.split_heads(attn.k_proj(attn_in))
    v = attn.split_heads(attn.v_proj(attn_in))

    _, seq_len, _ = attn_in.shape
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

    return {
        "q": q.detach().cpu(),
        "k": k.detach().cpu(),
        "v": v.detach().cpu(),
        "scores": scores.detach().cpu(),
        "pattern": pattern.detach().cpu(),
    }


def build_path_patched_attention_table(
    model: nn.Module,
    prompt: str,
    clean_cache: dict,
    corrupt_cache: dict,
    source_patch: dict,
    destination: dict,
    query_position: int = -1,
) -> pd.DataFrame:
    prompt_tokens = prompt.split()
    destination_head_index = destination["head_index"]
    details = compute_path_patched_head_details(
        model,
        clean_cache,
        corrupt_cache,
        source_patch=source_patch,
        destination=destination,
    )

    scores = details["scores"][0, destination_head_index, query_position, :].detach().cpu()
    pattern = details["pattern"][0, destination_head_index, query_position, :].detach().cpu()

    rows = []
    for source_position, source_token in enumerate(prompt_tokens):
        rows.append(
            {
                "source_position": source_position,
                "source_token": source_token,
                "qk_score": float(scores[source_position].item()),
                "attention_weight": float(pattern[source_position].item()),
            }
        )

    return pd.DataFrame(rows)


def residual_vector_to_logits(model: nn.Module, residual_vector: torch.Tensor) -> torch.Tensor:
    if residual_vector.ndim != 1:
        raise ValueError(
            f"Expected a 1D residual vector, got shape {tuple(residual_vector.shape)}"
        )

    device = model.token_embed.weight.device
    dtype = model.token_embed.weight.dtype
    residual_vector = residual_vector.to(device=device, dtype=dtype)
    normalized = model.norm_final(residual_vector.view(1, 1, -1))[0, 0]
    logits = torch.matmul(normalized, model.token_embed.weight.T)
    return logits.detach().cpu()


def build_layer_feature_readout_table(
    model: nn.Module,
    bundle: DatasetBundle,
    cache: dict,
    target_token: str,
    foil_token: str,
    position: int = -1,
    top_k: int = 5,
) -> pd.DataFrame:
    stage_specs = [
        ("L1 resid_in", cache["blocks"][0]["resid_in"]),
        ("L1 after attn", cache["blocks"][0]["resid_after_attn"]),
        ("L1 after mlp", cache["blocks"][0]["resid_after_mlp"]),
        ("L2 resid_in", cache["blocks"][1]["resid_in"]),
        ("L2 after attn", cache["blocks"][1]["resid_after_attn"]),
        ("L2 after mlp", cache["blocks"][1]["resid_after_mlp"]),
    ]

    target_id = bundle.token_to_id[target_token]
    foil_id = bundle.token_to_id[foil_token]
    rows = []
    for stage_name, stage_tensor in stage_specs:
        residual_vector = stage_tensor[0, position].detach().cpu()
        logits = residual_vector_to_logits(model, residual_vector)
        top_logits, top_indices = torch.topk(logits, k=min(top_k, logits.shape[0]))
        rows.append(
            {
                "stage": stage_name,
                "top_token": bundle.id_to_token[int(top_indices[0].item())],
                "top_logit": float(top_logits[0].item()),
                "target_logit": float(logits[target_id].item()),
                "foil_logit": float(logits[foil_id].item()),
                "target_minus_foil": float((logits[target_id] - logits[foil_id]).item()),
                "top_k_tokens": [
                    bundle.id_to_token[int(idx.item())] for idx in top_indices
                ],
                "top_k_logits": [float(value.item()) for value in top_logits],
            }
        )

    return pd.DataFrame(rows)


def top_token_rows(
    logits: torch.Tensor,
    bundle: DatasetBundle,
    top_k: int = 5,
) -> list[dict[str, object]]:
    if logits.ndim != 1:
        raise ValueError(f"Expected logits to be rank 1, got {tuple(logits.shape)}")
    top_logits, top_indices = torch.topk(logits, k=min(top_k, logits.shape[0]))
    return [
        {
            "token": bundle.id_to_token[int(index.item())],
            "logit": float(value.item()),
        }
        for value, index in zip(top_logits, top_indices)
    ]


def forward_with_mlp_neuron_ablation(
    model: nn.Module,
    input_ids: torch.Tensor,
    ablation: dict | None = None,
) -> torch.Tensor:
    resid = model.token_embed(input_ids)
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

        attn_out = attn.o_proj(attn.merge_heads(head_out))
        resid = resid + attn_out

        mlp_in = block.norm2(resid)
        gate = block.mlp.gate_proj(mlp_in)
        up = block.mlp.up_proj(mlp_in)
        activated = F.silu(gate) * up

        if ablation is not None and ablation["layer_index"] == layer_index:
            position_index = int(ablation.get("position_index", activated.shape[1] - 1))
            if position_index < 0:
                position_index += activated.shape[1]
            if position_index < 0 or position_index >= activated.shape[1]:
                raise ValueError(
                    f"Invalid MLP ablation position {position_index} for layer {layer_index + 1}"
                )
            activated = activated.clone()
            neuron_index = ablation.get("neuron_index")
            if neuron_index is None:
                activated[:, position_index, :] = 0.0
            else:
                neuron_index = int(neuron_index)
                if neuron_index < 0 or neuron_index >= activated.shape[-1]:
                    raise ValueError(
                        f"Invalid neuron index {neuron_index} for layer {layer_index + 1}"
                    )
                activated[:, position_index, neuron_index] = 0.0

        mlp_out = block.mlp.down_proj(activated)
        resid = resid + mlp_out

    final_hidden = model.norm_final(resid)
    return final_hidden @ model.token_embed.weight.T


def score_mlp_neuron_ablation_prompt(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    target_token: str,
    device: torch.device,
    layer_index: int,
    neuron_index: int | None = None,
    position_index: int = -1,
) -> dict:
    input_ids = encode_prompt(prompt, bundle.token_to_id).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = forward_with_mlp_neuron_ablation(
            model,
            input_ids,
            ablation={
                "layer_index": layer_index,
                "neuron_index": neuron_index,
                "position_index": position_index,
            },
        )
        final_logits = logits[0, -1].detach().cpu()
    result = summarize_logits_against_target(final_logits, bundle, target_token)
    result["layer_index"] = layer_index
    result["neuron_index"] = neuron_index
    result["position_index"] = position_index
    return result


def build_mlp_neuron_contribution_table(
    model: nn.Module,
    bundle: DatasetBundle,
    prompt: str,
    cache: dict,
    layer_index: int,
    target_token: str,
    foil_token: str,
    device: torch.device,
    position: int = -1,
    top_k: int = 5,
    include_exact_ablation: bool = True,
) -> pd.DataFrame:
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for MLP neuron table: {layer_index}")
    if target_token not in bundle.token_to_id:
        raise ValueError(f"Unknown target token: {target_token}")
    if foil_token not in bundle.token_to_id:
        raise ValueError(f"Unknown foil token: {foil_token}")

    mlp_cache = cache["blocks"][layer_index]["mlp"]
    gate = mlp_cache["gate"][0, position, :].detach().cpu()
    up = mlp_cache["up"][0, position, :].detach().cpu()
    activated = mlp_cache["activated"][0, position, :].detach().cpu()
    down_weight = model.blocks[layer_index].mlp.down_proj.weight.detach().cpu()

    baseline_result, _ = run_prompt(
        model,
        bundle,
        prompt,
        device=device,
        expected_target=target_token,
        return_cache=False,
    )

    target_id = bundle.token_to_id[target_token]
    foil_id = bundle.token_to_id[foil_token]
    rows = []
    for neuron_index in range(activated.shape[0]):
        write_vector = down_weight[:, neuron_index] * activated[neuron_index]
        standalone_logits = residual_vector_to_logits(model, write_vector)
        standalone_target_logit = float(standalone_logits[target_id].item())
        standalone_foil_logit = float(standalone_logits[foil_id].item())
        neuron_row: dict[str, object] = {
            "neuron_index": neuron_index,
            "gate": float(gate[neuron_index].item()),
            "up": float(up[neuron_index].item()),
            "activated": float(activated[neuron_index].item()),
            "write_norm": float(write_vector.norm().item()),
            "standalone_target_logit": standalone_target_logit,
            "standalone_foil_logit": standalone_foil_logit,
            "standalone_target_minus_foil": standalone_target_logit - standalone_foil_logit,
            "top_token": bundle.id_to_token[int(standalone_logits.argmax().item())],
            "top_k_tokens": top_token_rows(standalone_logits, bundle, top_k=top_k),
        }
        if include_exact_ablation:
            ablated_result = score_mlp_neuron_ablation_prompt(
                model=model,
                bundle=bundle,
                prompt=prompt,
                target_token=target_token,
                device=device,
                layer_index=layer_index,
                neuron_index=neuron_index,
                position_index=position,
            )
            neuron_row.update(
                {
                    "exact_ablation_predicted_token": ablated_result["predicted_token"],
                    "exact_ablation_margin": ablated_result["margin"],
                    "exact_ablation_margin_drop": baseline_result["margin"] - ablated_result["margin"],
                    "exact_ablation_correct": ablated_result["correct"],
                }
            )
        rows.append(neuron_row)

    return pd.DataFrame(rows)


def _mlp_neuron_column_name(neuron_index: int) -> str:
    return f"neuron_{int(neuron_index)}_activation"


def compute_mlp_from_resid_after_attn(
    model: nn.Module,
    resid_after_attn: torch.Tensor,
    layer_index: int,
) -> dict[str, torch.Tensor]:
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for MLP recompute: {layer_index}")
    if resid_after_attn.ndim != 3:
        raise ValueError(
            f"Expected resid_after_attn to be rank 3, got {tuple(resid_after_attn.shape)}"
        )
    if resid_after_attn.shape[0] != 1:
        raise ValueError(
            "MLP recompute helper currently expects batch size 1, "
            f"got {resid_after_attn.shape[0]}"
        )

    block = model.blocks[layer_index]
    resid_after_attn = resid_after_attn.to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    mlp_in = block.norm2(resid_after_attn)
    gate = block.mlp.gate_proj(mlp_in)
    up = block.mlp.up_proj(mlp_in)
    activated = F.silu(gate) * up
    mlp_out = block.mlp.down_proj(activated)
    resid_after_mlp = resid_after_attn + mlp_out
    return {
        "mlp_in": mlp_in.detach().cpu(),
        "gate": gate.detach().cpu(),
        "up": up.detach().cpu(),
        "activated": activated.detach().cpu(),
        "out": mlp_out.detach().cpu(),
        "resid_after_mlp": resid_after_mlp.detach().cpu(),
    }


def build_mlp_neuron_read_comparison_table(
    model: nn.Module,
    clean_cache: dict,
    corrupt_cache: dict,
    layer_index: int,
    neuron_index: int,
    position: int = -1,
) -> pd.DataFrame:
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for neuron read comparison table: {layer_index}")

    hidden_dim = model.blocks[layer_index].mlp.down_proj.weight.shape[1]
    neuron_index = int(neuron_index)
    if neuron_index < 0 or neuron_index >= hidden_dim:
        raise ValueError(
            f"Invalid neuron index {neuron_index} for layer {layer_index + 1}"
        )

    clean_mlp_in = clean_cache["blocks"][layer_index]["mlp_in"][0, position, :].detach().cpu()
    corrupt_mlp_in = corrupt_cache["blocks"][layer_index]["mlp_in"][0, position, :].detach().cpu()
    if clean_mlp_in.shape != corrupt_mlp_in.shape:
        raise ValueError(
            "Clean/corrupt mlp_in shape mismatch: "
            f"{tuple(clean_mlp_in.shape)} vs {tuple(corrupt_mlp_in.shape)}"
        )

    gate_weight = model.blocks[layer_index].mlp.gate_proj.weight[neuron_index].detach().cpu()
    up_weight = model.blocks[layer_index].mlp.up_proj.weight[neuron_index].detach().cpu()
    clean_gate_contribution = clean_mlp_in * gate_weight
    corrupt_gate_contribution = corrupt_mlp_in * gate_weight
    clean_up_contribution = clean_mlp_in * up_weight
    corrupt_up_contribution = corrupt_mlp_in * up_weight

    rows: list[dict[str, object]] = []
    for input_dim in range(clean_mlp_in.shape[0]):
        gate_delta = clean_gate_contribution[input_dim] - corrupt_gate_contribution[input_dim]
        up_delta = clean_up_contribution[input_dim] - corrupt_up_contribution[input_dim]
        rows.append(
            {
                "input_dim": input_dim,
                "clean_x_value": float(clean_mlp_in[input_dim].item()),
                "corrupt_x_value": float(corrupt_mlp_in[input_dim].item()),
                "x_delta": float((clean_mlp_in[input_dim] - corrupt_mlp_in[input_dim]).item()),
                "gate_weight": float(gate_weight[input_dim].item()),
                "clean_gate_contribution": float(clean_gate_contribution[input_dim].item()),
                "corrupt_gate_contribution": float(corrupt_gate_contribution[input_dim].item()),
                "gate_contribution_delta": float(gate_delta.item()),
                "up_weight": float(up_weight[input_dim].item()),
                "clean_up_contribution": float(clean_up_contribution[input_dim].item()),
                "corrupt_up_contribution": float(corrupt_up_contribution[input_dim].item()),
                "up_contribution_delta": float(up_delta.item()),
                "combined_abs_delta": float(gate_delta.abs().item() + up_delta.abs().item()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "combined_abs_delta",
        ascending=False,
    ).reset_index(drop=True)


def build_mlp_neuron_upstream_head_effect_table(
    model: nn.Module,
    prompt: str,
    cache: dict,
    layer_index: int,
    neuron_indices: list[int],
    position: int = -1,
) -> pd.DataFrame:
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for neuron upstream head table: {layer_index}")
    if not neuron_indices:
        raise ValueError("Expected at least one neuron index for upstream head table")

    attn = model.blocks[layer_index].attn
    normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
    invalid_neuron_indices = [
        neuron_index
        for neuron_index in normalized_neuron_indices
        if neuron_index < 0 or neuron_index >= model.blocks[layer_index].mlp.down_proj.weight.shape[1]
    ]
    if invalid_neuron_indices:
        raise ValueError(
            f"Invalid neuron indices for layer {layer_index + 1}: {invalid_neuron_indices}"
        )

    prompt_tokens = prompt.split()
    resid_after_attn = cache["blocks"][layer_index]["resid_after_attn"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    baseline_gate = cache["blocks"][layer_index]["mlp"]["gate"][0, position, :].detach().cpu()
    baseline_up = cache["blocks"][layer_index]["mlp"]["up"][0, position, :].detach().cpu()
    baseline_activated = cache["blocks"][layer_index]["mlp"]["activated"][0, position, :].detach().cpu()
    attention_pattern = cache["blocks"][layer_index]["attention"]["pattern"][0, :, position, :].detach().cpu()

    rows: list[dict[str, object]] = []
    for head_index in range(attn.n_heads):
        head_write = head_residual_contribution(
            model=model,
            cache=cache,
            layer_index=layer_index,
            head_index=head_index,
        )[0, position].detach().cpu()
        modified_resid = resid_after_attn.clone()
        modified_resid[0, position, :] = modified_resid[0, position, :] - head_write.to(
            device=modified_resid.device,
            dtype=modified_resid.dtype,
        )
        modified_mlp = compute_mlp_from_resid_after_attn(
            model=model,
            resid_after_attn=modified_resid,
            layer_index=layer_index,
        )
        modified_gate = modified_mlp["gate"][0, position, :]
        modified_up = modified_mlp["up"][0, position, :]
        modified_activated = modified_mlp["activated"][0, position, :]
        top_source_position = int(attention_pattern[head_index].argmax().item())
        top_source_weight = float(attention_pattern[head_index, top_source_position].item())
        top_source_token = prompt_tokens[top_source_position]

        for neuron_index in normalized_neuron_indices:
            baseline_value = float(baseline_activated[neuron_index].item())
            modified_value = float(modified_activated[neuron_index].item())
            rows.append(
                {
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "neuron_index": neuron_index,
                    "baseline_gate": float(baseline_gate[neuron_index].item()),
                    "head_ablated_gate": float(modified_gate[neuron_index].item()),
                    "gate_delta": float(
                        (modified_gate[neuron_index] - baseline_gate[neuron_index]).item()
                    ),
                    "baseline_up": float(baseline_up[neuron_index].item()),
                    "head_ablated_up": float(modified_up[neuron_index].item()),
                    "up_delta": float(
                        (modified_up[neuron_index] - baseline_up[neuron_index]).item()
                    ),
                    "baseline_activation": baseline_value,
                    "head_ablated_activation": modified_value,
                    "activation_delta": modified_value - baseline_value,
                    "abs_activation_delta": abs(modified_value - baseline_value),
                    "head_write_norm": float(head_write.norm().item()),
                    "top_attention_source_position": top_source_position,
                    "top_attention_source_token": top_source_token,
                    "top_attention_weight": top_source_weight,
                }
            )

    return pd.DataFrame(rows)


def build_mlp_neuron_clean_corrupt_head_patch_table(
    model: nn.Module,
    clean_prompt: str,
    clean_cache: dict,
    corrupt_prompt: str,
    corrupt_cache: dict,
    layer_index: int,
    neuron_indices: list[int],
    position: int = -1,
) -> pd.DataFrame:
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for clean/corrupt head patch table: {layer_index}")
    if not neuron_indices:
        raise ValueError("Expected at least one neuron index for clean/corrupt head patch table")

    clean_tokens = clean_prompt.split()
    corrupt_tokens = corrupt_prompt.split()
    if len(clean_tokens) != len(corrupt_tokens):
        raise ValueError(
            "Clean/corrupt prompts must have the same length for head patch comparison: "
            f"{len(clean_tokens)} vs {len(corrupt_tokens)}"
        )

    attn = model.blocks[layer_index].attn
    normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
    invalid_neuron_indices = [
        neuron_index
        for neuron_index in normalized_neuron_indices
        if neuron_index < 0 or neuron_index >= model.blocks[layer_index].mlp.down_proj.weight.shape[1]
    ]
    if invalid_neuron_indices:
        raise ValueError(
            f"Invalid neuron indices for layer {layer_index + 1}: {invalid_neuron_indices}"
        )

    clean_resid_after_attn = clean_cache["blocks"][layer_index]["resid_after_attn"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    corrupt_resid_after_attn = corrupt_cache["blocks"][layer_index]["resid_after_attn"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    clean_gate = clean_cache["blocks"][layer_index]["mlp"]["gate"][0, position, :].detach().cpu()
    clean_up = clean_cache["blocks"][layer_index]["mlp"]["up"][0, position, :].detach().cpu()
    clean_activated = clean_cache["blocks"][layer_index]["mlp"]["activated"][0, position, :].detach().cpu()
    corrupt_gate = corrupt_cache["blocks"][layer_index]["mlp"]["gate"][0, position, :].detach().cpu()
    corrupt_up = corrupt_cache["blocks"][layer_index]["mlp"]["up"][0, position, :].detach().cpu()
    corrupt_activated = corrupt_cache["blocks"][layer_index]["mlp"]["activated"][0, position, :].detach().cpu()

    rows: list[dict[str, object]] = []
    for head_index in range(attn.n_heads):
        clean_head_write = head_residual_contribution(
            model=model,
            cache=clean_cache,
            layer_index=layer_index,
            head_index=head_index,
        )[0, position].detach().cpu()
        corrupt_head_write = head_residual_contribution(
            model=model,
            cache=corrupt_cache,
            layer_index=layer_index,
            head_index=head_index,
        )[0, position].detach().cpu()
        modified_resid = corrupt_resid_after_attn.clone()
        modified_resid[0, position, :] = modified_resid[0, position, :] + (
            clean_head_write - corrupt_head_write
        ).to(device=modified_resid.device, dtype=modified_resid.dtype)
        modified_mlp = compute_mlp_from_resid_after_attn(
            model=model,
            resid_after_attn=modified_resid,
            layer_index=layer_index,
        )
        patched_gate = modified_mlp["gate"][0, position, :]
        patched_up = modified_mlp["up"][0, position, :]
        patched_activated = modified_mlp["activated"][0, position, :]

        for neuron_index in normalized_neuron_indices:
            clean_value = float(clean_activated[neuron_index].item())
            corrupt_value = float(corrupt_activated[neuron_index].item())
            patched_value = float(patched_activated[neuron_index].item())
            rows.append(
                {
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "neuron_index": neuron_index,
                    "clean_gate": float(clean_gate[neuron_index].item()),
                    "corrupt_gate": float(corrupt_gate[neuron_index].item()),
                    "patched_corrupt_gate": float(patched_gate[neuron_index].item()),
                    "gate_patch_delta": float(
                        (patched_gate[neuron_index] - corrupt_gate[neuron_index]).item()
                    ),
                    "clean_up": float(clean_up[neuron_index].item()),
                    "corrupt_up": float(corrupt_up[neuron_index].item()),
                    "patched_corrupt_up": float(patched_up[neuron_index].item()),
                    "up_patch_delta": float(
                        (patched_up[neuron_index] - corrupt_up[neuron_index]).item()
                    ),
                    "clean_activation": clean_value,
                    "corrupt_activation": corrupt_value,
                    "patched_corrupt_activation": patched_value,
                    "patch_delta": patched_value - corrupt_value,
                    "distance_to_clean_before": abs(corrupt_value - clean_value),
                    "distance_to_clean_after": abs(patched_value - clean_value),
                    "recovery_toward_clean": abs(corrupt_value - clean_value) - abs(patched_value - clean_value),
                }
            )

    return pd.DataFrame(rows)


def build_mlp_neuron_clean_corrupt_source_patch_table(
    model: nn.Module,
    clean_prompt: str,
    clean_cache: dict,
    corrupt_prompt: str,
    corrupt_cache: dict,
    layer_index: int,
    head_index: int,
    neuron_indices: list[int],
    position: int = -1,
) -> pd.DataFrame:
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for clean/corrupt source patch table: {layer_index}")
    if not neuron_indices:
        raise ValueError("Expected at least one neuron index for clean/corrupt source patch table")

    clean_tokens = clean_prompt.split()
    corrupt_tokens = corrupt_prompt.split()
    if len(clean_tokens) != len(corrupt_tokens):
        raise ValueError(
            "Clean/corrupt prompts must have the same length for source patch comparison: "
            f"{len(clean_tokens)} vs {len(corrupt_tokens)}"
        )

    normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
    invalid_neuron_indices = [
        neuron_index
        for neuron_index in normalized_neuron_indices
        if neuron_index < 0 or neuron_index >= model.blocks[layer_index].mlp.down_proj.weight.shape[1]
    ]
    if invalid_neuron_indices:
        raise ValueError(
            f"Invalid neuron indices for layer {layer_index + 1}: {invalid_neuron_indices}"
        )

    clean_pattern = clean_cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, position, :].detach().cpu()
    corrupt_pattern = corrupt_cache["blocks"][layer_index]["attention"]["pattern"][0, head_index, position, :].detach().cpu()
    clean_values = clean_cache["blocks"][layer_index]["attention"]["v"][0, head_index, :, :].detach().cpu()
    corrupt_values = corrupt_cache["blocks"][layer_index]["attention"]["v"][0, head_index, :, :].detach().cpu()
    corrupt_resid_after_attn = corrupt_cache["blocks"][layer_index]["resid_after_attn"].to(
        device=model.token_embed.weight.device,
        dtype=model.token_embed.weight.dtype,
    )
    clean_gate = clean_cache["blocks"][layer_index]["mlp"]["gate"][0, position, :].detach().cpu()
    clean_up = clean_cache["blocks"][layer_index]["mlp"]["up"][0, position, :].detach().cpu()
    clean_activated = clean_cache["blocks"][layer_index]["mlp"]["activated"][0, position, :].detach().cpu()
    corrupt_gate = corrupt_cache["blocks"][layer_index]["mlp"]["gate"][0, position, :].detach().cpu()
    corrupt_up = corrupt_cache["blocks"][layer_index]["mlp"]["up"][0, position, :].detach().cpu()
    corrupt_activated = corrupt_cache["blocks"][layer_index]["mlp"]["activated"][0, position, :].detach().cpu()

    rows: list[dict[str, object]] = []
    for source_position, clean_source_token in enumerate(clean_tokens):
        corrupt_source_token = corrupt_tokens[source_position]
        clean_value_mix = clean_pattern[source_position] * clean_values[source_position]
        corrupt_value_mix = corrupt_pattern[source_position] * corrupt_values[source_position]
        clean_source_write, _ = head_value_mix_to_write(
            model=model,
            layer_index=layer_index,
            head_index=head_index,
            value_mix=clean_value_mix.detach().cpu(),
        )
        corrupt_source_write, _ = head_value_mix_to_write(
            model=model,
            layer_index=layer_index,
            head_index=head_index,
            value_mix=corrupt_value_mix.detach().cpu(),
        )
        modified_resid = corrupt_resid_after_attn.clone()
        modified_resid[0, position, :] = modified_resid[0, position, :] + (
            clean_source_write - corrupt_source_write
        ).to(device=modified_resid.device, dtype=modified_resid.dtype)
        modified_mlp = compute_mlp_from_resid_after_attn(
            model=model,
            resid_after_attn=modified_resid,
            layer_index=layer_index,
        )
        patched_gate = modified_mlp["gate"][0, position, :]
        patched_up = modified_mlp["up"][0, position, :]
        patched_activated = modified_mlp["activated"][0, position, :]

        for neuron_index in normalized_neuron_indices:
            clean_value = float(clean_activated[neuron_index].item())
            corrupt_value = float(corrupt_activated[neuron_index].item())
            patched_value = float(patched_activated[neuron_index].item())
            rows.append(
                {
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "source_position": source_position,
                    "clean_source_token": clean_source_token,
                    "corrupt_source_token": corrupt_source_token,
                    "clean_attention_weight": float(clean_pattern[source_position].item()),
                    "corrupt_attention_weight": float(corrupt_pattern[source_position].item()),
                    "clean_source_write_norm": float(clean_source_write.norm().item()),
                    "corrupt_source_write_norm": float(corrupt_source_write.norm().item()),
                    "neuron_index": neuron_index,
                    "clean_gate": float(clean_gate[neuron_index].item()),
                    "corrupt_gate": float(corrupt_gate[neuron_index].item()),
                    "patched_corrupt_gate": float(patched_gate[neuron_index].item()),
                    "gate_patch_delta": float(
                        (patched_gate[neuron_index] - corrupt_gate[neuron_index]).item()
                    ),
                    "clean_up": float(clean_up[neuron_index].item()),
                    "corrupt_up": float(corrupt_up[neuron_index].item()),
                    "patched_corrupt_up": float(patched_up[neuron_index].item()),
                    "up_patch_delta": float(
                        (patched_up[neuron_index] - corrupt_up[neuron_index]).item()
                    ),
                    "clean_activation": clean_value,
                    "corrupt_activation": corrupt_value,
                    "patched_corrupt_activation": patched_value,
                    "patch_delta": patched_value - corrupt_value,
                    "distance_to_clean_before": abs(corrupt_value - clean_value),
                    "distance_to_clean_after": abs(patched_value - clean_value),
                    "recovery_toward_clean": abs(corrupt_value - clean_value) - abs(patched_value - clean_value),
                }
            )

    return pd.DataFrame(rows)


def collect_mlp_neuron_activation_table(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    layer_index: int,
    device: torch.device,
    neuron_indices: list[int] | None = None,
    position: int = -1,
    limit: int | None = None,
) -> pd.DataFrame:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split for neuron activation table: {split}")
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for neuron activation table: {layer_index}")

    hidden_dim = model.blocks[layer_index].mlp.down_proj.weight.shape[1]
    if neuron_indices is None:
        normalized_neuron_indices = list(range(hidden_dim))
    else:
        normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
        invalid_neuron_indices = [
            neuron_index
            for neuron_index in normalized_neuron_indices
            if neuron_index < 0 or neuron_index >= hidden_dim
        ]
        if invalid_neuron_indices:
            raise ValueError(
                f"Invalid neuron indices for layer {layer_index + 1}: {invalid_neuron_indices}"
            )

    split_rows = bundle.raw_splits[split]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        split_rows = split_rows[:limit]

    rows: list[dict[str, object]] = []
    for row_index, raw_row in enumerate(split_rows):
        prompt = raw_row["prompt"]
        target_token = raw_row["target"]
        result, cache = run_prompt(
            model,
            bundle,
            prompt,
            device=device,
            expected_target=target_token,
            return_cache=True,
        )
        if cache is None:
            raise ValueError(
                f"Expected cache while collecting neuron activations for split={split} row={row_index}"
            )

        mlp_cache = cache["blocks"][layer_index]["mlp"]
        activated = mlp_cache["activated"][0, position, :].detach().cpu()
        prompt_tokens = prompt.split()
        correct_value_positions = [
            position_index
            for position_index, token in enumerate(prompt_tokens)
            if token == target_token
        ]
        if len(correct_value_positions) != 1:
            raise ValueError(
                f"Expected exactly one correct value position for prompt {prompt!r}, "
                f"found {correct_value_positions}"
            )

        query_context_positions = [
            position_index
            for position_index, token in enumerate(prompt_tokens[:-2])
            if token == raw_row["query_key"]
        ]
        if len(query_context_positions) != 1:
            raise ValueError(
                f"Expected exactly one context occurrence of query key {raw_row['query_key']!r}, "
                f"found {query_context_positions}"
            )

        query_pair_matches = [
            pair for pair in raw_row["context_pairs"] if pair["key"] == raw_row["query_key"]
        ]
        if len(query_pair_matches) != 1:
            raise ValueError(
                f"Expected exactly one query-pair match for {raw_row['query_key']!r}, "
                f"found {len(query_pair_matches)}"
            )

        _, corrupt_target = make_query_swap_prompt(raw_row)
        corrupt_value_positions = [
            position_index
            for position_index, token in enumerate(prompt_tokens)
            if token == corrupt_target
        ]
        if len(corrupt_value_positions) != 1:
            raise ValueError(
                f"Expected exactly one corrupt value position for token {corrupt_target!r}, "
                f"found {corrupt_value_positions}"
            )

        table_row: dict[str, object] = {
            "split": split,
            "index": row_index,
            "prompt": prompt,
            "target": target_token,
            "query_key": raw_row["query_key"],
            "predicted_token": result["predicted_token"],
            "num_pairs": int(raw_row["num_pairs"]),
            "prompt_length": len(prompt_tokens),
            "query_suffix_position": len(prompt_tokens) - 2,
            "final_position": len(prompt_tokens) - 1,
            "query_context_position": query_context_positions[0],
            "correct_value_position": correct_value_positions[0],
            "corrupt_target": corrupt_target,
            "corrupt_value_position": corrupt_value_positions[0],
            "query_pair_index": int(query_pair_matches[0]["pair_index"]),
        }
        for neuron_index in normalized_neuron_indices:
            table_row[_mlp_neuron_column_name(neuron_index)] = float(
                activated[neuron_index].item()
            )
        rows.append(table_row)

    return pd.DataFrame(rows)


def build_mlp_neuron_group_summary_table(
    neuron_activation_df: pd.DataFrame,
    neuron_indices: list[int],
    group_column: str,
) -> pd.DataFrame:
    if group_column not in neuron_activation_df.columns:
        raise ValueError(f"Unknown group column for neuron summary: {group_column}")
    if not neuron_indices:
        raise ValueError("Expected at least one neuron index for grouped summary")

    normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
    missing_columns = [
        _mlp_neuron_column_name(neuron_index)
        for neuron_index in normalized_neuron_indices
        if _mlp_neuron_column_name(neuron_index) not in neuron_activation_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Neuron activation table is missing columns for grouped summary: {missing_columns}"
        )

    grouped = neuron_activation_df.groupby(group_column, dropna=False)
    rows: list[dict[str, object]] = []
    for group_value, group_df in grouped:
        row: dict[str, object] = {
            group_column: group_value,
            "num_examples": int(len(group_df)),
        }
        for neuron_index in normalized_neuron_indices:
            column_name = _mlp_neuron_column_name(neuron_index)
            row[column_name] = float(group_df[column_name].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def build_top_mlp_neuron_examples(
    neuron_activation_df: pd.DataFrame,
    neuron_indices: list[int],
    top_k: int = 5,
) -> dict[int, list[dict[str, object]]]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not neuron_indices:
        raise ValueError("Expected at least one neuron index for top-example summary")

    normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
    results: dict[int, list[dict[str, object]]] = {}
    for neuron_index in normalized_neuron_indices:
        column_name = _mlp_neuron_column_name(neuron_index)
        if column_name not in neuron_activation_df.columns:
            raise ValueError(
                f"Neuron activation table is missing column for neuron {neuron_index}"
            )
        top_df = neuron_activation_df.sort_values(column_name, ascending=False).head(top_k)
        results[neuron_index] = [
            {
                "split": row["split"],
                "index": int(row["index"]),
                "prompt": row["prompt"],
                "target": row["target"],
                "query_key": row["query_key"],
                "activation": float(row[column_name]),
            }
            for _, row in top_df.iterrows()
        ]

    return results


def build_mlp_neuron_batch_ablation_table(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    layer_index: int,
    device: torch.device,
    neuron_indices: list[int],
    limit: int | None = None,
    position: int = -1,
) -> pd.DataFrame:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split for neuron ablation table: {split}")
    if layer_index < 0 or layer_index >= len(model.blocks):
        raise ValueError(f"Invalid layer index for neuron ablation table: {layer_index}")
    if not neuron_indices:
        raise ValueError("Expected at least one neuron index for neuron ablation table")

    split_rows = bundle.raw_splits[split]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        split_rows = split_rows[:limit]

    normalized_neuron_indices = [int(neuron_index) for neuron_index in neuron_indices]
    rows: list[dict[str, object]] = []
    for neuron_index in normalized_neuron_indices:
        clean_correct_count = 0
        ablate_correct_count = 0
        prediction_change_count = 0
        clean_margin_values: list[float] = []
        ablate_margin_values: list[float] = []

        for raw_row in split_rows:
            prompt = raw_row["prompt"]
            target_token = raw_row["target"]
            baseline_result, _ = run_prompt(
                model,
                bundle,
                prompt,
                device=device,
                expected_target=target_token,
                return_cache=False,
            )
            ablated_result = score_mlp_neuron_ablation_prompt(
                model=model,
                bundle=bundle,
                prompt=prompt,
                target_token=target_token,
                device=device,
                layer_index=layer_index,
                neuron_index=neuron_index,
                position_index=position,
            )
            clean_correct_count += int(baseline_result["predicted_token"] == target_token)
            ablate_correct_count += int(ablated_result["predicted_token"] == target_token)
            prediction_change_count += int(
                baseline_result["predicted_token"] != ablated_result["predicted_token"]
            )
            clean_margin_values.append(float(baseline_result["margin"]))
            ablate_margin_values.append(float(ablated_result["margin"]))

        num_examples = len(split_rows)
        rows.append(
            {
                "neuron_index": neuron_index,
                "num_examples": num_examples,
                "clean_accuracy": clean_correct_count / num_examples,
                "ablate_accuracy": ablate_correct_count / num_examples,
                "mean_margin": float(sum(clean_margin_values) / num_examples),
                "mean_ablate_margin": float(sum(ablate_margin_values) / num_examples),
                "mean_margin_drop": float(
                    (sum(clean_margin_values) - sum(ablate_margin_values)) / num_examples
                ),
                "prediction_change_rate": prediction_change_count / num_examples,
            }
        )

    return pd.DataFrame(rows)


def normalize_position_index(position: int, seq_len: int) -> int:
    normalized_position = int(position)
    if normalized_position < 0:
        normalized_position += seq_len
    if normalized_position < 0 or normalized_position >= seq_len:
        raise ValueError(f"Invalid position {position} for sequence length {seq_len}")
    return normalized_position


def derive_kv_prompt_structure(row: dict) -> dict[str, object]:
    required_keys = {"prompt", "query_key", "target", "context_pairs"}
    missing_keys = sorted(required_keys - set(row))
    if missing_keys:
        raise ValueError(f"Row is missing required KV fields: {missing_keys}")

    prompt_tokens = row["prompt"].split()
    context_pairs = row["context_pairs"]
    if not isinstance(context_pairs, list) or not context_pairs:
        raise ValueError("Expected row['context_pairs'] to be a non-empty list")

    sorted_pairs = sorted(context_pairs, key=lambda pair: int(pair["pair_index"]))
    expected_pair_indices = list(range(len(sorted_pairs)))
    actual_pair_indices = [int(pair["pair_index"]) for pair in sorted_pairs]
    if actual_pair_indices != expected_pair_indices:
        raise ValueError(
            "Expected contiguous pair_index values starting at 0, got "
            f"{actual_pair_indices}"
        )

    expected_length = 1 + (3 * len(sorted_pairs)) + 3
    if len(prompt_tokens) != expected_length:
        raise ValueError(
            "Prompt length does not match the expected KV-retrieval format: "
            f"expected {expected_length}, got {len(prompt_tokens)} for {row['prompt']!r}"
        )
    if prompt_tokens[0] != "<bos>":
        raise ValueError(f"Expected prompt to start with <bos>, got {prompt_tokens[0]!r}")

    key_positions: list[int] = []
    value_positions: list[int] = []
    separator_positions: list[int] = []
    role_by_position: dict[int, dict[str, object]] = {
        0: {
            "position": 0,
            "token": prompt_tokens[0],
            "structural_role": "bos",
            "algorithm_role": "bos",
            "pair_index": None,
            "matches_query_key": False,
            "is_selected_pair": False,
        }
    }

    for pair in sorted_pairs:
        pair_index = int(pair["pair_index"])
        key_position = 1 + (3 * pair_index)
        value_position = key_position + 1
        separator_position = key_position + 2

        if prompt_tokens[key_position] != pair["key"]:
            raise ValueError(
                f"Prompt key mismatch at pair {pair_index}: "
                f"expected {pair['key']!r}, got {prompt_tokens[key_position]!r}"
            )
        if prompt_tokens[value_position] != pair["value"]:
            raise ValueError(
                f"Prompt value mismatch at pair {pair_index}: "
                f"expected {pair['value']!r}, got {prompt_tokens[value_position]!r}"
            )
        if prompt_tokens[separator_position] != ";":
            raise ValueError(
                f"Expected ';' after pair {pair_index}, got {prompt_tokens[separator_position]!r}"
            )

        is_selected_pair = pair["key"] == row["query_key"]
        key_positions.append(key_position)
        value_positions.append(value_position)
        separator_positions.append(separator_position)

        role_by_position[key_position] = {
            "position": key_position,
            "token": prompt_tokens[key_position],
            "structural_role": "context_key",
            "algorithm_role": "selected_key" if is_selected_pair else "distractor_key",
            "pair_index": pair_index,
            "matches_query_key": is_selected_pair,
            "is_selected_pair": is_selected_pair,
        }
        role_by_position[value_position] = {
            "position": value_position,
            "token": prompt_tokens[value_position],
            "structural_role": "context_value",
            "algorithm_role": "selected_value" if is_selected_pair else "distractor_value",
            "pair_index": pair_index,
            "matches_query_key": is_selected_pair,
            "is_selected_pair": is_selected_pair,
        }
        role_by_position[separator_position] = {
            "position": separator_position,
            "token": prompt_tokens[separator_position],
            "structural_role": "separator",
            "algorithm_role": "separator",
            "pair_index": pair_index,
            "matches_query_key": False,
            "is_selected_pair": is_selected_pair,
        }

    matching_pairs = [pair for pair in sorted_pairs if pair["key"] == row["query_key"]]
    if len(matching_pairs) != 1:
        raise ValueError(
            f"Expected exactly one selected pair for query key {row['query_key']!r}, found {len(matching_pairs)}"
        )
    selected_pair = matching_pairs[0]
    selected_pair_index = int(selected_pair["pair_index"])
    selected_key_position = 1 + (3 * selected_pair_index)
    selected_value_position = selected_key_position + 1

    query_marker_position = 1 + (3 * len(sorted_pairs))
    query_key_position = query_marker_position + 1
    arrow_position = query_marker_position + 2
    if prompt_tokens[query_marker_position] != "Q":
        raise ValueError(
            f"Expected query marker 'Q' at position {query_marker_position}, "
            f"got {prompt_tokens[query_marker_position]!r}"
        )
    if prompt_tokens[query_key_position] != row["query_key"]:
        raise ValueError(
            f"Prompt query-key mismatch: expected {row['query_key']!r}, "
            f"got {prompt_tokens[query_key_position]!r}"
        )
    if prompt_tokens[arrow_position] != "->":
        raise ValueError(
            f"Expected answer marker '->' at position {arrow_position}, got {prompt_tokens[arrow_position]!r}"
        )

    role_by_position[query_marker_position] = {
        "position": query_marker_position,
        "token": prompt_tokens[query_marker_position],
        "structural_role": "query_marker",
        "algorithm_role": "query_marker",
        "pair_index": None,
        "matches_query_key": False,
        "is_selected_pair": False,
    }
    role_by_position[query_key_position] = {
        "position": query_key_position,
        "token": prompt_tokens[query_key_position],
        "structural_role": "query_key",
        "algorithm_role": "query_key_token",
        "pair_index": None,
        "matches_query_key": True,
        "is_selected_pair": False,
    }
    role_by_position[arrow_position] = {
        "position": arrow_position,
        "token": prompt_tokens[arrow_position],
        "structural_role": "answer_marker",
        "algorithm_role": "answer_marker",
        "pair_index": None,
        "matches_query_key": False,
        "is_selected_pair": False,
    }

    distractor_pairs = [pair for pair in sorted_pairs if pair["key"] != row["query_key"]]
    return {
        "prompt_tokens": prompt_tokens,
        "num_pairs": len(sorted_pairs),
        "role_by_position": role_by_position,
        "context_pairs": sorted_pairs,
        "key_positions": key_positions,
        "value_positions": value_positions,
        "separator_positions": separator_positions,
        "query_marker_position": query_marker_position,
        "query_key_position": query_key_position,
        "arrow_position": arrow_position,
        "selected_pair": selected_pair,
        "selected_pair_index": selected_pair_index,
        "selected_key_position": selected_key_position,
        "selected_value_position": selected_value_position,
        "selected_key_token": selected_pair["key"],
        "selected_value_token": selected_pair["value"],
        "distractor_pairs": distractor_pairs,
        "distractor_key_positions": [1 + (3 * int(pair["pair_index"])) for pair in distractor_pairs],
        "distractor_value_positions": [2 + (3 * int(pair["pair_index"])) for pair in distractor_pairs],
        "distractor_key_tokens": [pair["key"] for pair in distractor_pairs],
        "distractor_value_tokens": [pair["value"] for pair in distractor_pairs],
    }


def build_kv_prompt_layout_table(row: dict) -> pd.DataFrame:
    structure = derive_kv_prompt_structure(row)
    layout_rows = [
        structure["role_by_position"][position]
        for position in range(len(structure["prompt_tokens"]))
    ]
    return pd.DataFrame(layout_rows)


def build_kv_algorithm_variable_table(row: dict) -> pd.DataFrame:
    structure = derive_kv_prompt_structure(row)
    return pd.DataFrame(
        [
            {
                "prompt": row["prompt"],
                "query_key": row["query_key"],
                "query_key_position": structure["query_key_position"],
                "selected_pair_index": structure["selected_pair_index"],
                "selected_key_position": structure["selected_key_position"],
                "selected_value_position": structure["selected_value_position"],
                "target_value": row["target"],
                "distractor_key_positions": structure["distractor_key_positions"],
                "distractor_value_positions": structure["distractor_value_positions"],
                "distractor_keys": structure["distractor_key_tokens"],
                "distractor_values": structure["distractor_value_tokens"],
            }
        ]
    )


def make_query_swap_row(row: dict) -> dict[str, object]:
    corrupt_prompt, corrupt_target = make_query_swap_prompt(row)
    corrupt_query_key = corrupt_prompt.split()[-2]
    corrupt_row = dict(row)
    corrupt_row["prompt"] = corrupt_prompt
    corrupt_row["target"] = corrupt_target
    corrupt_row["query_key"] = corrupt_query_key
    return corrupt_row


def build_stage_variable_readout_table(
    model: nn.Module,
    bundle: DatasetBundle,
    cache: dict,
    row: dict,
    position: int = -1,
) -> pd.DataFrame:
    structure = derive_kv_prompt_structure(row)
    stage_specs: list[tuple[str, torch.Tensor]] = []
    for layer_index, block_cache in enumerate(cache["blocks"], start=1):
        if layer_index == 1:
            stage_specs.append((f"L{layer_index} resid_in", block_cache["resid_in"]))
        else:
            stage_specs.append((f"L{layer_index} resid_in", block_cache["resid_in"]))
        stage_specs.append((f"L{layer_index} after attn", block_cache["resid_after_attn"]))
        stage_specs.append((f"L{layer_index} after mlp", block_cache["resid_after_mlp"]))

    key_tokens = bundle.metadata["vocabulary"]["keys"]
    key_token_ids = [bundle.token_to_id[token] for token in key_tokens]
    value_token_ids = [bundle.token_to_id[token] for token in bundle.value_tokens]

    query_key_id = bundle.token_to_id[row["query_key"]]
    target_value_id = bundle.token_to_id[row["target"]]
    distractor_key_ids = [bundle.token_to_id[token] for token in structure["distractor_key_tokens"]]
    distractor_value_ids = [bundle.token_to_id[token] for token in structure["distractor_value_tokens"]]

    rows: list[dict[str, object]] = []
    for stage_name, stage_tensor in stage_specs:
        normalized_position = normalize_position_index(position, stage_tensor.shape[1])
        residual_vector = stage_tensor[0, normalized_position].detach().cpu()
        logits = residual_vector_to_logits(model, residual_vector)

        top_token_id = int(logits.argmax().item())
        key_logits = logits[key_token_ids]
        key_top_index = int(key_logits.argmax().item())
        top_key_token = key_tokens[key_top_index]
        top_key_logit = float(key_logits[key_top_index].item())

        top_value_relative_index = int(logits[value_token_ids].argmax().item())
        top_value_token = bundle.value_tokens[top_value_relative_index]
        top_value_logit = float(logits[value_token_ids[top_value_relative_index]].item())

        best_distractor_key_logit = max(float(logits[token_id].item()) for token_id in distractor_key_ids)
        best_distractor_value_logit = max(float(logits[token_id].item()) for token_id in distractor_value_ids)

        query_key_logit = float(logits[query_key_id].item())
        target_value_logit = float(logits[target_value_id].item())
        rows.append(
            {
                "stage": stage_name,
                "top_token": bundle.id_to_token[top_token_id],
                "top_logit": float(logits[top_token_id].item()),
                "top_key_token": top_key_token,
                "top_key_logit": top_key_logit,
                "query_key_logit": query_key_logit,
                "query_is_top_key": top_key_token == row["query_key"],
                "query_minus_best_distractor_key": query_key_logit - best_distractor_key_logit,
                "top_value_token": top_value_token,
                "top_value_logit": top_value_logit,
                "target_value_logit": target_value_logit,
                "target_is_top_value": top_value_token == row["target"],
                "target_minus_best_distractor_value": target_value_logit - best_distractor_value_logit,
            }
        )

    return pd.DataFrame(rows)


def collect_stage_variable_readout_table(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    device: torch.device,
    limit: int | None = None,
    position: int = -1,
) -> pd.DataFrame:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split for stage variable readout table: {split}")

    split_rows = bundle.raw_splits[split]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        split_rows = split_rows[:limit]

    rows: list[dict[str, object]] = []
    for example_index, row in enumerate(split_rows):
        _, cache = run_prompt(
            model,
            bundle,
            row["prompt"],
            device=device,
            expected_target=row["target"],
            return_cache=True,
        )
        if cache is None:
            raise ValueError("Expected cache while collecting stage variable readouts")

        stage_df = build_stage_variable_readout_table(
            model=model,
            bundle=bundle,
            cache=cache,
            row=row,
            position=position,
        )
        for record in stage_df.to_dict(orient="records"):
            record.update(
                {
                    "split": split,
                    "index": example_index,
                    "query_key": row["query_key"],
                    "target": row["target"],
                }
            )
            rows.append(record)

    return pd.DataFrame(rows)


def build_stage_variable_summary_table(stage_variable_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "stage",
        "query_is_top_key",
        "target_is_top_value",
        "query_minus_best_distractor_key",
        "target_minus_best_distractor_value",
    }
    missing_columns = sorted(required_columns - set(stage_variable_df.columns))
    if missing_columns:
        raise ValueError(
            f"stage_variable_df is missing required columns for stage summary: {missing_columns}"
        )

    return (
        stage_variable_df
        .groupby("stage", as_index=False)
        .agg(
            num_examples=("stage", "size"),
            query_top_key_rate=("query_is_top_key", "mean"),
            target_top_value_rate=("target_is_top_value", "mean"),
            mean_query_minus_best_distractor_key=("query_minus_best_distractor_key", "mean"),
            mean_target_minus_best_distractor_value=("target_minus_best_distractor_value", "mean"),
        )
    )


def build_single_prompt_head_role_table(
    model: nn.Module,
    bundle: DatasetBundle,
    row: dict,
    cache: dict,
    target_token: str,
    foil_token: str,
    query_position: int = -1,
) -> pd.DataFrame:
    structure = derive_kv_prompt_structure(row)
    prompt_tokens = structure["prompt_tokens"]
    normalized_query_position = normalize_position_index(query_position, len(prompt_tokens))

    rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        attention = block.attn
        for head_index in range(attention.n_heads):
            qk_df = build_qk_table(
                prompt=row["prompt"],
                cache=cache,
                layer_index=layer_index,
                head_index=head_index,
                query_position=normalized_query_position,
            )
            source_df = build_head_source_contribution_table(
                model=model,
                bundle=bundle,
                prompt=row["prompt"],
                cache=cache,
                layer_index=layer_index,
                head_index=head_index,
                target_token=target_token,
                foil_token=foil_token,
                query_position=normalized_query_position,
            )

            top_attention_row = qk_df.sort_values("attention_weight", ascending=False).iloc[0]
            top_weighted_row = source_df.sort_values("weighted_target_minus_foil", ascending=False).iloc[0]
            selected_key_row = qk_df[qk_df["position"] == structure["selected_key_position"]].iloc[0]
            selected_value_row = qk_df[qk_df["position"] == structure["selected_value_position"]].iloc[0]
            selected_value_source_row = source_df[
                source_df["source_position"] == structure["selected_value_position"]
            ].iloc[0]

            distractor_key_attention = float(
                qk_df[qk_df["position"].isin(structure["distractor_key_positions"])]["attention_weight"].sum()
            )
            distractor_value_attention = float(
                qk_df[qk_df["position"].isin(structure["distractor_value_positions"])]["attention_weight"].sum()
            )
            separator_attention = float(
                qk_df[qk_df["position"].isin(structure["separator_positions"])]["attention_weight"].sum()
            )
            query_key_row = qk_df[qk_df["position"] == structure["query_key_position"]].iloc[0]
            query_marker_row = qk_df[qk_df["position"] == structure["query_marker_position"]].iloc[0]
            arrow_row = qk_df[qk_df["position"] == structure["arrow_position"]].iloc[0]

            rows.append(
                {
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "top_attention_position": int(top_attention_row["position"]),
                    "top_attention_token": top_attention_row["token"],
                    "top_attention_role": structure["role_by_position"][int(top_attention_row["position"])]["algorithm_role"],
                    "top_attention_weight": float(top_attention_row["attention_weight"]),
                    "query_key_attention": float(query_key_row["attention_weight"]),
                    "selected_key_attention": float(selected_key_row["attention_weight"]),
                    "selected_value_attention": float(selected_value_row["attention_weight"]),
                    "distractor_key_attention": distractor_key_attention,
                    "distractor_value_attention": distractor_value_attention,
                    "separator_attention": separator_attention,
                    "query_marker_attention": float(query_marker_row["attention_weight"]),
                    "answer_marker_attention": float(arrow_row["attention_weight"]),
                    "selected_value_ov_target_minus_foil": float(
                        selected_value_source_row["ov_target_minus_foil"]
                    ),
                    "selected_value_weighted_target_minus_foil": float(
                        selected_value_source_row["weighted_target_minus_foil"]
                    ),
                    "top_weighted_source_position": int(top_weighted_row["source_position"]),
                    "top_weighted_source_token": top_weighted_row["source_token"],
                    "top_weighted_source_role": structure["role_by_position"][int(top_weighted_row["source_position"])]["algorithm_role"],
                    "top_weighted_target_minus_foil": float(top_weighted_row["weighted_target_minus_foil"]),
                }
            )

    return pd.DataFrame(rows)


def build_query_swap_head_role_comparison_table(
    model: nn.Module,
    clean_row: dict,
    clean_cache: dict,
    corrupt_row: dict,
    corrupt_cache: dict,
    query_position: int = -1,
) -> pd.DataFrame:
    clean_structure = derive_kv_prompt_structure(clean_row)
    corrupt_structure = derive_kv_prompt_structure(corrupt_row)
    clean_query_position = normalize_position_index(query_position, len(clean_structure["prompt_tokens"]))
    corrupt_query_position = normalize_position_index(query_position, len(corrupt_structure["prompt_tokens"]))

    rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        attention = block.attn
        for head_index in range(attention.n_heads):
            clean_attention = clean_cache["blocks"][layer_index]["attention"]["pattern"][
                0, head_index, clean_query_position, :
            ].detach().cpu()
            corrupt_attention = corrupt_cache["blocks"][layer_index]["attention"]["pattern"][
                0, head_index, corrupt_query_position, :
            ].detach().cpu()

            clean_top_position = int(clean_attention.argmax().item())
            corrupt_top_position = int(corrupt_attention.argmax().item())

            rows.append(
                {
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "clean_query_key_attention": float(clean_attention[clean_structure["query_key_position"]].item()),
                    "corrupt_query_key_attention": float(corrupt_attention[corrupt_structure["query_key_position"]].item()),
                    "clean_selected_key_attention": float(clean_attention[clean_structure["selected_key_position"]].item()),
                    "corrupt_selected_key_attention": float(corrupt_attention[corrupt_structure["selected_key_position"]].item()),
                    "clean_selected_value_attention": float(clean_attention[clean_structure["selected_value_position"]].item()),
                    "corrupt_selected_value_attention": float(corrupt_attention[corrupt_structure["selected_value_position"]].item()),
                    "clean_distractor_value_attention": float(
                        clean_attention[clean_structure["distractor_value_positions"]].sum().item()
                    ),
                    "corrupt_distractor_value_attention": float(
                        corrupt_attention[corrupt_structure["distractor_value_positions"]].sum().item()
                    ),
                    "clean_top_attention_position": clean_top_position,
                    "clean_top_attention_token": clean_structure["prompt_tokens"][clean_top_position],
                    "clean_top_attention_role": clean_structure["role_by_position"][clean_top_position]["algorithm_role"],
                    "clean_top_attention_weight": float(clean_attention[clean_top_position].item()),
                    "corrupt_top_attention_position": corrupt_top_position,
                    "corrupt_top_attention_token": corrupt_structure["prompt_tokens"][corrupt_top_position],
                    "corrupt_top_attention_role": corrupt_structure["role_by_position"][corrupt_top_position]["algorithm_role"],
                    "corrupt_top_attention_weight": float(corrupt_attention[corrupt_top_position].item()),
                }
            )

    return pd.DataFrame(rows)


def collect_head_role_attention_table(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    device: torch.device,
    limit: int | None = None,
    query_position: int = -1,
) -> pd.DataFrame:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split for head-role collection: {split}")

    split_rows = bundle.raw_splits[split]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        split_rows = split_rows[:limit]

    rows: list[dict[str, object]] = []
    for example_index, row in enumerate(split_rows):
        _, cache = run_prompt(
            model,
            bundle,
            row["prompt"],
            device=device,
            expected_target=row["target"],
            return_cache=True,
        )
        if cache is None:
            raise ValueError("Expected cache while collecting head-role attention table")

        structure = derive_kv_prompt_structure(row)
        normalized_query_position = normalize_position_index(query_position, len(structure["prompt_tokens"]))
        for layer_index, block in enumerate(model.blocks):
            attention = block.attn
            for head_index in range(attention.n_heads):
                head_attention = cache["blocks"][layer_index]["attention"]["pattern"][
                    0, head_index, normalized_query_position, :
                ].detach().cpu()
                top_attention_position = int(head_attention.argmax().item())
                rows.append(
                    {
                        "split": split,
                        "index": example_index,
                        "query_key": row["query_key"],
                        "target": row["target"],
                        "selected_pair_index": structure["selected_pair_index"],
                        "layer_index": layer_index,
                        "head_index": head_index,
                        "component": f"L{layer_index + 1}H{head_index}",
                        "query_key_attention": float(head_attention[structure["query_key_position"]].item()),
                        "selected_key_attention": float(head_attention[structure["selected_key_position"]].item()),
                        "selected_value_attention": float(head_attention[structure["selected_value_position"]].item()),
                        "distractor_key_attention": float(
                            head_attention[structure["distractor_key_positions"]].sum().item()
                        ),
                        "distractor_value_attention": float(
                            head_attention[structure["distractor_value_positions"]].sum().item()
                        ),
                        "separator_attention": float(
                            head_attention[structure["separator_positions"]].sum().item()
                        ),
                        "query_marker_attention": float(head_attention[structure["query_marker_position"]].item()),
                        "answer_marker_attention": float(head_attention[structure["arrow_position"]].item()),
                        "top_attention_position": top_attention_position,
                        "top_attention_role": structure["role_by_position"][top_attention_position]["algorithm_role"],
                        "top_attention_is_query_key": top_attention_position == structure["query_key_position"],
                        "top_attention_is_selected_key": top_attention_position == structure["selected_key_position"],
                        "top_attention_is_selected_value": top_attention_position == structure["selected_value_position"],
                        "top_attention_is_distractor_key": top_attention_position in structure["distractor_key_positions"],
                        "top_attention_is_distractor_value": top_attention_position in structure["distractor_value_positions"],
                    }
                )

    return pd.DataFrame(rows)


def build_head_role_summary_table(head_role_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "component",
        "query_key_attention",
        "selected_key_attention",
        "selected_value_attention",
        "distractor_key_attention",
        "distractor_value_attention",
        "separator_attention",
        "query_marker_attention",
        "answer_marker_attention",
        "top_attention_is_query_key",
        "top_attention_is_selected_key",
        "top_attention_is_selected_value",
        "top_attention_is_distractor_key",
        "top_attention_is_distractor_value",
    }
    missing_columns = sorted(required_columns - set(head_role_df.columns))
    if missing_columns:
        raise ValueError(
            f"head_role_df is missing required columns for head-role summary: {missing_columns}"
        )

    return (
        head_role_df
        .groupby(["layer_index", "head_index", "component"], as_index=False)
        .agg(
            num_examples=("component", "size"),
            mean_query_key_attention=("query_key_attention", "mean"),
            mean_selected_key_attention=("selected_key_attention", "mean"),
            mean_selected_value_attention=("selected_value_attention", "mean"),
            mean_distractor_key_attention=("distractor_key_attention", "mean"),
            mean_distractor_value_attention=("distractor_value_attention", "mean"),
            mean_separator_attention=("separator_attention", "mean"),
            mean_query_marker_attention=("query_marker_attention", "mean"),
            mean_answer_marker_attention=("answer_marker_attention", "mean"),
            top_query_key_rate=("top_attention_is_query_key", "mean"),
            top_selected_key_rate=("top_attention_is_selected_key", "mean"),
            top_selected_value_rate=("top_attention_is_selected_value", "mean"),
            top_distractor_key_rate=("top_attention_is_distractor_key", "mean"),
            top_distractor_value_rate=("top_attention_is_distractor_value", "mean"),
        )
    )


def build_slot_attention_breakdown(
    row: dict,
    attention_vector: torch.Tensor,
) -> pd.DataFrame:
    structure = derive_kv_prompt_structure(row)
    if attention_vector.ndim != 1:
        raise ValueError(
            f"Expected a rank-1 attention vector for slot attention breakdown, got {tuple(attention_vector.shape)}"
        )
    if attention_vector.shape[0] != len(structure["prompt_tokens"]):
        raise ValueError(
            "Attention length does not match prompt length for slot attention breakdown: "
            f"{attention_vector.shape[0]} vs {len(structure['prompt_tokens'])}"
        )

    rows: list[dict[str, object]] = []
    for pair in structure["context_pairs"]:
        pair_index = int(pair["pair_index"])
        key_position = 1 + (3 * pair_index)
        value_position = key_position + 1
        separator_position = key_position + 2
        key_attention = float(attention_vector[key_position].item())
        value_attention = float(attention_vector[value_position].item())
        separator_attention = float(attention_vector[separator_position].item())
        total_attention = key_attention + value_attention + separator_attention
        is_selected_slot = pair_index == structure["selected_pair_index"]
        rows.append(
            {
                "pair_index": pair_index,
                "key_token": pair["key"],
                "value_token": pair["value"],
                "key_position": key_position,
                "value_position": value_position,
                "separator_position": separator_position,
                "key_attention": key_attention,
                "value_attention": value_attention,
                "separator_attention": separator_attention,
                "slot_attention": total_attention,
                "is_selected_slot": is_selected_slot,
            }
        )

    return pd.DataFrame(rows)


def build_single_prompt_slot_routing_table(
    model: nn.Module,
    row: dict,
    cache: dict,
    query_position: int = -1,
) -> pd.DataFrame:
    structure = derive_kv_prompt_structure(row)
    normalized_query_position = normalize_position_index(query_position, len(structure["prompt_tokens"]))

    rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        attention = block.attn
        for head_index in range(attention.n_heads):
            head_attention = cache["blocks"][layer_index]["attention"]["pattern"][
                0, head_index, normalized_query_position, :
            ].detach().cpu()
            slot_df = build_slot_attention_breakdown(row, head_attention)
            top_slot_row = slot_df.sort_values(
                ["slot_attention", "value_attention", "key_attention"],
                ascending=[False, False, False],
            ).iloc[0]
            selected_slot_row = slot_df[slot_df["is_selected_slot"]].iloc[0]

            rows.append(
                {
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "top_slot_pair_index": int(top_slot_row["pair_index"]),
                    "top_slot_key_token": top_slot_row["key_token"],
                    "top_slot_value_token": top_slot_row["value_token"],
                    "top_slot_attention": float(top_slot_row["slot_attention"]),
                    "top_slot_key_attention": float(top_slot_row["key_attention"]),
                    "top_slot_value_attention": float(top_slot_row["value_attention"]),
                    "top_slot_matches_selected": bool(top_slot_row["is_selected_slot"]),
                    "selected_pair_index": int(selected_slot_row["pair_index"]),
                    "selected_slot_attention": float(selected_slot_row["slot_attention"]),
                    "selected_slot_key_attention": float(selected_slot_row["key_attention"]),
                    "selected_slot_value_attention": float(selected_slot_row["value_attention"]),
                    "slot_breakdown": slot_df.to_dict(orient="records"),
                }
            )

    return pd.DataFrame(rows)


def build_query_swap_slot_routing_comparison_table(
    model: nn.Module,
    clean_row: dict,
    clean_cache: dict,
    corrupt_row: dict,
    corrupt_cache: dict,
    query_position: int = -1,
) -> pd.DataFrame:
    clean_structure = derive_kv_prompt_structure(clean_row)
    corrupt_structure = derive_kv_prompt_structure(corrupt_row)
    clean_query_position = normalize_position_index(query_position, len(clean_structure["prompt_tokens"]))
    corrupt_query_position = normalize_position_index(query_position, len(corrupt_structure["prompt_tokens"]))

    rows: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.blocks):
        attention = block.attn
        for head_index in range(attention.n_heads):
            clean_attention = clean_cache["blocks"][layer_index]["attention"]["pattern"][
                0, head_index, clean_query_position, :
            ].detach().cpu()
            corrupt_attention = corrupt_cache["blocks"][layer_index]["attention"]["pattern"][
                0, head_index, corrupt_query_position, :
            ].detach().cpu()

            clean_slot_df = build_slot_attention_breakdown(clean_row, clean_attention)
            corrupt_slot_df = build_slot_attention_breakdown(corrupt_row, corrupt_attention)
            clean_top_slot = clean_slot_df.sort_values(
                ["slot_attention", "value_attention", "key_attention"],
                ascending=[False, False, False],
            ).iloc[0]
            corrupt_top_slot = corrupt_slot_df.sort_values(
                ["slot_attention", "value_attention", "key_attention"],
                ascending=[False, False, False],
            ).iloc[0]
            clean_selected_slot = clean_slot_df[clean_slot_df["is_selected_slot"]].iloc[0]
            corrupt_selected_slot = corrupt_slot_df[corrupt_slot_df["is_selected_slot"]].iloc[0]

            rows.append(
                {
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "component": f"L{layer_index + 1}H{head_index}",
                    "clean_selected_pair_index": int(clean_selected_slot["pair_index"]),
                    "corrupt_selected_pair_index": int(corrupt_selected_slot["pair_index"]),
                    "clean_top_slot_pair_index": int(clean_top_slot["pair_index"]),
                    "corrupt_top_slot_pair_index": int(corrupt_top_slot["pair_index"]),
                    "clean_top_slot_matches_selected": bool(clean_top_slot["is_selected_slot"]),
                    "corrupt_top_slot_matches_selected": bool(corrupt_top_slot["is_selected_slot"]),
                    "clean_selected_slot_attention": float(clean_selected_slot["slot_attention"]),
                    "corrupt_selected_slot_attention": float(corrupt_selected_slot["slot_attention"]),
                    "selected_slot_attention_delta": float(
                        clean_selected_slot["slot_attention"] - corrupt_selected_slot["slot_attention"]
                    ),
                    "clean_selected_slot_value_attention": float(clean_selected_slot["value_attention"]),
                    "corrupt_selected_slot_value_attention": float(corrupt_selected_slot["value_attention"]),
                    "clean_slot_breakdown": clean_slot_df.to_dict(orient="records"),
                    "corrupt_slot_breakdown": corrupt_slot_df.to_dict(orient="records"),
                }
            )

    return pd.DataFrame(rows)


def collect_slot_routing_table(
    model: nn.Module,
    bundle: DatasetBundle,
    split: str,
    device: torch.device,
    limit: int | None = None,
    query_position: int = -1,
) -> pd.DataFrame:
    if split not in bundle.raw_splits:
        raise ValueError(f"Unknown split for slot-routing collection: {split}")

    split_rows = bundle.raw_splits[split]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        split_rows = split_rows[:limit]

    rows: list[dict[str, object]] = []
    for example_index, row in enumerate(split_rows):
        _, cache = run_prompt(
            model,
            bundle,
            row["prompt"],
            device=device,
            expected_target=row["target"],
            return_cache=True,
        )
        if cache is None:
            raise ValueError("Expected cache while collecting slot-routing table")

        slot_df = build_single_prompt_slot_routing_table(
            model=model,
            row=row,
            cache=cache,
            query_position=query_position,
        )
        for record in slot_df.to_dict(orient="records"):
            record.update(
                {
                    "split": split,
                    "index": example_index,
                    "query_key": row["query_key"],
                    "target": row["target"],
                    "selected_pair_index": derive_kv_prompt_structure(row)["selected_pair_index"],
                }
            )
            rows.append(record)

    return pd.DataFrame(rows)


def build_slot_routing_summary_table(slot_routing_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "layer_index",
        "head_index",
        "component",
        "top_slot_matches_selected",
        "selected_slot_attention",
        "selected_slot_key_attention",
        "selected_slot_value_attention",
        "top_slot_attention",
    }
    missing_columns = sorted(required_columns - set(slot_routing_df.columns))
    if missing_columns:
        raise ValueError(
            f"slot_routing_df is missing required columns for slot-routing summary: {missing_columns}"
        )

    return (
        slot_routing_df
        .groupby(["layer_index", "head_index", "component"], as_index=False)
        .agg(
            num_examples=("component", "size"),
            top_selected_slot_rate=("top_slot_matches_selected", "mean"),
            mean_selected_slot_attention=("selected_slot_attention", "mean"),
            mean_selected_slot_key_attention=("selected_slot_key_attention", "mean"),
            mean_selected_slot_value_attention=("selected_slot_value_attention", "mean"),
            mean_top_slot_attention=("top_slot_attention", "mean"),
        )
    )
