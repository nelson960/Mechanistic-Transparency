from __future__ import annotations

import torch


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _cuda_is_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_training_device(device_arg: str) -> str:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        if _mps_is_available():
            return "mps"
        if _cuda_is_available():
            return "cuda"
        return "cpu"
    if normalized == "mps":
        if not _mps_is_available():
            raise ValueError("Requested device 'mps' is not available on this machine")
        return "mps"
    if normalized.startswith("cuda"):
        if not _cuda_is_available():
            raise ValueError(f"Requested device {device_arg!r} is not available on this machine")
        return device_arg
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported training device {device_arg!r}; expected one of auto, cpu, mps, cuda")
