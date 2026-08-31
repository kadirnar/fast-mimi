"""fast-mimi **fp32-exact**: the fast-kernel campaign's kernels applied in place to `transformers.MimiModel`.

Same public API as the Hugging Face model (`encode(audio, padding_mask)` / `decode(codes, padding_mask)`), same
fp32 weights, bit-identical discrete codes and a decoded waveform within rtol 2e-4 / atol 2e-5 of the reference
on every gated input (see the benchmarks).  No tensor cores, no reduced precision: exact fp32 FMA
everywhere, deterministic.

    from fast_mimi.fp32 import build
    model = build()                                   # transformers MimiModel (kyutai/mimi, fp32, CUDA) + kernels
    codes = model.encode(audio, mask).audio_codes     # identical to the stock model
    wave = model.decode(codes, mask).audio_values

Requirements: torch >= 2.13 (CUDA 13), triton >= 3.7, transformers >= 5.14, and an nvcc for the CUDA C++
kernels (`pip install nvidia-cuda-nvcc nvidia-cuda-cccl nvidia-cuda-crt nvidia-nvvm nvidia-cuda-runtime`
provides one; `ensure_cuda_home()` discovers it).  The first call for a new input length captures a CUDA graph
(the kernels compile once, cached under ~/.cache/fast-mimi/fp32).
"""
from __future__ import annotations

from typing import Any

from ._compat import Graphed, eager_mode, ensure_cuda_home  # noqa: F401
from .runtime import apply, report  # noqa: F401

MODEL_ID = "kyutai/mimi"


class _Ctx:
    """The minimal context `apply(model, ctx)` expects."""

    def __init__(self, device, log=None):
        import torch
        self.device = torch.device(device)
        self.logs: list[str] = []
        self._log = log

    def log(self, message: str) -> None:
        self.logs.append(str(message))
        if self._log:
            self._log(f"[v4] {message}")


def load_reference(model_id: str = MODEL_ID, device: str = "cuda"):
    """The stock fp32 `transformers.MimiModel` on `device` (TF32 off is the caller's responsibility)."""
    import torch
    from transformers import MimiModel
    try:
        model = MimiModel.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:
        model = MimiModel.from_pretrained(model_id, torch_dtype=torch.float32)
    return model.to(device).eval()


def build(model: Any | None = None, model_id: str = MODEL_ID, device: str = "cuda", log=None):
    """Patch a stock fp32 `MimiModel` (loaded from `model_id` when not given) with the exact kernels + CUDA graphs."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ensure_cuda_home()
    if model is None:
        model = load_reference(model_id, device)
    return apply(model, _Ctx(device, log))
