"""Fast inference for Kyutai's Mimi neural audio codec.

Two backends, both batch 1, both on NVIDIA GPUs:

    from fast_mimi.v4 import build            # exact fp32, codes identical to transformers
    model = build()
    codes = model.encode(audio).audio_codes
    wave = model.decode(codes).audio_values

    from fast_mimi.v3 import load_mimi_state, build   # bf16 tensor cores, fastest
    codec = build("graph+triton", load_mimi_state("kyutai/mimi"))
    codes = codec.encode(audio)
    wave = codec.decode(codes, length=audio.shape[-1])

The submodules are imported lazily: `fast_mimi.v4` needs `transformers` and an nvcc, `fast_mimi.v3` needs only
Triton, and neither is loaded until you ask for it.
"""

from __future__ import annotations

__all__ = ["v3", "v4"]


def __getattr__(name: str):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
