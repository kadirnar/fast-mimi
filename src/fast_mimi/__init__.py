"""Fast inference for Kyutai's Mimi neural audio codec, on top of `transformers.MimiModel`.

    from transformers import MimiModel
    import fast_mimi

    model = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()

    fast_mimi.optimize(model)                    # FP32 - codes bit-identical to transformers
    fast_mimi.optimize(model, dtype="fp16")      # FP16 - tensor cores, fastest

`optimize` patches the model in place and returns it.  `encode` and `decode` keep the transformers signatures and
return `MimiEncoderOutput` / `MimiDecoderOutput`, so anything written against `MimiModel` keeps working; anything
the fast path cannot serve (a batch it was not built for, a CPU tensor, an unusual keyword) falls back to the
stock implementation rather than failing.

The two precisions are the only options.  FP32 replaces the model's own kernels with exact fp32 ones: same FLOPs,
same results, no tensor cores.  FP16 runs the convolutions and transformers on tensor cores, which is faster and
gives the same reconstruction quality, but the discrete codes are no longer bit-identical (~80 % match).
"""

from __future__ import annotations

FP32 = "fp32"
FP16 = "fp16"

__all__ = ["FP16", "FP32", "optimize"]


def optimize(model, dtype: str = FP32, **kwargs):
    """Patch `model` (a `transformers.MimiModel` on CUDA) with the fast kernels and return it.

    Args:
        model: the model to patch, in place.
        dtype: `"fp32"` for the exact kernels, `"fp16"` for the tensor-core ones.
        **kwargs: forwarded to the backend.

    Returns:
        The same model object.
    """
    name = str(dtype).lower().replace("torch.", "")
    if name in ("fp32", "float32"):
        import torch

        from .fp32 import _Ctx, apply, ensure_cuda_home
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ensure_cuda_home()
        return apply(model, _Ctx(next(model.parameters()).device, kwargs.pop("log", None)), **kwargs)
    if name in ("fp16", "float16", "half"):
        import torch

        from .fp16.transformers_api import apply
        return apply(model, dtype=torch.float16, **kwargs)
    raise ValueError(f"dtype must be {FP32!r} or {FP16!r}, got {dtype!r}")
