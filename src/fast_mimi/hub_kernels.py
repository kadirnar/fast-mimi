"""Expose fast-mimi's convolution kernel through the Hugging Face `kernels` library.

`kernels.kernelize(model)` swaps the forward of layer classes that were marked with
`use_kernel_forward_from_hub`.  `transformers`' Mimi classes are not marked, so `register()` marks `MimiConv1d`
itself and registers fast-mimi as the CUDA provider for it.  After that the standard entry point works:

    import fast_mimi.hub_kernels as hk
    from kernels import Mode, kernelize

    hk.register()
    model = kernelize(model, mode=Mode.INFERENCE)

This is the per-layer path, so it keeps the stock control flow and gives up what needs a whole-model view:
`fast_mimi.optimize(model)` additionally folds the activations between layers and captures encode/decode into
CUDA graphs, and is the faster of the two.  The kernels here are Triton and CUDA C++ compiled on first use, not
prebuilt Hub artefacts, so the repository below is local rather than a `LayerRepository` on the Hub.
"""
from __future__ import annotations

import torch

LAYER_NAME = "MimiConv1d"


class _Conv1dLayer(torch.nn.Module):
    """Drop-in forward for `transformers`' `MimiConv1d`, backed by the exact fp32 implicit-GEMM conv."""

    def forward(self, hidden_states, padding_cache=None):
        plan = getattr(self, "_fast_mimi_plan", None)
        conv = self.conv
        if (padding_cache is not None or not hidden_states.is_cuda or hidden_states.dtype != torch.float32
                or hidden_states.dim() != 3 or not self.causal or self.pad_mode != "constant"
                or conv.groups != 1 or conv.dilation != (1,) or conv.padding != (0,)):
            return self._fast_mimi_stock_forward(hidden_states, padding_cache)
        if plan is None:
            import math

            from .fp32.kernels.conv1d_fp32 import Conv1dPlan
            plan = self._fast_mimi_plan = Conv1dPlan(conv.weight, conv.bias, conv.stride[0])
            self._fast_mimi_geom = (int(self.kernel_size), int(self.stride), int(self.padding_total), math)
        k, s, pt, math = self._fast_mimi_geom
        length = hidden_states.shape[-1]
        extra = (math.ceil((length - k + pt) / s + 1) - 1) * s + k - pt - length
        return plan(hidden_states.contiguous(), pt, extra)


class _Repository:
    """`kernels`' repository protocol: anything with `load() -> type[nn.Module]`."""

    layer_name = LAYER_NAME

    def load(self):
        return _Conv1dLayer

    def __eq__(self, other):
        return isinstance(other, _Repository)

    def __hash__(self):
        return hash(LAYER_NAME)


def register() -> None:
    """Mark `MimiConv1d` for kernel replacement and register fast-mimi as its CUDA provider."""
    from kernels import Device, register_kernel_mapping, replace_kernel_forward_from_hub
    from transformers.models.mimi.modeling_mimi import MimiConv1d

    if not hasattr(MimiConv1d, "_fast_mimi_stock_forward"):
        MimiConv1d._fast_mimi_stock_forward = MimiConv1d.forward
        replace_kernel_forward_from_hub(MimiConv1d, LAYER_NAME)
    register_kernel_mapping({LAYER_NAME: {Device(type="cuda"): _Repository()}})
