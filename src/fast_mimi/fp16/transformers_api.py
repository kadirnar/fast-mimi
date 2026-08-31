"""Drive the fp16 Triton backend through the `transformers.MimiModel` API.

The backend keeps the weights in its own packed layout, so it is built once from `model.state_dict()` (the key
names are identical to the published checkpoint) and then bound to the model's `encode` / `decode`.  Signatures
and return types stay the transformers ones, so code written against `MimiModel` does not change.
"""
from __future__ import annotations

import types

import torch


def _encode(self, input_values, padding_mask=None, *args, **kwargs):
    from transformers.models.mimi.modeling_mimi import MimiEncoderOutput
    backend = self._fast_mimi_fp16
    if args or kwargs or not input_values.is_cuda or input_values.shape[0] != 1:
        return self._fast_mimi_encode(input_values, padding_mask, *args, **kwargs)
    return MimiEncoderOutput(audio_codes=backend.encode(input_values))


def _decode(self, audio_codes, padding_mask=None, *args, **kwargs):
    from transformers.models.mimi.modeling_mimi import MimiDecoderOutput
    backend = self._fast_mimi_fp16
    if args or kwargs or not audio_codes.is_cuda or audio_codes.shape[0] != 1:
        return self._fast_mimi_decode(audio_codes, padding_mask, *args, **kwargs)
    # transformers truncates to the mask when one is given, and returns the full decoder output otherwise
    length = padding_mask.shape[-1] if padding_mask is not None else None
    return MimiDecoderOutput(audio_values=backend.decode(audio_codes, length=length))


def apply(model, dtype=torch.float16, **kw):
    """Bind the fp16 backend to `model` in place and return it."""
    from .backends import build
    state = {k: v.detach().to("cuda") for k, v in model.state_dict().items()}
    model._fast_mimi_fp16 = build("graph+triton", state, dtype=dtype, **kw)
    model._fast_mimi_encode, model._fast_mimi_decode = model.encode, model.decode
    model.encode = types.MethodType(_encode, model)
    model.decode = types.MethodType(_decode, model)
    return model
