"""fast-mimi v4 runtime (ported from the fast-kernel campaign candidate): CUDA-graph capture of encode()/decode() + host-side conv padding math.

Experiment #1 showed that `MimiConv1d._get_extra_padding_for_conv1d` returns a CUDA scalar tensor
(kernel_size / stride / padding_total are int64 buffers on the GPU); `F.pad` converts it with
`.item()`, which is a host sync — illegal during graph capture and a needless GPU->CPU round trip
per conv layer in eager mode. Experiment #2 showed `padding_left/right` are meta tensors after
from_pretrained. We compute the same paddings with Python ints (identical values) and capture each
encode()/decode() call into a CUDA graph per input shape.

Experiment #4: the SEANet nn.Conv1d layers run through a Triton fp32 implicit GEMM
(kernels/conv1d_fp32.py). The profile after #3 showed cuDNN's precomputed_convolve_sgemm 5-10x off the
weight-bandwidth floor on the small-T / big-weight layers (encoder.layers.12.conv: 498 us for a 33 MB
weight); a deterministic split-K implicit GEMM streams those weights with enough CTAs in flight.

Experiment #5: the 32-stage residual vector quantizer (cdist + argmin + gather + subtract, ~17 launches per
stage, 1.24 ms) runs through two Triton kernels per stage (exact fp32 distance + block argmin, then
reduce + residual update) and one gather-and-sum kernel for decode (kernels/rvq_fp32.py).
"""
from __future__ import annotations

import math
import types

import torch

from ._compat import Graphed

from .kernels.conv1d_fp32 import conv1d_fp32
from .kernels.rvq_fp32 import rvq_decode, rvq_encode
from .kernels.transformer_fp32 import patch_transformers
from .kernels.resnet_fp32 import patch_resnet_blocks

_STATE = {"active": False, "kernels": ["cuda-graph:encode", "cuda-graph:decode", "host-padding:MimiConv1d",
                                       "triton:conv1d_fp32 (implicit GEMM, space-to-channel, ConvT phases, fused pads/ELU)",
                                       "cublas:im2col_gemm (3 strided layers)", "cuda:rvq_stage (PDL, next-slab L2 prefetch)",
                                       "cuda:rvq_tiled (register-tiled search past 112 frames)", "triton:rvq_decode",
                                       "cublas:transformer_qkv_fc1_fc2", "cuda:attention_rope_oproj (+FC prefetch, M <= 32)",
                                       "triton:attn_window (windowed fp32 attention past M = 32)",
                                       "triton:transformer_row_kernels", "cuda:resnet_block (64/128 ch, ELU epilogue)",
                                       "triton:tail_conv1d (single output channel)", "cublas:split_quantizer_proj"],
          "invocations": 0, "graphs": 0, "patched_convs": 0, "triton_convs": 0, "rvq_modules": 0}


def _host_padding_forward(self, hidden_states, padding_cache=None):
    """Same semantics as MimiConv1d.forward, with the padding arithmetic on the host (no .item())."""
    k, s, pt = self._fk_kernel, self._fk_stride, self._fk_padding_total
    length = hidden_states.shape[-1]
    n_frames = math.ceil((length - k + pt) / s + 1) - 1
    extra = n_frames * s + k - pt - length
    elu = getattr(self, "_fk_elu_in", False)                        # set by _seanet_forward: input is ELU(hidden_states)
    elu_out = getattr(self, "_fk_elu_out", False)                   # the consumer wants ELU(output) (emitted by the epilogue)
    dual = getattr(self, "_fk_dual_elu", False)                     # the next chain block wants (output, ELU(output))
    self._fk_elu_result = None
    plan = getattr(self.conv, "_fk_plan", None)
    fused = plan is not None and self.causal and padding_cache is None and self.pad_mode == "constant" and _triton_ok(hidden_states)
    if elu and not fused:
        hidden_states = torch.nn.functional.elu(hidden_states)
    if not self.causal and padding_cache is not None:
        raise ValueError("`padding_cache` is not supported for non-causal convolutions.")
    if self.causal and padding_cache is not None:
        layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
        hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
    elif self.causal:
        if fused:
            if dual:
                out, self._fk_elu_result = plan(hidden_states.contiguous(), pt, extra, elu=elu, dual_elu=True)
                return out
            return plan(hidden_states.contiguous(), pt, extra, elu=elu, elu_out=elu_out)   # padding (and ELUs) fused into the kernel
        hidden_states = self._pad1d(hidden_states, (pt, extra), mode=self.pad_mode)
    else:
        hidden_states = self._pad1d(hidden_states, (self._fk_padding_left, self._fk_padding_right + extra), mode=self.pad_mode)
    out = self.conv(hidden_states)
    return torch.nn.functional.elu(out) if elu_out else out


def _triton_ok(x) -> bool:
    return x.is_cuda and x.dtype == torch.float32 and x.dim() == 3


def _triton_conv_forward(self, x):
    """nn.Conv1d.forward: exact fp32 Triton implicit GEMM (prepacked plan), cuDNN for anything unusual."""
    if _triton_ok(x):
        return self._fk_plan(x.contiguous(), 0, 0)
    return self._conv_forward(x, self.weight, self.bias)


def _patch_nn_convs(model) -> int:
    from .kernels.conv1d_fp32 import Conv1dPlan
    count = 0
    for module in model.modules():
        if (type(module) is torch.nn.Conv1d and module.groups == 1 and module.dilation == (1,)
                and module.padding == (0,) and module.padding_mode == "zeros"):
            module._fk_plan = Conv1dPlan(module.weight, module.bias, module.stride[0])
            module.forward = types.MethodType(_triton_conv_forward, module)
            count += 1
    if count:
        _STATE["kernels"] += ["triton:s2c_pad", "triton:splitk_reduce"]
    return count


def _convt_forward(self, hidden_states):
    """MimiConvTranspose1d.forward: exact fp32 implicit GEMM with the causal right trim folded in."""
    elu = getattr(self, "_fk_elu_in", False)
    elu_out = getattr(self, "_fk_elu_out", False)
    dual = getattr(self, "_fk_dual_elu", False)
    self._fk_elu_result = None
    if _triton_ok(hidden_states):
        if dual:
            out, self._fk_elu_result = self._fk_plan(hidden_states.contiguous(), elu=elu, dual_elu=True)
            return out
        return self._fk_plan(hidden_states.contiguous(), elu=elu, elu_out=elu_out)
    out = self._fk_forward(torch.nn.functional.elu(hidden_states) if elu else hidden_states)
    return torch.nn.functional.elu(out) if elu_out else out


def _patch_convt(model) -> int:
    from transformers.models.mimi.modeling_mimi import MimiConvTranspose1d
    from .kernels.conv1d_fp32 import ConvT1dPlan
    count = 0
    for module in model.modules():
        conv = getattr(module, "conv", None)
        if (isinstance(module, MimiConvTranspose1d) and type(conv) is torch.nn.ConvTranspose1d and conv.groups == 1
                and conv.dilation == (1,) and conv.padding == (0,) and conv.output_padding == (0,)
                and conv.kernel_size[0] == 2 * conv.stride[0] and module.padding_left == 0
                and module.padding_right == conv.kernel_size[0] - conv.stride[0]):
            module._fk_plan = ConvT1dPlan(conv.weight, conv.bias, conv.stride[0])
            module._fk_forward = module.forward
            module.forward = types.MethodType(_convt_forward, module)
            count += 1
    if count:
        _STATE["kernels"].append("triton:convt1d_fp32")
    return count


def _is_elu(m) -> bool:
    return type(m) is torch.nn.ELU and m.alpha == 1.0 and not m.inplace


def _conv_out_len(conv, T: int):
    """(output length, (pad_left, pad_right)) of a plan-backed MimiConv1d / MimiConvTranspose1d on the fused path, else None."""
    from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d
    if isinstance(conv, MimiConv1d) and hasattr(conv.conv, "_fk_plan") and conv.causal and conv.pad_mode == "constant":
        k, s, pt = conv._fk_kernel, conv._fk_stride, conv._fk_padding_total
        extra = (math.ceil((T - k + pt) / s + 1) - 1) * s + k - pt - T
        return conv.conv._fk_plan.out_len(T, pt, extra), (pt, extra)
    if isinstance(conv, MimiConvTranspose1d) and hasattr(conv, "_fk_plan"):
        return T * conv._fk_plan.stride, None
    return None


def _chain_block(block, C: int, T: int) -> bool:
    """True when this resnet block runs the conv chain (not the fused kernel) at (C, T)."""
    from .kernels.resnet_fp32 import _cfg as resnet_cfg
    return getattr(block, "_fk_c", None) == C and resnet_cfg(C, T) < 0 and hasattr(block.block[1].conv, "_fk_plan")


def _elu_free_at(conv, B: int, T: int) -> bool:
    """Whether the consumer conv folds its input ELU for free at this shape (s2c / im2col) or would launch F.elu."""
    from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d
    r = _conv_out_len(conv, T)
    if r is None:
        return True
    if isinstance(conv, MimiConv1d):
        return conv.conv._fk_plan.elu_fold_free(B, T, *r[1])
    return conv._fk_plan.elu_fold_free(B, T)


def _seanet_forward(self, hidden_states, padding_cache=None):
    """MimiEncoder/MimiDecoder.forward with the standalone nn.ELUs folded into neighbouring kernels (bit-exact expm1):
    into the following conv's input load / copy where that is free, otherwise into the producing conv's or chain block's
    epilogue; a conv feeding a chain block also emits ELU(out) for the block's k3 conv from its epilogue.  Every fold is a
    pure launch removal (same FMAs, same order); anything unusual runs the stock sequence."""
    from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiResnetBlock
    layers = list(self.layers)
    n = len(layers)
    i = 0
    while i < n:
        layer = layers[i]
        nxt = layers[i + 1] if i + 1 < n else None
        fast = padding_cache is None and _triton_ok(hidden_states)
        # the module about to run: a conv reached through a foldable [ELU, conv] pair, or the layer itself
        if fast and _is_elu(layer) and nxt is not None and getattr(nxt, "_fk_fold_elu", False):
            j, target, elu_in = i + 1, nxt, True
        else:
            j, target, elu_in = i, layer, False
        if fast and isinstance(target, (MimiConv1d, MimiConvTranspose1d, MimiResnetBlock)):
            B, C, T = hidden_states.shape
            is_block = isinstance(target, MimiResnetBlock)
            r = None if is_block else _conv_out_len(target, T)
            block_chain = is_block and _chain_block(target, C, T)
            # a patched resnet block preserves T and emits ELU(out) from its epilogue (fused kernel or chain);
            # an unpatched one would silently drop the folded ELU, so it stays on the stock ELU module
            T_out = T if (is_block and getattr(target, "_fk_c", None) == C) else (r[0] if r is not None else None)
            a1 = layers[j + 1] if j + 1 < n else None
            a2 = layers[j + 2] if j + 2 < n else None
            dual = False
            if r is not None and isinstance(a1, MimiResnetBlock) and _chain_block(a1, target.conv.out_channels, r[0]):
                plan = target.conv._fk_plan if isinstance(target, MimiConv1d) else target._fk_plan
                dual = plan.dual_ok(B, T, *r[1]) if isinstance(target, MimiConv1d) else plan.dual_ok(B, T)
            elu_out = (T_out is not None and _is_elu(a1) and a2 is not None and getattr(a2, "_fk_fold_elu", False)
                       and not _elu_free_at(a2, B, T_out))
            target._fk_elu_in, target._fk_dual_elu, target._fk_elu_out = elu_in, dual, elu_out
            try:
                hidden_states = target(hidden_states, padding_cache=padding_cache) if isinstance(target, (MimiConv1d, MimiResnetBlock)) else target(hidden_states)
            finally:
                target._fk_elu_in, target._fk_dual_elu, target._fk_elu_out = False, False, False
            if dual:
                a1._fk_elu_input = getattr(target, "_fk_elu_result", None)
                target._fk_elu_result = None
            i = j + (2 if elu_out else 1)                  # elu_out: the ELU module after the target is consumed too
            continue
        if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        else:
            hidden_states = layer(hidden_states)
        i += 1
    return hidden_states


def _patch_seanet_elu(model) -> int:
    from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiDecoder, MimiEncoder
    count = 0
    for seanet in (getattr(model, "encoder", None), getattr(model, "decoder", None)):
        if not isinstance(seanet, (MimiEncoder, MimiDecoder)):
            continue
        layers = list(seanet.layers)
        for layer, nxt in zip(layers, layers[1:]):
            if type(layer) is torch.nn.ELU and ((isinstance(nxt, MimiConv1d) and hasattr(nxt.conv, "_fk_plan"))
                                                 or (isinstance(nxt, MimiConvTranspose1d) and hasattr(nxt, "_fk_plan"))):
                nxt._fk_fold_elu = True
                count += 1
        seanet.forward = types.MethodType(_seanet_forward, seanet)
    if count:
        _STATE["kernels"] += ["triton:elu_on_load", "fused:elu_in_epilogue", "fused:dual_elu_output"]
    return count


def _rvq_encode_forward(self, embeddings, num_quantizers=None):
    if not embeddings.is_cuda or embeddings.dtype != torch.float32:
        return self._fk_encode(embeddings, num_quantizers)
    if self.input_proj is not None:
        embeddings = self.input_proj(embeddings)
    num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers
    return rvq_encode(embeddings, self._fk_codebooks, num_quantizers)


def _rvq_decode_forward(self, codes):
    if not codes.is_cuda:
        return self._fk_decode(codes)
    quantized_out = rvq_decode(codes, self._fk_codebooks)
    if self.output_proj is not None:
        quantized_out = self.output_proj(quantized_out)
    return quantized_out


def _patch_rvq(model) -> int:
    from transformers.models.mimi.modeling_mimi import MimiResidualVectorQuantizer
    count = 0
    for module in model.modules():
        if isinstance(module, MimiResidualVectorQuantizer):
            # `embed` is the reference's own (cached) embed_sum / cluster_usage tensor: identical values
            module._fk_codebooks = torch.stack([layer.codebook.embed for layer in module.layers]).contiguous()
            from .kernels.rvq_fp32 import rvq_prepare
            rvq_prepare(module._fk_codebooks)      # builds the exact-search extension before warm-up/capture
            module._fk_encode, module._fk_decode = module.encode, module.decode
            module.encode = types.MethodType(_rvq_encode_forward, module)
            module.decode = types.MethodType(_rvq_decode_forward, module)
            count += 1
    from .kernels.rvq_glue import patch_split_quantizer
    count += patch_split_quantizer(model)          # one stacked projection GEMM per direction for both RVQs
    return count


def _patch_convs(model, ctx) -> int:
    from transformers.models.mimi.modeling_mimi import MimiConv1d
    count = 0
    for module in model.modules():
        if isinstance(module, MimiConv1d):
            module._fk_kernel = int(module.kernel_size)
            module._fk_stride = int(module.stride)
            module._fk_padding_total = int(module.padding_total)
            # padding_left/right are meta tensors after from_pretrained: derive them from the int padding_total
            module._fk_padding_right = module._fk_padding_total // 2
            module._fk_padding_left = module._fk_padding_total - module._fk_padding_right
            module.forward = types.MethodType(_host_padding_forward, module)
            count += 1
    return count


def apply(model, ctx):
    from transformers.models.mimi.modeling_mimi import MimiDecoderOutput, MimiEncoderOutput

    _STATE["patched_convs"] = _patch_convs(model, ctx)
    ctx.log(f"host-side padding math on {_STATE['patched_convs']} MimiConv1d layers")
    _STATE["triton_convs"] = _patch_nn_convs(model)
    ctx.log(f"Triton fp32 implicit-GEMM conv1d on {_STATE['triton_convs']} nn.Conv1d layers")
    _STATE["triton_convt"] = _patch_convt(model)
    ctx.log(f"Triton fp32 implicit-GEMM ConvTranspose1d on {_STATE['triton_convt']} MimiConvTranspose1d layers")
    _STATE["elu_folded"] = _patch_seanet_elu(model)
    ctx.log(f"{_STATE['elu_folded']} standalone SEANet ELUs folded into the following conv's input load")
    _STATE["rvq_modules"] = _patch_rvq(model)
    ctx.log(f"fused Triton RVQ search/decode on {_STATE['rvq_modules']} residual vector quantizers")
    encode, decode = model.encode, model.decode
    graphs: dict[tuple, Graphed] = {}

    def graphed(kind, fn, *tensors):
        key = (kind, tuple((tuple(t.shape), str(t.dtype)) for t in tensors))
        g = graphs.get(key)
        if g is None:
            g = Graphed(fn, tensors)
            graphs[key] = g
            _STATE["graphs"] = len(graphs)
            ctx.log(f"captured {kind} graph for {key[1]}")
        _STATE["invocations"] += 1
        return g(*tensors)

    ones: dict[tuple, torch.Tensor] = {}

    def ones_mask(x):
        """The all-ones mask `MimiModel.encode` builds itself when none is given (cached: allocating one per call
        would be a kernel launch on the hot path, and it must exist before capture)."""
        key = (tuple(x.shape), x.device)
        m = ones.get(key)
        if m is None:
            m = ones[key] = torch.ones(x.shape, dtype=torch.bool, device=x.device)
        return m

    def fast_encode(input_values, padding_mask=None, *args, **kwargs):
        if args or kwargs or not input_values.is_cuda:
            return encode(input_values, padding_mask, *args, **kwargs)     # anything unusual -> reference path
        # encode never reads the mask (modeling_mimi._encode_frame takes it and does not use it), and the reference
        # substitutes ones_like(input_values) for None -- so a missing mask is the same graph, not the eager path
        mask = padding_mask if padding_mask is not None else ones_mask(input_values)
        codes = graphed("encode", lambda a, m: encode(a, m).audio_codes, input_values, mask)
        return MimiEncoderOutput(audio_codes=codes.clone())

    def fast_decode(audio_codes, padding_mask=None, *args, **kwargs):
        if args or kwargs or not audio_codes.is_cuda:
            return decode(audio_codes, padding_mask, *args, **kwargs)
        if padding_mask is None:                                           # no mask -> no truncation: its own graph
            audio = graphed("decode-untruncated", lambda c: decode(c, None).audio_values, audio_codes)
        else:
            audio = graphed("decode", lambda c, m: decode(c, m).audio_values, audio_codes, padding_mask)
        return MimiDecoderOutput(audio_values=audio.clone())

    _STATE["transformer"] = patch_transformers(model, ctx)
    ctx.log(f"fused fp32 transformer layers: {_STATE['transformer']}")
    model.encode = fast_encode
    model.decode = fast_decode
    _STATE["resnet"] = patch_resnet_blocks(model, ctx)
    ctx.log(f"fused SEANet resnet blocks: {_STATE['resnet']}")
    _STATE["active"] = True
    ctx.log("encode/decode wrapped in shape-bucketed CUDA graphs")
    return model


def report() -> dict:
    return dict(_STATE)
