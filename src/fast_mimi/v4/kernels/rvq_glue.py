"""Glue around the exact RVQ kernels: fused split-quantizer encode / decode.

The semantic and acoustic RVQs both project the same downsampled embeddings (Conv1d k=1, 512 -> 256, no bias), so
the two projections become ONE fp32 GEMM with the stacked weight [W_s; W_a] (512 x 512), computed directly in row
layout ((B*T, 512) = x^T @ Win^T) which the stage kernels read in place, and the codes of both RVQs are written
straight into one (K, B*T) buffer (no cat).  On decode the two gather sums are written by one launch into the two
halves of one (B, 512, T) buffer and the two output projections + the add become ONE GEMM with [W_s | W_a].
fp32 FMA (cuBLAS, TF32 off); only the accumulation order differs from the stock chain (~1e-7 relative).
"""
from __future__ import annotations

import types

import torch

from .rvq_fp32 import rvq_decode, rvq_decode_split, rvq_encode


def _split_encode(self, embeddings, num_quantizers=None):
    nq = self.max_num_quantizers if num_quantizers is None else num_quantizers
    if nq > self.max_num_quantizers:
        raise ValueError(f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {nq}.")
    if nq < self.num_semantic_quantizers:
        raise ValueError(f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {nq}.")
    x = embeddings
    if not x.is_cuda or x.dtype != torch.float32 or x.dim() != 3 or x.shape[1] != self._fk_win.shape[1]:
        return self._fk_encode(embeddings, num_quantizers)
    B, Cin, T = x.shape
    nq = int(nq)
    ns = self.num_semantic_quantizers
    dsem = self._fk_dsem
    # projection rows (B*T, Ds + Da): row n = frame (b, t); cuBLAS fp32 GEMM (TF32 off), no copy for B == 1
    xr = x[0].t() if B == 1 else x.permute(0, 2, 1).reshape(B * T, Cin)
    rows = torch.matmul(xr, self._fk_win_t)                                  # (B*T, 512), contiguous
    idx = torch.empty((nq, B * T), device=x.device, dtype=torch.int64)
    proj = rows.t().reshape(1, -1, B * T) if B == 1 else rows.view(B, T, -1).permute(0, 2, 1)
    rvq_encode(proj[:, :dsem], self.semantic_residual_vector_quantizer._fk_codebooks, ns, out=idx[:ns])
    if nq > ns:
        rvq_encode(proj[:, dsem:], self.acoustic_residual_vector_quantizer._fk_codebooks, nq - ns, out=idx[ns:nq])
    return idx.view(nq, B, T)


def _split_decode(self, codes):
    if not codes.is_cuda or codes.dim() != 3:
        return self._fk_decode(codes)
    B, K, T = codes.shape
    ns = self.num_semantic_quantizers
    dsem = self._fk_dsem
    buf = torch.empty((B, self._fk_wout.shape[1], T), device=codes.device, dtype=torch.float32)
    if K > ns:
        rvq_decode_split(codes, self.semantic_residual_vector_quantizer._fk_codebooks,
                         self.acoustic_residual_vector_quantizer._fk_codebooks, ns, buf)   # both halves, one launch
    else:
        rvq_decode(codes[:, :ns], self.semantic_residual_vector_quantizer._fk_codebooks, out=buf, offset=0)
        buf[:, dsem:].zero_()
    return torch.matmul(self._fk_wout, buf[0])[None] if B == 1 else torch.matmul(self._fk_wout, buf)


def patch_split_quantizer(model) -> int:
    """Stack the two RVQs' projections and replace MimiSplitResidualVectorQuantizer.encode/decode (after the two
    MimiResidualVectorQuantizer modules have their _fk_codebooks)."""
    from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer
    count = 0
    for m in model.modules():
        if not isinstance(m, MimiSplitResidualVectorQuantizer):
            continue
        sem, ac = m.semantic_residual_vector_quantizer, m.acoustic_residual_vector_quantizer
        ok = all(hasattr(q, "_fk_codebooks") for q in (sem, ac))
        for q in (sem, ac):
            ok = ok and q.input_proj is not None and q.output_proj is not None
            ok = ok and isinstance(q.input_proj, torch.nn.Conv1d) and q.input_proj.kernel_size == (1,) and q.input_proj.bias is None
            ok = ok and isinstance(q.output_proj, torch.nn.Conv1d) and q.output_proj.kernel_size == (1,) and q.output_proj.bias is None
            ok = ok and q.input_proj.stride == (1,) and q.output_proj.stride == (1,) and q.input_proj.groups == 1 and q.output_proj.groups == 1
        if not ok or sem.input_proj.weight.shape[0] != sem._fk_codebooks.shape[2] or ac.input_proj.weight.shape[0] != ac._fk_codebooks.shape[2]:
            continue
        with torch.no_grad():
            m._fk_win = torch.cat([sem.input_proj.weight[:, :, 0], ac.input_proj.weight[:, :, 0]], 0).detach().float().contiguous()    # (Ds + Da, Cin)
            m._fk_win_t = m._fk_win.t().contiguous()                                                                                  # (Cin, Ds + Da)
            m._fk_wout = torch.cat([sem.output_proj.weight[:, :, 0], ac.output_proj.weight[:, :, 0]], 1).detach().float().contiguous()  # (Cout, Ds + Da)
        m._fk_dsem = int(sem.input_proj.weight.shape[0])
        m._fk_encode, m._fk_decode = m.encode, m.decode
        m.encode = types.MethodType(_split_encode, m)
        m.decode = types.MethodType(_split_decode, m)
        count += 1
    return count
