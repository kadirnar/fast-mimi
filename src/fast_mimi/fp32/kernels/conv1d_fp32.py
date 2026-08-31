"""fp32 implicit-GEMM Conv1d / ConvTranspose1d (groups=1, dilation=1) for Mimi's SEANet convs.

Conv1d, GEMM view (M = Cout, N = T_out, K = Cin*KS):
    out[b, co, t] = bias[co] + sum_kk A[co, kk] * B[kk, t]
The input is addressed virtually as [pad_left zeros | x | pad_right zeros] (masked loads, no F.pad copy).
Two B-operand addressings, both the same fp32 FMAs in the same (ci, tap) order as the reference:
  MODE_DIRECT  B[kk, t] = x[ci, t*stride + kt - pad_left]                (stride-1 layers: contiguous along t)
  MODE_S2C     stride s, KS = 2s: x is first re-laid out as x'[ci*s + j, t'] = xpad[ci, t'*s + j]
               (one fused pad + space-to-channel copy), then B[kk, t] = x'[ci*s + j, t + q] with
               kt = q*s + j -- a stride-1, 2-tap conv whose loads are contiguous along t.
ConvTranspose1d (k = 2s, stride s, causal right trim = s) is the same 2-tap stride-1 GEMM over
[0 | x] with M = Cout*s rows (co, j) and a phase-interleaved store out[co, t*s + j] (PHASES = s).
Accumulation: tl.dot(input_precision="ieee") -> exact fp32 FMA; split-K writes partials to a
(SPLIT, B, M, N) buffer reduced in fixed order (no atomics) -> deterministic.
Tile configs come from candidate/tuned/conv1d_fp32.json (graph-replay sweeps per layer shape).
"""
from __future__ import annotations

import json
import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MODE_DIRECT, MODE_S2C = 0, 1


@triton.jit
def _elu(v):
    """ELU(alpha=1) exactly as ATen computes it (expm1 on the negative side); zeros (padding) stay zero."""
    return tl.where(v > 0, v, libdevice.expm1(v))


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, bias_ptr, res_ptr, out_ptr, elu_ptr,
                   Cin, T_x, pad_left, M, N, KK, stride_s, k_per_split,
                   KS: tl.constexpr, S: tl.constexpr, MODE: tl.constexpr, ELU_IN: tl.constexpr, HAS_RES: tl.constexpr,
                   ELU_OUT: tl.constexpr, DUAL_ELU: tl.constexpr, HAS_BIAS: tl.constexpr, SPLIT: tl.constexpr, PHASES: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bz = tl.program_id(2)
    b = pid_bz // SPLIT
    z = pid_bz % SPLIT
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = rm < M
    nmask = rn < N
    k_begin = z * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, KK)
    x_base = x_ptr + b * Cin * T_x
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BK):
        kk = k0 + rk
        kmask = kk < k_end
        a = tl.load(w_ptr + rm[:, None] * KK + kk[None, :], mask=mmask[:, None] & kmask[None, :], other=0.0)
        ci = kk // KS
        kt = kk - ci * KS
        if MODE == 0:
            row = ci
            p = rn[None, :] * stride_s + (kt[:, None] - pad_left)
        else:
            q = kt // S
            row = ci * S + (kt - q * S)
            p = rn[None, :] + (q[:, None] - pad_left)
        bv = tl.load(x_base + row[:, None] * T_x + p, mask=kmask[:, None] & nmask[None, :] & (p >= 0) & (p < T_x), other=0.0)
        if ELU_IN:
            bv = _elu(bv)                                   # short-T chain convs only: ELU(x) computed on load (zeros stay zero)
        acc = tl.dot(a, bv, acc, input_precision="ieee")
    omask = mmask[:, None] & nmask[None, :]
    if SPLIT == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + rm, mask=mmask, other=0.0)[:, None]
        if HAS_RES:                                         # residual + (conv + bias): the stock add, one fp32 add
            acc = tl.load(res_ptr + (b * M + rm[:, None]) * N + rn[None, :], mask=omask, other=0.0) + acc
        if ELU_OUT:                                         # the block-internal ELU on the conv output (once per element)
            acc = _elu(acc)
        if PHASES == 1:
            offs = (b * M + rm[:, None]) * N + rn[None, :]
        else:
            co = rm // PHASES
            j = rm - co * PHASES
            offs = (b * (M // PHASES) + co[:, None]) * (N * PHASES) + rn[None, :] * PHASES + j[:, None]
        tl.store(out_ptr + offs, acc, mask=omask)
        if DUAL_ELU:                                        # second output ELU(out) for the next block's k3 conv
            tl.store(elu_ptr + offs, _elu(acc), mask=omask)
    else:
        nb = tl.num_programs(2) // SPLIT
        tl.store(out_ptr + ((z * nb + b) * M + rm[:, None]) * N + rn[None, :], acc, mask=omask)


@triton.jit
def _splitk_reduce_kernel(part_ptr, bias_ptr, res_ptr, out_ptr, elu_ptr, B, M, N,
                          HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr, ELU_OUT: tl.constexpr, DUAL_ELU: tl.constexpr,
                          SPLIT: tl.constexpr, PHASES: tl.constexpr, BLOCK: tl.constexpr):
    """out[b] = (+ residual) sum_z part[z, b] (+ bias), summed in fixed z order; optional phase-interleaved store."""
    b = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mn = M * N
    mask = i < mn
    m = i // N
    n = i - m * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for z in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + (z * B + b) * mn + i, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + m, mask=mask, other=0.0)
    if HAS_RES:
        acc = tl.load(res_ptr + b * mn + i, mask=mask, other=0.0) + acc
    if ELU_OUT:
        acc = _elu(acc)
    if PHASES == 1:
        offs = b * mn + i
    else:
        co = m // PHASES
        j = m - co * PHASES
        offs = (b * (M // PHASES) + co) * (N * PHASES) + n * PHASES + j
    tl.store(out_ptr + offs, acc, mask=mask)
    if DUAL_ELU:
        tl.store(elu_ptr + offs, _elu(acc), mask=mask)


@triton.jit
def _s2c_pad_kernel(x_ptr, y_ptr, T_x, pad_left, T_new,
                    S: tl.constexpr, SP: tl.constexpr, ELU_IN: tl.constexpr, BT: tl.constexpr):
    """y[c*S + j, t'] = xpad[c, t'*S + j] for one (b, c) row block; xpad = [pad_left zeros | x | zeros]."""
    c = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    j = tl.arange(0, SP)
    p = t[:, None] * S + (j[None, :] - pad_left)
    valid = (t[:, None] < T_new) & (j[None, :] < S)
    v = tl.load(x_ptr + c * T_x + p, mask=valid & (p >= 0) & (p < T_x), other=0.0)
    if ELU_IN:
        v = _elu(v)
    tl.store(y_ptr + (c * S + j[None, :]) * T_new + t[:, None], v, mask=valid)


@triton.jit
def _im2col_kernel(x_ptr, out_ptr, Cin, T_x, pad_left, KK, KB, N, stride_s,
                   KS: tl.constexpr, ELU_IN: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    """Explicit im2col of the virtually padded input for a cuBLAS GEMM.

    cols[b, kk, n] = xpad[b, kk // KS, n*stride + kk % KS]  for kk < KK; row KK = 1.0 (bias row), rows > KK = 0
    (KB >= KK + 1 keeps the leading dimension aligned).
    """
    b = tl.program_id(2)
    kk = tl.program_id(0) * BK + tl.arange(0, BK)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    ci = kk // KS
    kt = kk - ci * KS
    p = rn[None, :] * stride_s + (kt[:, None] - pad_left)
    nmask = rn < N
    v = tl.load(x_ptr + (b * Cin + ci[:, None]) * T_x + p,
                mask=(kk < KK)[:, None] & nmask[None, :] & (p >= 0) & (p < T_x), other=0.0)
    if ELU_IN:
        v = _elu(v)
    v = tl.where((kk == KK)[:, None], 1.0, v)
    omask = (kk < KB)[:, None] & nmask[None, :]
    tl.store(out_ptr + (b * KB + kk[:, None]) * N + rn[None, :], v, mask=omask)


@triton.jit
def _epilogue_kernel(src_ptr, bias_ptr, out_ptr, elu_ptr, B, M, N,
                     HAS_BIAS: tl.constexpr, PHASES: tl.constexpr, SPLIT: tl.constexpr,
                     DUAL_ELU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """out[b, m, n] (phase-interleaved when PHASES > 1) = sum_z src[z, b, n, m] (the GEMM result, transposed) + bias[m].

    src holds SPLIT partial GEMM results (K chunks); they are summed in fixed z order -> deterministic."""
    b = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    v = tl.zeros((BM, BN), dtype=tl.float32)
    for z in tl.static_range(SPLIT):
        t = tl.load(src_ptr + ((z * B + b) * N + rn[:, None]) * M + rm[None, :], mask=nmask[:, None] & mmask[None, :], other=0.0)
        v += tl.trans(t)
    if HAS_BIAS:
        v += tl.load(bias_ptr + rm, mask=mmask, other=0.0)[:, None]
    omask = mmask[:, None] & nmask[None, :]
    if PHASES == 1:
        offs = (b * M + rm[:, None]) * N + rn[None, :]
    else:
        co = rm // PHASES
        j = rm - co * PHASES
        offs = (b * (M // PHASES) + co[:, None]) * (N * PHASES) + rn[None, :] * PHASES + j[:, None]
    tl.store(out_ptr + offs, v, mask=omask)
    if DUAL_ELU:
        tl.store(elu_ptr + offs, _elu(v), mask=omask)


# ----------------------------------------------------------------------------------------------- configs

_TUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tuned", "conv1d_fp32.json")
_TUNED: dict[str, dict] | None = None


@triton.jit
def _tail_conv_kernel(x_ptr, w_ptr, out_ptr, bias_ptr, T, N,
                      CIN: tl.constexpr, KS: tl.constexpr, ELU_IN: tl.constexpr, HAS_BIAS: tl.constexpr,
                      CB: tl.constexpr, BT: tl.constexpr):
    """out[t] = bias + sum_ci sum_k w[ci, k] * xpad[ci, t + k], xpad = [KS-1 zeros | x] (causal, stride 1, Cout = 1).

    The implicit GEMM puts this layer's single output channel in a BM = 16 tile and re-reads x once per tap through
    the K axis; the decoder's last conv (64 -> 1 over 600k samples) therefore ran at 27% of its own bandwidth floor.
    Here the output tile is the only tile: x streams once per (channel block, tap) with contiguous BT-wide loads,
    so the DRAM traffic is exactly the 64 x T input read.
    """
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    m = t < N
    acc = tl.zeros((BT,), dtype=tl.float32)
    for c0 in range(0, CIN, CB):
        ci = c0 + tl.arange(0, CB)
        for k in tl.static_range(KS):
            p = t + k - (KS - 1)
            v = tl.load(x_ptr + ci[:, None] * T + p[None, :], mask=m[None, :] & (p >= 0)[None, :] & (p < T)[None, :], other=0.0)
            if ELU_IN:
                v = _elu(v)
            w = tl.load(w_ptr + ci * KS + k)
            acc += tl.sum(v * w[:, None], 0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr)
    tl.store(out_ptr + t, acc, mask=m)


def _tail_conv(x: torch.Tensor, w: torch.Tensor, bias, out: torch.Tensor, KS: int, elu: bool) -> torch.Tensor:
    """x: (1, Cin, T) contiguous; w: (Cin * KS,) row (ci, k); out: (1, 1, N)."""
    Cin, T = x.shape[1], x.shape[2]
    N = out.shape[2]
    CB = 8 if Cin >= 8 else 1
    BT = 1024
    _tail_conv_kernel[(triton.cdiv(N, BT),)](x, w, out, bias if bias is not None else x, T, N,
                                             CIN=Cin, KS=KS, ELU_IN=elu, HAS_BIAS=bias is not None, CB=CB, BT=BT, num_warps=4)
    return out


def _load_tuned() -> dict[str, dict]:
    global _TUNED
    if _TUNED is None:
        try:
            with open(_TUNED_PATH) as f:
                _TUNED = json.load(f)
        except (OSError, ValueError):
            _TUNED = {}
    return _TUNED


def _key(kind: str, B: int, M: int, N: int, KK: int, stride: int) -> str:
    return f"{kind}|B{B}|M{M}|N{N}|K{KK}|s{stride}"


def _lookup(kind: str, B: int, M: int, N: int, KK: int, stride: int) -> dict | None:
    """Tuned config for this exact shape, else the tuned entry of the same layer with the nearest N (any N is valid)."""
    tuned = _load_tuned()
    ent = tuned.get(_key(kind, B, M, N, KK, stride))
    if ent is not None:
        return ent
    best, best_d = None, None
    prefix, suffix = f"{kind}|B", f"|M{M}|N"
    for key, val in tuned.items():
        if not key.startswith(prefix) or suffix not in key or not key.endswith(f"|K{KK}|s{stride}"):
            continue
        b_str, rest = key[len(prefix):].split("|M", 1)
        n_str = rest.split("|N", 1)[1].split("|", 1)[0]
        d = abs(math.log((int(n_str) + 1) / (N + 1))) + (0.0 if int(b_str) == B else 0.5)
        if best_d is None or d < best_d:
            best, best_d = val, d
    return best


def _entries(kind: str, M: int, KK: int, stride: int) -> list[dict]:
    """All tuned entries of one layer (any B, N)."""
    return [val for key, val in _load_tuned().items()
            if key.startswith(f"{kind}|B") and f"|M{M}|N" in key and key.endswith(f"|K{KK}|s{stride}")]


def _pow2_at_least(n: int, lo: int, hi: int) -> int:
    v = lo
    while v < n and v < hi:
        v *= 2
    return v


def _n_sm(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _default_config(B: int, M: int, N: int, KK: int, n_sm: int) -> dict:
    """Fallback (BM, BN, BK, SPLIT, num_warps, num_stages) for shapes without a tuned entry."""
    if N <= 64:
        BN = _pow2_at_least(N, 16, 64)
        if KK <= 1024:                        # k=1 projections: one CTA per 16 rows, no split
            return dict(BM=16, BN=BN, BK=64, SPLIT=1, num_warps=4, num_stages=3)
        BM = 64 if M >= 64 else 16
        tiles = B * triton.cdiv(M, BM) * triton.cdiv(N, BN)
        split = 1
        while tiles * split < 2 * n_sm and split < 8 and KK // (split * 2) >= 4 * 64:
            split *= 2
        return dict(BM=BM, BN=BN, BK=64, SPLIT=split, num_warps=4, num_stages=3)
    BM = 32 if M >= 32 else 16
    if N <= 1024:
        return dict(BM=BM, BN=32, BK=64, SPLIT=1, num_warps=4, num_stages=3)
    if KK <= 256:
        return dict(BM=BM, BN=128, BK=16, SPLIT=1, num_warps=4, num_stages=3)
    return dict(BM=BM, BN=64, BK=32, SPLIT=1, num_warps=4, num_stages=3)


def _drop_useless_split(cfg: dict, B: int, M: int, N: int, device) -> dict:
    """Split-K only pays when the (M, N) tiling cannot fill the machine.  The tuned table was measured on ~1 s
    inputs, and `_lookup` reuses a layer's entry at any N, so long-form shapes inherit a SPLIT that now costs a
    (SPLIT, B, M, N) partial buffer plus a reduction pass -- 245 MB and 0.4 ms for one 100 s layer.  Drop it once
    the plain grid already has two tiles per SM (the accumulation becomes a single fixed-order chain, still
    deterministic)."""
    if cfg.get("backend") == "cublas" or cfg.get("SPLIT", 1) <= 1:
        return cfg
    tiles = B * triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"])
    if tiles >= 2 * _n_sm(device):
        cfg = dict(cfg)
        cfg["SPLIT"] = 1
    return cfg


# ----------------------------------------------------------------------------------------------- launchers

def _launch_gemm(x: torch.Tensor, w: torch.Tensor, bias, out: torch.Tensor, *, Cin_rows: int, T_x: int, pad_left: int,
                 M: int, N: int, KK: int, stride: int, KS: int, S: int, mode: int, phases: int, cfg: dict, B: int,
                 elu: bool = False, residual: torch.Tensor | None = None, elu_out: bool = False, dual: torch.Tensor | None = None):
    BM, BN, BK, split = cfg["BM"], cfg["BN"], cfg["BK"], cfg["SPLIT"]
    k_per_split = triton.cdiv(triton.cdiv(KK, split), BK) * BK
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), B * split)
    common = dict(KS=KS, S=S, MODE=mode, ELU_IN=elu, BM=BM, BN=BN, BK=BK, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    res = residual if residual is not None else w
    dl = dual if dual is not None else w
    if split == 1:
        _conv1d_kernel[grid](x, w, bias if bias is not None else w, res, out, dl,
                             Cin_rows, T_x, pad_left, M, N, KK, stride, k_per_split,
                             HAS_RES=residual is not None, ELU_OUT=elu_out, DUAL_ELU=dual is not None, HAS_BIAS=bias is not None,
                             SPLIT=1, PHASES=phases, **common)
        return out
    part = torch.empty((split, B, M, N), device=x.device, dtype=torch.float32)
    _conv1d_kernel[grid](x, w, w, w, part, w,
                         Cin_rows, T_x, pad_left, M, N, KK, stride, k_per_split,
                         HAS_RES=False, ELU_OUT=False, DUAL_ELU=False, HAS_BIAS=False, SPLIT=split, PHASES=1, **common)
    block = 1024
    _splitk_reduce_kernel[(triton.cdiv(M * N, block), B)](part, bias if bias is not None else w, res, out, dl, B, M, N,
                                                          HAS_BIAS=bias is not None, HAS_RES=residual is not None, ELU_OUT=elu_out,
                                                          DUAL_ELU=dual is not None, SPLIT=split, PHASES=phases, BLOCK=block, num_warps=4)
    return out


def _kb(K: int, kb_pad: bool) -> int:
    """Rows of the im2col matrix: K, or K padded (multiple of 8) with a ones row at index K carrying the bias."""
    return K if (not kb_pad and K % 8 == 0) else ((K + 1 + 7) // 8) * 8


def _launch_cublas(x: torch.Tensor, wT: torch.Tensor, bias, out: torch.Tensor, *, KS: int, stride: int, pad_left: int,
                   K: int, M: int, N: int, phases: int, variant: str, kb_pad: bool, elu: bool = False,
                   dual: torch.Tensor | None = None):
    """Explicit im2col (Triton, virtual padding) + cuBLAS fp32 GEMM (TF32 off, deterministic) + epilogue.

    wT is the prepacked (KB, M) weight (row K = bias when kb_pad).  Variants (measured per layer shape):
      E: cols (B,KB,N) -> mm(cols^T view, wT, out=out^T view)        bias via the ones row, no epilogue
      D: cols (B,KB,N) -> matmul(cols^T view, wT) -> (B,N,M) -> transpose epilogue (bias, ConvTranspose phases)
    """
    B, Cin, T_x = x.shape
    KB = wT.shape[0]
    out_t = variant == "E"
    cols = torch.empty((B, KB, N), device=x.device, dtype=torch.float32)
    BK, BN = 64, 64
    _im2col_kernel[(triton.cdiv(KB, BK), triton.cdiv(N, BN), B)](x, cols, Cin, T_x, pad_left, K, KB, N, stride,
                                                                 KS=KS, ELU_IN=elu, BK=BK, BN=BN, num_warps=4)
    if out_t:
        for b in range(B):
            torch.mm(cols[b].t(), wT, out=out[b].t())
        return out
    r = torch.matmul(cols.transpose(1, 2), wT)                          # (B, N, M)
    epi_bias = None if (kb_pad or bias is None) else bias
    BM = 64
    _epilogue_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), B)](r, epi_bias if epi_bias is not None else r, out,
                                                                  dual if dual is not None else out, B, M, N,
                                                                  HAS_BIAS=epi_bias is not None,
                                                                  PHASES=phases, SPLIT=1, DUAL_ELU=dual is not None, BM=BM, BN=BN, num_warps=4)
    return out


def _pack_wT(w2d: torch.Tensor, bias, kb_pad: bool) -> torch.Tensor:
    """(M, K) row-major weight -> (KB, M) transposed copy; when kb_pad, row K holds the bias (0 if none)."""
    M, K = w2d.shape
    KB = _kb(K, kb_pad)
    wb = torch.zeros((M, KB), device=w2d.device, dtype=torch.float32)
    wb[:, :K] = w2d
    if kb_pad and bias is not None:
        wb[:, K] = bias
    return wb.t().contiguous()


class Conv1dPlan:
    """Prepacked state for one nn.Conv1d (groups=1, dilation=1, padding=0): weight view + per-shape configs."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, stride: int):
        self.Cout, self.Cin, self.KS = weight.shape
        self.stride = int(stride)
        self.w = weight.detach().contiguous()
        self.bias = bias.detach().contiguous() if bias is not None else None
        # (ci, k) flat weight for the single-output-channel streaming kernel (the decoder's last conv)
        self._tail_w = self.w.view(-1) if self.Cout == 1 and self.stride == 1 else None
        self.polyphase = self.stride > 1 and self.KS == 2 * self.stride
        self.SP = triton.next_power_of_2(self.stride) if self.polyphase else 1
        self._cfg: dict[tuple, tuple[int, dict]] = {}
        self._wT: dict[bool, torch.Tensor] = {}
        for ent in _entries("conv1" if self.polyphase else "conv0", self.Cout, self.Cin * self.KS, self.stride):
            if ent.get("backend") == "cublas":        # prepack once for the orientations the tuned entries use
                self.cublas_pack(bool(ent.get("kb_pad", True)))

    def cublas_pack(self, kb_pad: bool) -> torch.Tensor:
        wT = self._wT.get(kb_pad)
        if wT is None:
            wT = self._wT[kb_pad] = _pack_wT(self.w.view(self.Cout, self.Cin * self.KS), self.bias, kb_pad)
        return wT

    def _plan(self, B: int, N: int) -> tuple[int, dict]:
        key = (B, N)
        hit = self._cfg.get(key)
        if hit is None:
            KK = self.Cin * self.KS
            mode = MODE_S2C if self.polyphase else MODE_DIRECT
            ent = _lookup(f"conv{mode}", B, self.Cout, N, KK, self.stride)
            cfg = dict(ent) if ent is not None else _default_config(B, self.Cout, N, KK, _n_sm(self.w.device))
            hit = (mode, _drop_useless_split(cfg, B, self.Cout, N, self.w.device))
            self._cfg[key] = hit
        return hit

    def out_len(self, T: int, pad_left: int, pad_right: int) -> int:
        return (T + pad_left + pad_right - self.KS) // self.stride + 1

    def elu_fold_free(self, B: int, T: int, pad_left: int, pad_right: int) -> bool:
        """True when elu=True costs no extra launch here (ELU inside the s2c copy or the im2col); False on the direct GEMM path."""
        mode, cfg = self._plan(B, self.out_len(T, pad_left, pad_right))
        return cfg.get("backend") == "cublas" or mode == MODE_S2C

    def dual_ok(self, B: int, T: int, pad_left: int, pad_right: int) -> bool:
        """True when this shape's backend can emit the second ELU(out) tensor from its epilogue."""
        mode, cfg = self._plan(B, self.out_len(T, pad_left, pad_right))
        return cfg.get("backend") != "cublas" or cfg["variant"] not in ("E", "C")

    def __call__(self, x: torch.Tensor, pad_left: int = 0, pad_right: int = 0, mode: int | None = None, cfg: dict | None = None,
                 elu: bool = False, elu_in_load: bool = False, residual: torch.Tensor | None = None, elu_out: bool = False,
                 dual_elu: bool = False):
        """Conv1d of [pad_left zeros | x | pad_right zeros]; with elu=True the input is ELU(x): applied on load where the
        input is re-laid out once (space-to-channel copy, im2col) or, when elu_in_load, in the GEMM's B loads (short-T
        chain convs), else by the stock elementwise kernel. `residual` (B, Cout, N) is added in the epilogue."""
        B, Cin, T_x = x.shape
        assert Cin == self.Cin
        T_in = T_x + pad_left + pad_right
        N = (T_in - self.KS) // self.stride + 1
        out = torch.empty((B, self.Cout, N), device=x.device, dtype=torch.float32)
        dual = torch.empty_like(out) if dual_elu else None                    # ELU(out), written by the same epilogue
        if N <= 0:
            return (out, dual) if dual_elu else out
        if (self._tail_w is not None and mode is None and pad_right == 0 and pad_left == self.KS - 1
                and residual is None and not elu_out and not dual_elu and B == 1):
            _tail_conv(x, self._tail_w, self.bias, out, self.KS, elu)
            return out
        if mode is None:
            mode, cfg = self._plan(B, N)
        s = self.stride
        if cfg.get("backend") == "cublas":
            kb_pad = bool(cfg.get("kb_pad", True)) or cfg["variant"] == "E"
            if cfg["variant"] == "E":
                dual = None                                                   # no epilogue kernel on this variant
            _launch_cublas(x, self.cublas_pack(kb_pad), self.bias, out, KS=self.KS, stride=s, pad_left=pad_left,
                           K=Cin * self.KS, M=self.Cout, N=N, phases=1, variant=cfg["variant"], kb_pad=kb_pad, elu=elu, dual=dual)
            out = out if residual is None else torch.add(residual, out)       # stock order: residual + hidden
            out = torch.nn.functional.elu(out) if elu_out else out
            return (out, dual) if dual_elu else out
        if mode == MODE_DIRECT:
            if elu and not elu_in_load:
                x = torch.nn.functional.elu(x)
            _launch_gemm(x, self.w, self.bias, out, Cin_rows=Cin, T_x=T_x, pad_left=pad_left, M=self.Cout, N=N,
                         KK=Cin * self.KS, stride=s, KS=self.KS, S=1, mode=MODE_DIRECT, phases=1, cfg=cfg, B=B,
                         elu=elu and elu_in_load, residual=residual, elu_out=elu_out, dual=dual)
        else:
            if residual is not None:
                raise ValueError("residual fusion is only supported on the direct GEMM path")
            T_new = N + 1
            xs = torch.empty((B, Cin * s, T_new), device=x.device, dtype=torch.float32)
            BT = 128
            _s2c_pad_kernel[(B * Cin, triton.cdiv(T_new, BT))](x, xs, T_x, pad_left, T_new, S=s, SP=self.SP, ELU_IN=elu, BT=BT, num_warps=4)
            _launch_gemm(xs, self.w, self.bias, out, Cin_rows=Cin * s, T_x=T_new, pad_left=0, M=self.Cout, N=N,
                         KK=Cin * self.KS, stride=1, KS=self.KS, S=s, mode=MODE_S2C, phases=1, cfg=cfg, B=B, elu_out=elu_out, dual=dual)
        return (out, dual) if dual_elu else out


class ConvT1dPlan:
    """nn.ConvTranspose1d (groups=1, k = 2*stride, no padding) followed by the causal right trim of k - stride samples.

    y[b, co, t*s + j] = bias[co] + sum_ci x[b, ci, t-1] * w[ci, co, s + j] + x[b, ci, t] * w[ci, co, j]
    Prepacked W'[co*s + j, ci*2 + kt] = w[ci, co, j + (1 - kt) * s]  (kt = 0 pairs with x[t-1], kt = 1 with x[t]).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, stride: int):
        self.Cin, self.Cout, self.KS = weight.shape
        self.stride = int(stride)
        assert self.KS == 2 * self.stride, "ConvT1dPlan needs kernel_size == 2 * stride"
        s = self.stride
        w = weight.detach()
        self.w = (w.permute(1, 0, 2).reshape(self.Cout, self.Cin, 2, s).permute(0, 3, 1, 2).flip(3)
                  .reshape(self.Cout * s, self.Cin * 2).contiguous())
        self.bias = bias.detach().repeat_interleave(s).contiguous() if bias is not None else None
        self._cfg: dict[tuple, dict] = {}
        self._wT: dict[bool, torch.Tensor] = {}
        for ent in _entries("convt", self.Cout * s, self.Cin * 2, s):
            if ent.get("backend") == "cublas":
                self.cublas_pack(bool(ent.get("kb_pad", False)))

    def cublas_pack(self, kb_pad: bool) -> torch.Tensor:
        wT = self._wT.get(kb_pad)
        if wT is None:
            wT = self._wT[kb_pad] = _pack_wT(self.w, self.bias, kb_pad)
        return wT

    def _plan(self, B: int, N: int) -> dict:
        key = (B, N)
        cfg = self._cfg.get(key)
        if cfg is None:
            M, KK = self.Cout * self.stride, self.Cin * 2
            ent = _lookup("convt", B, M, N, KK, self.stride)
            cfg = dict(ent) if ent is not None else _default_config(B, M, N, KK, _n_sm(self.w.device))
            self._cfg[key] = cfg = _drop_useless_split(cfg, B, M, N, self.w.device)
        return cfg

    def elu_fold_free(self, B: int, T_in: int) -> bool:
        return self._plan(B, T_in).get("backend") == "cublas"

    def dual_ok(self, B: int, T_in: int) -> bool:
        cfg = self._plan(B, T_in)
        return cfg.get("backend") != "cublas" or cfg["variant"] not in ("E", "C")

    def __call__(self, x: torch.Tensor, cfg: dict | None = None, elu: bool = False, elu_out: bool = False, dual_elu: bool = False):
        """Trimmed ConvTranspose1d of x (of ELU(x) when elu=True, applied on load); elu_out applies ELU to the output in the
        epilogue; dual_elu returns (out, ELU(out)) from the same epilogue (None when the backend has no epilogue kernel)."""
        B, Cin, T_in = x.shape
        assert Cin == self.Cin
        s = self.stride
        out = torch.empty((B, self.Cout, T_in * s), device=x.device, dtype=torch.float32)
        dual = torch.empty_like(out) if dual_elu else None
        if T_in == 0:
            return (out, dual) if dual_elu else out
        if cfg is None:
            cfg = self._plan(B, T_in)
        if cfg.get("backend") == "cublas":
            kb_pad = bool(cfg.get("kb_pad", False))
            if cfg["variant"] in ("E", "C"):
                dual = None
            _launch_cublas(x, self.cublas_pack(kb_pad), self.bias, out, KS=2, stride=1, pad_left=1, K=Cin * 2,
                           M=self.Cout * s, N=T_in, phases=s, variant=cfg["variant"], kb_pad=kb_pad, elu=elu, dual=dual)
            out = torch.nn.functional.elu(out) if elu_out else out
            return (out, dual) if dual_elu else out
        if elu:
            x = torch.nn.functional.elu(x)
        _launch_gemm(x, self.w, self.bias, out, Cin_rows=Cin, T_x=T_in, pad_left=1, M=self.Cout * s, N=T_in,
                     KK=Cin * 2, stride=1, KS=2, S=1, mode=MODE_DIRECT, phases=s, cfg=cfg, B=B, elu_out=elu_out, dual=dual)
        return (out, dual) if dual_elu else out


def conv1d_fp32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int,
                pad_left: int = 0, pad_right: int = 0) -> torch.Tensor:
    """One-shot API (builds a plan per call; prefer Conv1dPlan for repeated use)."""
    return Conv1dPlan(weight, bias, stride)(x.contiguous(), pad_left, pad_right)
