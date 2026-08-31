"""Implicit-GEMM 1-D convolution kernels for the SEANet encoder/decoder (channels-last activations [T, C]).

conv1d:  y[t, co] = b[co] (+ res[t, co]) + sum_j sum_ci act(x[t*stride + j*dil - pad, ci]) * w[j, ci, co]
convT1d: y[t*S + s, co] = b[co] + sum_{j in 0,1} sum_ci act(x[t - j, ci]) * w[s + j*S, ci, co]     (causal trim)
Weights are pre-arranged as [k, C_in, C_out] bf16. act = ELU (optional). Inputs may be bf16 or fp32; outputs
are bf16 or fp32 partial sums accumulated with atomics (split-K over taps / input channels).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _elu(a):
    return tl.where(a > 0, a, tl.exp(tl.minimum(a, 0.0)) - 1.0)


@triton.jit
def _wtile(w_ptr, sc_ptr, koff, co, C_out: tl.constexpr, INT8: tl.constexpr):
    w = tl.load(w_ptr + koff[:, None] * C_out + co[None, :])
    if INT8:
        return (w.to(tl.float32) * tl.load(sc_ptr + co)[None, :]).to(tl.bfloat16)
    else:
        return w


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, sc_ptr, b_ptr, res_ptr, y_ptr, T_in, T_out, stride, dil, pad,
                   C_in: tl.constexpr, C_out: tl.constexpr, K: tl.constexpr, KSPLIT: tl.constexpr,
                   BT: tl.constexpr, BCO: tl.constexpr, BCI: tl.constexpr,
                   ELU_IN: tl.constexpr, HAS_RES: tl.constexpr, CLAMP: tl.constexpr, ATOMIC: tl.constexpr, HAS_BIAS: tl.constexpr,
                   INT8: tl.constexpr, ELU_OUT: tl.constexpr, DUAL: tl.constexpr, y2_ptr,
                   RES_CONV0: tl.constexpr, audio_ptr, w0_ptr, b0_ptr, K0: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)
    t = pid_t * BT + tl.arange(0, BT)
    tmask = t < T_out
    co = pid_c * BCO + tl.arange(0, BCO)
    acc = tl.zeros([BT, BCO], dtype=tl.float32)
    NCI: tl.constexpr = C_in // BCI
    NSTEP: tl.constexpr = (K * NCI) // KSPLIT
    for step in range(pid_k * NSTEP, (pid_k + 1) * NSTEP):
        j = step // NCI
        ci = (step % NCI) * BCI + tl.arange(0, BCI)
        r = t * stride + j * dil - pad
        if CLAMP:
            r = tl.minimum(tl.maximum(r, 0), T_in - 1)
            rmask = tmask
        else:
            rmask = (r >= 0) & (r < T_in) & tmask
        a = tl.load(x_ptr + r[:, None] * C_in + ci[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        if ELU_IN:
            a = _elu(a)
        w = _wtile(w_ptr, sc_ptr, j * C_in + ci, co, C_out, INT8)
        acc += tl.dot(a.to(tl.bfloat16), w)
    if HAS_BIAS:
        if pid_k == 0:
            acc += tl.load(b_ptr + co)[None, :]
    if HAS_RES:
        if pid_k == 0:
            acc += tl.load(res_ptr + t[:, None] * C_out + co[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    if RES_CONV0:
        acc += _conv0_rows(audio_ptr, w0_ptr, b0_ptr, t, tmask, K0, C_out)
    if ATOMIC:
        tl.atomic_add(y_ptr + t[:, None] * C_out + co[None, :], acc, mask=tmask[:, None], sem="relaxed")
    else:
        if DUAL:
            tl.store(y_ptr + t[:, None] * C_out + co[None, :], acc.to(y_ptr.dtype.element_ty), mask=tmask[:, None])
            tl.store(y2_ptr + t[:, None] * C_out + co[None, :], _elu(acc).to(y2_ptr.dtype.element_ty), mask=tmask[:, None])
        elif ELU_OUT:
            tl.store(y_ptr + t[:, None] * C_out + co[None, :], _elu(acc).to(y_ptr.dtype.element_ty), mask=tmask[:, None])
        else:
            tl.store(y_ptr + t[:, None] * C_out + co[None, :], acc.to(y_ptr.dtype.element_ty), mask=tmask[:, None])


@triton.jit
def _finalize_kernel(y32_ptr, raw_ptr, act_ptr, n, WRITE_RAW: tl.constexpr, WRITE_ACT: tl.constexpr, BLOCK: tl.constexpr):
    """fp32 split-K accumulator -> raw copy and/or ELU copy (any dtype)"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(y32_ptr + i, mask=m, other=0.0)
    if WRITE_RAW:
        tl.store(raw_ptr + i, v.to(raw_ptr.dtype.element_ty), mask=m)
    if WRITE_ACT:
        tl.store(act_ptr + i, _elu(v).to(act_ptr.dtype.element_ty), mask=m)


@triton.jit
def _convT1d_kernel(x_ptr, w_ptr, sc_ptr, b_ptr, y_ptr, T_in, T_out, S,
                    C_in: tl.constexpr, C_out: tl.constexpr, KSPLIT: tl.constexpr,
                    BT: tl.constexpr, BCO: tl.constexpr, BCI: tl.constexpr, ELU_IN: tl.constexpr, ATOMIC: tl.constexpr,
                    INT8: tl.constexpr, ELU_OUT: tl.constexpr, DUAL: tl.constexpr, y2_ptr):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2) % (C_out // BCO)
    pid_k = tl.program_id(2) // (C_out // BCO)
    t = pid_t * BT + tl.arange(0, BT)
    tmask = t < T_in
    co = pid_c * BCO + tl.arange(0, BCO)
    acc = tl.zeros([BT, BCO], dtype=tl.float32)
    CS: tl.constexpr = C_in // KSPLIT
    for j in tl.static_range(2):
        r = t - j
        rmask = (r >= 0) & tmask
        for ci0 in range(pid_k * CS, (pid_k + 1) * CS, BCI):
            ci = ci0 + tl.arange(0, BCI)
            a = tl.load(x_ptr + r[:, None] * C_in + ci[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            if ELU_IN:
                a = _elu(a)
            w = _wtile(w_ptr, sc_ptr, (pid_s + j * S) * C_in + ci, co, C_out, INT8)
            acc += tl.dot(a.to(tl.bfloat16), w)
    if pid_k == 0:
        acc += tl.load(b_ptr + co)[None, :]
    p = t * S + pid_s
    pmask = p < T_out
    if ATOMIC:
        tl.atomic_add(y_ptr + p[:, None] * C_out + co[None, :], acc, mask=pmask[:, None], sem="relaxed")
    else:
        if DUAL:
            tl.store(y_ptr + p[:, None] * C_out + co[None, :], acc.to(y_ptr.dtype.element_ty), mask=pmask[:, None])
            tl.store(y2_ptr + p[:, None] * C_out + co[None, :], _elu(acc).to(y2_ptr.dtype.element_ty), mask=pmask[:, None])
        elif ELU_OUT:
            tl.store(y_ptr + p[:, None] * C_out + co[None, :], _elu(acc).to(y_ptr.dtype.element_ty), mask=pmask[:, None])
        else:
            tl.store(y_ptr + p[:, None] * C_out + co[None, :], acc.to(y_ptr.dtype.element_ty), mask=pmask[:, None])


@triton.jit
def _conv0_rows(x_ptr, w0_ptr, b0_ptr, r, rmask, K0: tl.constexpr, C: tl.constexpr):
    """conv0 (1 -> C channels, K0 taps, causal) evaluated at rows r -> [BT, C] fp32 (zero for masked rows)"""
    co = tl.arange(0, C)
    acc = tl.zeros([r.shape[0], C], dtype=tl.float32) + tl.load(b0_ptr + co)[None, :]
    for j in tl.static_range(K0):
        q = r + j - (K0 - 1)
        xv = tl.load(x_ptr + q, mask=(q >= 0) & rmask, other=0.0)
        acc += xv[:, None] * tl.load(w0_ptr + co * K0 + j)[None, :]
    return tl.where(rmask[:, None], acc, 0.0)


@triton.jit
def _conv_first_res_kernel(x_ptr, w0_ptr, b0_ptr, w1_ptr, b1_ptr, y_ptr, T,
                           K0: tl.constexpr, C: tl.constexpr, K1: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr):
    """audio [T] -> h = ELU( conv_k1( ELU(conv0(audio)) ) )  [T, CH]   (first residual-block conv fused with conv0)"""
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tmask = t < T
    ch = tl.arange(0, CH)
    acc = tl.zeros([BT, CH], dtype=tl.float32)
    for j in tl.static_range(K1):
        r = t + j - (K1 - 1)
        rmask = (r >= 0) & tmask
        a = _elu(_conv0_rows(x_ptr, w0_ptr, b0_ptr, r, rmask, K0, C))
        w = tl.load(w1_ptr + (j * C + tl.arange(0, C))[:, None] * CH + ch[None, :])     # [C, CH] bf16
        acc += tl.dot(a.to(tl.bfloat16), w)
    acc += tl.load(b1_ptr + ch)[None, :]
    tl.store(y_ptr + t[:, None] * CH + ch[None, :], _elu(acc).to(y_ptr.dtype.element_ty), mask=tmask[:, None])


@triton.jit
def _conv_first_kernel(x_ptr, w_ptr, b_ptr, y_ptr, y2_ptr, T, K: tl.constexpr, C_out: tl.constexpr, BT: tl.constexpr, WRITE_RAW: tl.constexpr):
    """audio [T] fp32 -> y [T, C_out] bf16 ; causal k-tap conv with a single input channel."""
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tmask = t < T
    co = tl.arange(0, C_out)
    acc = tl.zeros([BT, C_out], dtype=tl.float32) + tl.load(b_ptr + co)[None, :]
    for j in tl.static_range(K):
        r = t + j - (K - 1)
        xv = tl.load(x_ptr + r, mask=(r >= 0) & tmask, other=0.0)
        wv = tl.load(w_ptr + co * K + j)
        acc += xv[:, None] * wv[None, :]
    if WRITE_RAW:
        tl.store(y_ptr + t[:, None] * C_out + co[None, :], acc.to(y_ptr.dtype.element_ty), mask=tmask[:, None])
    tl.store(y2_ptr + t[:, None] * C_out + co[None, :], _elu(acc).to(y2_ptr.dtype.element_ty), mask=tmask[:, None])


@triton.jit
def _conv_last_kernel(x_ptr, w_ptr, b_ptr, y_ptr, T, K: tl.constexpr, C_in: tl.constexpr, BT: tl.constexpr):
    """x [T, C_in] (ELU applied on load) -> audio [T] fp32 ; causal k-tap conv to a single output channel."""
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tmask = t < T
    ci = tl.arange(0, C_in)
    acc = tl.zeros([BT], dtype=tl.float32) + tl.load(b_ptr)
    for j in tl.static_range(K):
        r = t + j - (K - 1)
        a = tl.load(x_ptr + r[:, None] * C_in + ci[None, :], mask=((r >= 0) & tmask)[:, None], other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + ci * K + j)
        acc += tl.sum(a * wv[None, :], axis=1)
    tl.store(y_ptr + t, acc, mask=tmask)


# ------------------------------------------------------------------------------------------ python side

def _quant(w_kco: torch.Tensor, wdtype: str):
    """[k, C_in, C_out] fp32 -> (bf16, dummy) or (int8, per-C_out scale)"""
    if wdtype == "int8":
        scale = w_kco.abs().amax(dim=(0, 1)).clamp(min=1e-12) / 127.0
        q = torch.round(w_kco / scale).clamp(-127, 127).to(torch.int8).contiguous()
        return q, scale.float().contiguous()
    return w_kco.bfloat16().contiguous(), w_kco.new_zeros(1)


def prep_conv_weight(w: torch.Tensor, wdtype="bf16"):
    """nn.Conv1d weight [C_out, C_in, k] -> [k, C_in, C_out]"""
    return _quant(w.detach().float().permute(2, 1, 0).contiguous(), wdtype)


def prep_convT_weight(w: torch.Tensor, wdtype="bf16"):
    """nn.ConvTranspose1d weight [C_in, C_out, k] -> [k, C_in, C_out]"""
    return _quant(w.detach().float().permute(2, 0, 1).contiguous(), wdtype)


def conv_out_len(T_in, k, stride, dil=1):
    import math
    return math.ceil(T_in / stride) if stride > 1 else T_in


def _time_kernel(fn, n_in_graph=20, reps=5):
    """GPU time per call (ms) measured by replaying a CUDA graph of n_in_graph back-to-back calls."""
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_in_graph):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en) / n_in_graph)
    ts.sort()
    return ts[len(ts) // 2]


import json, os
_TUNE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tuned_conv.json")
try:
    _TUNE_CACHE = {}
    for k, v in json.load(open(_TUNE_FILE)).items():
        kk = json.loads(k)
        if len(kk) < 10:          # legacy keys (pre-v2 timing/epilogues)
            continue
        _TUNE_CACHE[tuple(kk)] = v
except Exception:
    _TUNE_CACHE = {}


def _save_tune_cache():
    try:
        json.dump({json.dumps(list(k)): v for k, v in _TUNE_CACHE.items()}, open(_TUNE_FILE, "w"), indent=1)
    except Exception:
        pass


def _tbucket(T):
    return int(triton.next_power_of_2(max(T, 16)))


def _bt_choices(T_out):
    base = 16 if T_out <= 32 else (32 if T_out <= 256 else (64 if T_out <= 4096 else 128))
    return sorted({base, min(base * 2, 128)})


class Conv1d:
    def __init__(self, w, b, stride=1, dil=1, elu_in=False, res=False, clamp=False, bt=None, bco=None, bci=None,
                 ksplit=1, num_warps=4, num_stages=3, name="", wdtype="bf16"):
        self.C_out, self.C_in, self.K = w.shape
        self.w, self.sc = prep_conv_weight(w, wdtype)
        self.int8 = wdtype == "int8"
        self.b = b.detach().float().contiguous() if b is not None else None
        self.stride, self.dil, self.elu_in, self.res, self.clamp = stride, dil, elu_in, res, clamp
        self.k_eff = (self.K - 1) * dil + 1
        self.pad = self.k_eff - stride
        self.name = name
        self.cfg = dict(bt=bt or (64 if self.C_out <= 128 else (32 if self.C_out <= 256 else 16)),
                        bco=bco or min(self.C_out, 64), bci=bci or min(self.C_in, 64), ksplit=ksplit,
                        num_warps=num_warps, num_stages=num_stages)

    def candidates(self, T_out, allow_split):
        cands = []
        flops = T_out * self.K * self.C_in * self.C_out * 2
        heavy = flops >= 100e6 and T_out >= 128            # compute-bound layers: search wider tiles
        bcos = [c for c in ((64, 128, 256) if heavy else (64, 128)) if c <= self.C_out] or [self.C_out]
        bcis = [c for c in ((32, 64, 128) if heavy else (64, 128, 256)) if c <= self.C_in] or [self.C_in]
        bts = ([64, 128] if heavy else _bt_choices(T_out))
        for bt in bts:
            for bco in bcos:
                if bt * bco > (128 * 128 if heavy else 128 * 64):
                    continue
                for bci in bcis:
                    nsteps = self.K * (self.C_in // bci)
                    splits = [1] + ([k for k in (4,) if allow_split and nsteps % k == 0] if T_out <= 64 else [])
                    for ks in splits:
                        for nw in ((4, 8) if heavy else (8 if bt * bco >= 64 * 128 else 4,)):
                            for nst in ((2, 3, 4) if heavy else (3,)):
                                cands.append(dict(bt=bt, bco=bco, bci=bci, ksplit=ks, num_warps=nw, num_stages=nst))
        return cands

    def tune(self, x, y, res=None, **kw):
        yy = y if y is not None else kw.get("y_act")
        key = (self.name, self.C_in, self.C_out, self.K, self.stride, _tbucket(yy.shape[0]), str(yy.dtype), str(x.dtype), "i8" if self.int8 else "bf16", "v3")
        if key in _TUNE_CACHE:
            self.cfg = _TUNE_CACHE[key]
            return
        allow_split = kw.get("y32") is not None
        best, best_t = self.cfg, float("inf")
        for c in self.candidates(yy.shape[0], allow_split):
            self.cfg = c
            try:
                t = _time_kernel(lambda: self(x, y, res, **kw))
            except Exception:
                continue
            if t < best_t:
                best, best_t = c, t
        self.cfg = best
        _TUNE_CACHE[key] = best
        _save_tune_cache()

    def __call__(self, x, y, res=None, y_act=None, elu_out=False, y32=None, res_conv0=None):
        """y: raw output (or None if only the activated output is wanted); y_act: ELU(output) buffer.
        If y32 is given (fp32 scratch) split-K atomics accumulate there and a finalize kernel writes y / y_act.
        res_conv0=(audio, w0, b0): residual recomputed as conv0(audio) (first encoder block)."""
        c = self.cfg
        T_in, T_out = x.shape[0], (y if y is not None else y_act).shape[0]
        atomic = c["ksplit"] > 1
        if atomic:
            assert y32 is not None
            y32.zero_()
            target = y32
        else:
            target = y if y is not None else y_act
        dual = (not atomic) and y is not None and y_act is not None
        eo = (not atomic) and (elu_out or (y is None and y_act is not None))
        grid = (triton.cdiv(T_out, c["bt"]), self.C_out // c["bco"], c["ksplit"])
        _conv1d_kernel[grid](x, self.w, self.sc, self.b if self.b is not None else self.sc, res if res is not None else x, target,
                             T_in, T_out, self.stride, self.dil, self.pad,
                             C_in=self.C_in, C_out=self.C_out, K=self.K, KSPLIT=c["ksplit"], BT=c["bt"], BCO=c["bco"], BCI=c["bci"],
                             ELU_IN=self.elu_in, HAS_RES=res is not None, CLAMP=self.clamp, ATOMIC=atomic, HAS_BIAS=self.b is not None,
                             INT8=self.int8, ELU_OUT=eo, DUAL=dual, y2_ptr=y_act if dual else target,
                             RES_CONV0=res_conv0 is not None, audio_ptr=res_conv0[0] if res_conv0 else x,
                             w0_ptr=res_conv0[1] if res_conv0 else self.sc, b0_ptr=res_conv0[2] if res_conv0 else self.sc,
                             K0=res_conv0[1].shape[-1] if res_conv0 else 1,
                             num_warps=c["num_warps"], num_stages=c["num_stages"])
        if atomic:
            finalize(y32, y, y_act, elu_out)
        return y if y is not None else y_act


class ConvT1d:
    def __init__(self, w, b, stride, elu_in=True, bt=None, bco=None, bci=None, ksplit=1, num_warps=4, num_stages=3, name="", wdtype="bf16"):
        self.C_in, self.C_out, self.K = w.shape
        assert self.K == 2 * stride
        self.S = stride
        self.w, self.sc = prep_convT_weight(w, wdtype)
        self.int8 = wdtype == "int8"
        self.b = b.detach().float().contiguous()
        self.elu_in = elu_in
        self.name = name
        self.cfg = dict(bt=bt or (64 if self.C_out <= 128 else (32 if self.C_out <= 256 else 16)),
                        bco=bco or min(self.C_out, 64), bci=bci or min(self.C_in // ksplit, 64), ksplit=ksplit,
                        num_warps=num_warps, num_stages=num_stages)

    def candidates(self, T_in, allow_split):
        cands = []
        flops = T_in * self.S * 2 * self.C_in * self.C_out * 2
        heavy = flops >= 100e6 and T_in >= 64
        bcos = [c for c in ((64, 128, 256) if heavy else (64, 128)) if c <= self.C_out] or [self.C_out]
        bts = ([32, 64, 128] if heavy else _bt_choices(T_in))
        for bt in bts:
            for bco in bcos:
                if bt * bco > (128 * 128 if heavy else 128 * 64):
                    continue
                splits = [1] + ([k for k in (4,) if allow_split and self.C_in % (k * 64) == 0] if T_in <= 64 else [])
                for ks in splits:
                    for bci in [c for c in ((32, 64, 128) if heavy else (64, 128, 256)) if c <= self.C_in // ks]:
                        for nw in ((4, 8) if heavy else (8 if bt * bco >= 64 * 128 else 4,)):
                            for nst in ((2, 3, 4) if heavy else (3,)):
                                cands.append(dict(bt=bt, bco=bco, bci=bci, ksplit=ks, num_warps=nw, num_stages=nst))
        return cands

    def tune(self, x, y, **kw):
        yy = y if y is not None else kw.get("y_act")
        key = (self.name, self.C_in, self.C_out, self.K, self.S, _tbucket(x.shape[0]), str(yy.dtype), str(x.dtype), "i8" if self.int8 else "bf16", "v3")
        if key in _TUNE_CACHE:
            self.cfg = _TUNE_CACHE[key]
            return
        best, best_t = self.cfg, float("inf")
        for c in self.candidates(x.shape[0], kw.get("y32") is not None):
            self.cfg = c
            try:
                t = _time_kernel(lambda: self(x, y, **kw))
            except Exception:
                continue
            if t < best_t:
                best, best_t = c, t
        self.cfg = best
        _TUNE_CACHE[key] = best
        _save_tune_cache()

    def __call__(self, x, y, y_act=None, elu_out=False, y32=None):
        c = self.cfg
        T_in, T_out = x.shape[0], (y if y is not None else y_act).shape[0]
        atomic = c["ksplit"] > 1
        if atomic:
            assert y32 is not None
            y32.zero_()
            target = y32
        else:
            target = y if y is not None else y_act
        dual = (not atomic) and y is not None and y_act is not None
        eo = (not atomic) and (elu_out or (y is None and y_act is not None))
        grid = (triton.cdiv(T_in, c["bt"]), self.S, (self.C_out // c["bco"]) * c["ksplit"])
        _convT1d_kernel[grid](x, self.w, self.sc, self.b, target, T_in, T_out, self.S, C_in=self.C_in, C_out=self.C_out, KSPLIT=c["ksplit"],
                              BT=c["bt"], BCO=c["bco"], BCI=c["bci"], ELU_IN=self.elu_in, ATOMIC=atomic, INT8=self.int8,
                              ELU_OUT=eo, DUAL=dual, y2_ptr=y_act if dual else target,
                              num_warps=c["num_warps"], num_stages=c["num_stages"])
        if atomic:
            finalize(y32, y, y_act, elu_out)
        return y if y is not None else y_act


def finalize(y32, y, y_act, elu_out=False):
    n = y32.numel()
    raw = y if (y is not None and not elu_out) else None
    act = y_act if y_act is not None else (y if elu_out else None)
    _finalize_kernel[(triton.cdiv(n, 1024),)](y32, raw if raw is not None else y32, act if act is not None else y32, n,
                                              WRITE_RAW=raw is not None, WRITE_ACT=act is not None, BLOCK=1024, num_warps=4)


def conv_first_res(x, w0, b0, conv1: "Conv1d", y, bt=32):
    """fused conv0 + first residual-block conv (ELU in/out); y: [T, C/2]"""
    C_out, _, K0 = w0.shape
    _conv_first_res_kernel[(triton.cdiv(x.shape[0], bt),)](x, w0.contiguous().float(), b0.float(), conv1.w, conv1.b, y, x.shape[0],
                                                           K0=K0, C=C_out, K1=conv1.K, CH=conv1.C_out, BT=bt, num_warps=4)
    return y


def conv_first(x, w, b, y, y_act, bt=128):
    C_out, _, K = w.shape
    _conv_first_kernel[(triton.cdiv(x.shape[0], bt),)](x, w.contiguous().float(), b.float(), y if y is not None else y_act, y_act, x.shape[0],
                                                       K=K, C_out=C_out, BT=bt, WRITE_RAW=y is not None, num_warps=4)
    return y


def conv_last(x, w, b, y, bt=128):
    _, C_in, K = w.shape
    _conv_last_kernel[(triton.cdiv(x.shape[0], bt),)](x, w.contiguous().float(), b.float(), y, x.shape[0], K=K, C_in=C_in, BT=bt, num_warps=4)
    return y
