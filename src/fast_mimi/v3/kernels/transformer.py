"""Fused Triton kernels for the Mimi transformer (pre-LN, RoPE, causal attention, layer-scale, GELU MLP).

Residual stream x is fp32 [M, D]; weights bf16 (pre-transposed to [K, N]); GEMMs on bf16 tensor cores with
fp32 accumulation. Four launches per layer:
  1. ln_qkv:  LN1 -> q,k,v = h @ Wqkv ; RoPE on q,k           (grid: 24 head-tiles x M-tiles)
  2. attn_o:  causal attention (all heads) @ Wo ; x += ls1*.  (grid: N-tiles x M-tiles)
  3. ln_fc1:  LN2 -> gelu(h @ W1)                              (grid: N-tiles x M-tiles)
  4. fc2:     x += ls2 * (g @ W2)   (split-K, fp32 atomic adds)
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _wload(w_ptr, sc_ptr, koff, noff, NOUT: tl.constexpr, INT8: tl.constexpr):
    """Load a [BK, BN] weight tile W[koff, noff] (row-major [K, NOUT]); dequantize per-column int8 if INT8."""
    w = tl.load(w_ptr + koff[:, None] * NOUT + noff[None, :])
    if INT8:
        sc = tl.load(sc_ptr + noff)
        return (w.to(tl.float32) * sc[None, :]).to(tl.bfloat16)
    else:
        return w


@triton.jit
def _row_stats(x_ptr, rows, rmask, D: tl.constexpr, BK: tl.constexpr, eps):
    s1 = tl.zeros([rows.shape[0]], dtype=tl.float32)
    s2 = tl.zeros([rows.shape[0]], dtype=tl.float32)
    for c in tl.static_range(D // BK):
        kk = c * BK + tl.arange(0, BK)
        xc = tl.load(x_ptr + rows[:, None] * D + kk[None, :], mask=rmask[:, None], other=0.0)
        s1 += tl.sum(xc, axis=1)
        s2 += tl.sum(xc * xc, axis=1)
    mean = s1 / D
    var = tl.maximum(s2 / D - mean * mean, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    return mean, rstd


@triton.jit
def _ln_kernel(x_ptr, lnw_ptr, lnb_ptr, h_ptr, M, eps, D: tl.constexpr, BM: tl.constexpr):
    """h = LN(x) in bf16 (used for long sequences where recomputing LN inside every GEMM tile is wasteful)"""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = tl.arange(0, D)
    x = tl.load(x_ptr + rows[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    mean = tl.sum(x, axis=1) / D
    xc = tl.where(rmask[:, None], x - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=1) / D + eps)
    h = xc * rstd[:, None] * tl.load(lnw_ptr + cols)[None, :] + tl.load(lnb_ptr + cols)[None, :]
    tl.store(h_ptr + rows[:, None] * D + cols[None, :], h.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _ln_gemm_kernel(x_ptr, lnw_ptr, lnb_ptr, w_ptr, sc_ptr, out_ptr, cos_ptr, sin_ptr, M, eps,
                    D: tl.constexpr, NOUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    ROPE: tl.constexpr, GELU: tl.constexpr, N_ROPE: tl.constexpr, INT8: tl.constexpr, PRENORM: tl.constexpr):
    """out[:, n-tile] = act( LN(x) @ W[:, n-tile] ) in bf16. ROPE tiles (pid_n*BN < N_ROPE) get rotary embedding
    (BN must equal head_dim; the tile is computed as two half-dots so rotate_half pairs are available)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    if PRENORM:
        mean = tl.zeros([BM], dtype=tl.float32)
        rstd = tl.zeros([BM], dtype=tl.float32)
    else:
        mean, rstd = _row_stats(x_ptr, rows, rmask, D, BK, eps)
    n0 = pid_n * BN
    if ROPE:
        HALF: tl.constexpr = BN // 2
        acc1 = tl.zeros([BM, HALF], dtype=tl.float32)
        acc2 = tl.zeros([BM, HALF], dtype=tl.float32)
        for k0 in range(0, D, BK):
            kk = k0 + tl.arange(0, BK)
            if PRENORM:
                hk = tl.load(x_ptr + rows[:, None] * D + kk[None, :], mask=rmask[:, None], other=0.0)
            else:
                hk = tl.load(x_ptr + rows[:, None] * D + kk[None, :], mask=rmask[:, None], other=0.0)
                hk = ((hk - mean[:, None]) * rstd[:, None] * tl.load(lnw_ptr + kk)[None, :] + tl.load(lnb_ptr + kk)[None, :]).to(tl.bfloat16)
            w1 = _wload(w_ptr, sc_ptr, kk, n0 + tl.arange(0, HALF), NOUT, INT8)
            w2 = _wload(w_ptr, sc_ptr, kk, n0 + HALF + tl.arange(0, HALF), NOUT, INT8)
            acc1 += tl.dot(hk, w1)
            acc2 += tl.dot(hk, w2)
        if n0 < N_ROPE:
            ang = tl.arange(0, HALF)
            c = tl.load(cos_ptr + rows[:, None] * HALF + ang[None, :], mask=rmask[:, None], other=1.0)
            s_ = tl.load(sin_ptr + rows[:, None] * HALF + ang[None, :], mask=rmask[:, None], other=0.0)
            o1 = acc1 * c - acc2 * s_
            o2 = acc2 * c + acc1 * s_
        else:
            o1 = acc1
            o2 = acc2
        tl.store(out_ptr + rows[:, None] * NOUT + (n0 + tl.arange(0, HALF))[None, :], o1.to(tl.bfloat16), mask=rmask[:, None])
        tl.store(out_ptr + rows[:, None] * NOUT + (n0 + HALF + tl.arange(0, HALF))[None, :], o2.to(tl.bfloat16), mask=rmask[:, None])
    else:
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, D, BK):
            kk = k0 + tl.arange(0, BK)
            if PRENORM:
                hk = tl.load(x_ptr + rows[:, None] * D + kk[None, :], mask=rmask[:, None], other=0.0)
            else:
                hk = tl.load(x_ptr + rows[:, None] * D + kk[None, :], mask=rmask[:, None], other=0.0)
                hk = ((hk - mean[:, None]) * rstd[:, None] * tl.load(lnw_ptr + kk)[None, :] + tl.load(lnb_ptr + kk)[None, :]).to(tl.bfloat16)
            w = _wload(w_ptr, sc_ptr, kk, n0 + tl.arange(0, BN), NOUT, INT8)
            acc += tl.dot(hk, w)
        if GELU:
            acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
        tl.store(out_ptr + rows[:, None] * NOUT + (n0 + tl.arange(0, BN))[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _attn_o_kernel(qkv_ptr, wo_ptr, sc_ptr, ls_ptr, x_ptr, M, scale,
                   D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr, INT8: tl.constexpr):
    """x[m-tile, n-tile] += ls * (attn(q,k,v)[m-tile] @ Wo[:, n-tile]); attention recomputed per n-tile (small T)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    tcols = tl.arange(0, BT)
    tmask = tcols < M
    hd = tl.arange(0, HD)
    ncols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    allowed = (tcols[None, :] <= rows[:, None]) & tmask[None, :]
    for h in range(H):
        q = tl.load(qkv_ptr + rows[:, None] * (3 * D) + (h * HD + hd)[None, :], mask=rmask[:, None], other=0.0)
        k = tl.load(qkv_ptr + tcols[:, None] * (3 * D) + (D + h * HD + hd)[None, :], mask=tmask[:, None], other=0.0)
        v = tl.load(qkv_ptr + tcols[:, None] * (3 * D) + (2 * D + h * HD + hd)[None, :], mask=tmask[:, None], other=0.0)
        wo = _wload(wo_ptr, sc_ptr, h * HD + hd, ncols, D, INT8)                     # [HD, BN]
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(allowed, s, float("-inf"))
        mx = tl.max(s, axis=1)
        p = tl.exp(s - mx[:, None])
        p = p / tl.sum(p, axis=1)[:, None]
        o = tl.dot(p.to(tl.bfloat16), v)                                          # [BM, HD] fp32
        acc += tl.dot(o.to(tl.bfloat16), wo)
    ls = tl.load(ls_ptr + ncols)
    xo = tl.load(x_ptr + rows[:, None] * D + ncols[None, :], mask=rmask[:, None], other=0.0)
    tl.store(x_ptr + rows[:, None] * D + ncols[None, :], xo + ls[None, :] * acc, mask=rmask[:, None])


@triton.jit
def _attn_o_batched_kernel(qkv_ptr, wo_ptr, sc_ptr, ls_ptr, x_ptr, M, scale,
                           D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr, INT8: tl.constexpr):
    """Same as _attn_o_kernel but all heads at once via batched dots (fewer reductions / syncs)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    tcols = tl.arange(0, BT)
    tmask = tcols < M
    dcols = tl.arange(0, D)
    ncols = pid_n * BN + tl.arange(0, BN)
    q = tl.load(qkv_ptr + rows[:, None] * (3 * D) + dcols[None, :], mask=rmask[:, None], other=0.0)          # [BM, D]
    k = tl.load(qkv_ptr + tcols[:, None] * (3 * D) + (D + dcols)[None, :], mask=tmask[:, None], other=0.0)     # [BT, D]
    v = tl.load(qkv_ptr + tcols[:, None] * (3 * D) + (2 * D + dcols)[None, :], mask=tmask[:, None], other=0.0)
    wo = _wload(wo_ptr, sc_ptr, dcols, ncols, D, INT8)                                                        # [D, BN]
    q3 = tl.permute(tl.reshape(q, [BM, H, HD]), (1, 0, 2))      # [H, BM, HD]
    k3 = tl.permute(tl.reshape(k, [BT, H, HD]), (1, 2, 0))      # [H, HD, BT]
    v3 = tl.permute(tl.reshape(v, [BT, H, HD]), (1, 0, 2))      # [H, BT, HD]
    s = tl.dot(q3, k3) * scale                                  # [H, BM, BT]
    allowed = (tcols[None, None, :] <= rows[None, :, None]) & tmask[None, None, :]
    s = tl.where(allowed, s, float("-inf"))
    mx = tl.max(s, axis=2)
    p = tl.exp(s - mx[:, :, None])
    p = p / tl.sum(p, axis=2)[:, :, None]
    o3 = tl.dot(p.to(tl.bfloat16), v3)                          # [H, BM, HD]
    o = tl.reshape(tl.permute(o3, (1, 0, 2)), [BM, D])          # [BM, D]
    acc = tl.dot(o.to(tl.bfloat16), wo)                         # [BM, BN]
    ls = tl.load(ls_ptr + ncols)
    xo = tl.load(x_ptr + rows[:, None] * D + ncols[None, :], mask=rmask[:, None], other=0.0)
    tl.store(x_ptr + rows[:, None] * D + ncols[None, :], xo + ls[None, :] * acc, mask=rmask[:, None])


@triton.jit
def _attn_flash_kernel(qkv_ptr, o_ptr, M, scale,
                       D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Causal flash attention (online softmax) for long sequences: o[m, h*HD:(h+1)*HD] = softmax(q k^T) v."""
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    hd = tl.arange(0, HD)
    q = tl.load(qkv_ptr + rows[:, None] * (3 * D) + (h * HD + hd)[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, HD], dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, M)
    for n0 in range(0, hi, BN):
        cols = n0 + tl.arange(0, BN)
        cmask = cols < M
        k = tl.load(qkv_ptr + cols[:, None] * (3 * D) + (D + h * HD + hd)[None, :], mask=cmask[:, None], other=0.0)
        v = tl.load(qkv_ptr + cols[:, None] * (3 * D) + (2 * D + h * HD + hd)[None, :], mask=cmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where((cols[None, :] <= rows[:, None]) & cmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + rows[:, None] * D + (h * HD + hd)[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _fc2_kernel(g_ptr, w_ptr, sc_ptr, ls_ptr, x_ptr, M,
                KIN: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NSPLIT: tl.constexpr, INT8: tl.constexpr):
    """x[:, n-tile] += ls * (g[:, k-split] @ W2[k-split, n-tile])   (fp32 atomic add over the K splits)"""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    ncols = pid_n * BN + tl.arange(0, BN)
    KS: tl.constexpr = KIN // NSPLIT
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(pid_s * KS, (pid_s + 1) * KS, BK):
        kk = k0 + tl.arange(0, BK)
        g = tl.load(g_ptr + rows[:, None] * KIN + kk[None, :], mask=rmask[:, None], other=0.0)
        w = _wload(w_ptr, sc_ptr, kk, ncols, D, INT8)
        acc += tl.dot(g, w)
    ls = tl.load(ls_ptr + ncols)
    if NSPLIT == 1:
        xo = tl.load(x_ptr + rows[:, None] * D + ncols[None, :], mask=rmask[:, None], other=0.0)
        tl.store(x_ptr + rows[:, None] * D + ncols[None, :], xo + ls[None, :] * acc, mask=rmask[:, None])
    else:
        tl.atomic_add(x_ptr + rows[:, None] * D + ncols[None, :], ls[None, :] * acc, mask=rmask[:, None], sem="relaxed")


def quantize_int8(w_kn: torch.Tensor):
    """[K, N] float -> (int8 [K, N], fp32 scale [N]) with per-output-channel symmetric scaling."""
    w = w_kn.float()
    scale = w.abs().amax(dim=0).clamp(min=1e-12) / 127.0
    q = torch.round(w / scale[None, :]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


# Tile config per token count, from a coordinate-descent sweep timed on CUDA-graph replays (RTX 5070 Ti, bf16).
# The constructor's values were a single fixed setting for every shape; the forward only ever specialised M > 256.
#   (bm, bk, bn_fc1, bn_fc2, bn_attn, nsplit, num_warps, num_stages)
_TILES = (
    (32, (16, 128, 64, 64, 32, 4, 4, 3)),      # M <= 32:  154.1 -> 137.1 us per 8-layer stack (measured at M = 25)
    (256, (32, 64, 128, 64, 64, 4, 4, 3)),     # M <= 256: 341.5 -> 268.7 us (measured at M = 125)
    (1024, (64, 32, 64, 64, 64, 4, 4, 3)),     # M <= 1024: 1269.0 -> 671.3 us (measured at M = 625)
    (1 << 30, (64, 32, 128, 64, 64, 2, 4, 2)),  # above:     3445.2 -> 2631.5 us (measured at M = 2500)
)


def _TILES_FOR(M: int):
    for hi, tile in _TILES:
        if M <= hi:
            return tile
    return None


class TritonTransformer:
    def __init__(self, layers, D=512, H=8, HD=64, FF=2048, eps=1e-5, theta=10000.0,
                 bm=32, bk=128, bn_fc1=64, bn_attn=64, bn_fc2=64, nsplit=4, num_warps=4, num_stages=3, max_bt=128, attn="loop",
                 wdtype="bf16"):
        self.D, self.H, self.HD, self.FF, self.eps, self.theta = D, H, HD, FF, eps, theta
        self.attn = attn
        self.int8 = wdtype == "int8"
        def prep(w_nk):   # nn.Linear weight [N, K] -> [K, N] (bf16) or (int8, scale)
            w = w_nk.float().t().contiguous()
            if self.int8:
                return quantize_int8(w)
            return w.bfloat16(), w.new_zeros(1)
        self.bm, self.bk, self.bn_fc1, self.bn_attn, self.bn_fc2, self.nsplit = bm, bk, bn_fc1, bn_attn, bn_fc2, nsplit
        self.num_warps, self.num_stages, self.max_bt = num_warps, num_stages, max_bt
        self.L = []
        for L in layers:
            wqkv, sqkv = prep(torch.cat([L["wq"], L["wk"], L["wv"]], 0))
            wo, so = prep(L["wo"])
            w1, s1 = prep(L["w1"])
            w2, s2 = prep(L["w2"])
            self.L.append(dict(
                ln1_w=L["ln1_w"].float().contiguous(), ln1_b=L["ln1_b"].float().contiguous(),
                wqkv=wqkv, sqkv=sqkv, wo=wo, so=so, ls1=L["ls1"].float().contiguous(),
                ln2_w=L["ln2_w"].float().contiguous(), ln2_b=L["ln2_b"].float().contiguous(),
                w1=w1, s1=s1, w2=w2, s2=s2, ls2=L["ls2"].float().contiguous()))
        dev = self.L[0]["wo"].device
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, HD, 2, device=dev).float() / HD))
        self._buf = {}

    def _buffers(self, M, dev):
        if M not in self._buf:
            pos = torch.arange(M, device=dev).float()
            ang = torch.outer(pos, self.inv_freq)                                    # [M, HD/2]
            self._buf[M] = dict(cos=ang.cos().contiguous(), sin=ang.sin().contiguous(),
                                qkv=torch.empty(M, 3 * self.D, dtype=torch.bfloat16, device=dev),
                                g=torch.empty(M, self.FF, dtype=torch.bfloat16, device=dev))
        return self._buf[M]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [M, D] fp32 contiguous (modified in place and returned)."""
        M, D = x.shape
        b = self._buffers(M, x.device)
        BM, BK, nw, ns = self.bm, self.bk, self.num_warps, self.num_stages
        bn_fc1, bn_fc2, nsplit = self.bn_fc1, self.bn_fc2, self.nsplit
        bn_attn = self.bn_attn
        tile = _TILES_FOR(M)
        if tile is not None:  # long sequences get real GEMM tiles (+ pre-normalized inputs, below) instead of the
            BM, BK, bn_fc1, bn_fc2, bn_attn, nsplit, nw, ns = tile   # latency-oriented tiny tiles
        mt = triton.cdiv(M, BM)
        BT = max(16, triton.next_power_of_2(M))
        fused_attn = BT <= self.max_bt
        prenorm = M > 256
        if not fused_attn and "o" not in b:
            b["o"] = torch.empty(M, D, dtype=torch.bfloat16, device=x.device)
        if prenorm and "h" not in b:
            b["h"] = torch.empty(M, D, dtype=torch.bfloat16, device=x.device)
        for L in self.L:
            i8 = self.int8
            if prenorm:
                _ln_kernel[(triton.cdiv(M, 32),)](x, L["ln1_w"], L["ln1_b"], b["h"], M, self.eps, D=D, BM=32, num_warps=4)
            _ln_gemm_kernel[(3 * D // self.HD, mt)](
                b["h"] if prenorm else x, L["ln1_w"], L["ln1_b"], L["wqkv"], L["sqkv"], b["qkv"], b["cos"], b["sin"], M, self.eps,
                D=D, NOUT=3 * D, BM=BM, BN=self.HD, BK=BK, ROPE=True, GELU=False, N_ROPE=2 * D, INT8=i8, PRENORM=prenorm, num_warps=nw, num_stages=ns)
            if fused_attn:
                attn_kernel = _attn_o_batched_kernel if self.attn == "batched" else _attn_o_kernel
                attn_kernel[(D // bn_attn, mt)](b["qkv"], L["wo"], L["so"], L["ls1"], x, M, 1.0 / math.sqrt(self.HD),
                                                D=D, H=self.H, HD=self.HD, BM=BM, BN=bn_attn, BT=BT, INT8=i8, num_warps=nw, num_stages=ns)
            else:
                BMa = 64
                _attn_flash_kernel[(triton.cdiv(M, BMa), self.H)](b["qkv"], b["o"], M, 1.0 / math.sqrt(self.HD),
                                                                   D=D, H=self.H, HD=self.HD, BM=BMa, BN=64, num_warps=4, num_stages=2)
                _fc2_kernel[(D // bn_fc2, mt, 1)](b["o"], L["wo"], L["so"], L["ls1"], x, M, KIN=D, D=D, BM=BM, BN=bn_fc2,
                                                  BK=BK, NSPLIT=1, INT8=i8, num_warps=nw, num_stages=ns)
            if prenorm:
                _ln_kernel[(triton.cdiv(M, 32),)](x, L["ln2_w"], L["ln2_b"], b["h"], M, self.eps, D=D, BM=32, num_warps=4)
            _ln_gemm_kernel[(self.FF // bn_fc1, mt)](
                b["h"] if prenorm else x, L["ln2_w"], L["ln2_b"], L["w1"], L["s1"], b["g"], b["cos"], b["sin"], M, self.eps,
                D=D, NOUT=self.FF, BM=BM, BN=bn_fc1, BK=BK, ROPE=False, GELU=True, N_ROPE=0, INT8=i8, PRENORM=prenorm, num_warps=nw, num_stages=ns)
            _fc2_kernel[(D // bn_fc2, mt, nsplit)](b["g"], L["w2"], L["s2"], L["ls2"], x, M, KIN=self.FF, D=D, BM=BM, BN=bn_fc2,
                                                   BK=BK, NSPLIT=nsplit, INT8=i8, num_warps=nw, num_stages=ns)
        return x
