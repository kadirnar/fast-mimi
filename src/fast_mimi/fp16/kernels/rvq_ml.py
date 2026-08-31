"""Multi-launch two-stage RVQ search (one kernel per codebook stage) for large frame counts, where a persistent
kernel cannot keep all frame groups co-resident. Same numerics as rvq3: fp16 coarse pass per codebook slice,
exact fp32 re-rank of the local top-2, global argmin via atomic_min on packed keys."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _rvq_ml_stage(x_ptr, r_in_ptr, r_out_ptr, keys_prev_ptr, keys_ptr, cb16_ptr, cb32_prev_ptr, cb32_ptr, enorm_ptr, codes_prev_ptr, N, T, K,
                  k_prev, HAS_PREV: tl.constexpr, WRITE_R: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr,
                  BPC: tl.constexpr):
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    cols = tl.arange(0, D)
    ent = split * BLOCK_E + tl.arange(0, BLOCK_E)
    tile = tl.load(cb16_ptr + ent[:, None] * D + cols[None, :])
    en = tl.load(enorm_ptr + ent)
    for bi in range(BPC):
        rows = (pid_n * BPC + bi) * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        _rvq_ml_block(x_ptr, r_in_ptr, r_out_ptr, keys_prev_ptr, keys_ptr, tile, en, cb32_prev_ptr, cb32_ptr, codes_prev_ptr, rows, rmask, cols, ent,
                      split, T, K, k_prev, HAS_PREV, WRITE_R, D, BLOCK_E)


@triton.jit
def _rvq_ml_block(x_ptr, r_in_ptr, r_out_ptr, keys_prev_ptr, keys_ptr, tile, en, cb32_prev_ptr, cb32_ptr, codes_prev_ptr, rows, rmask, cols, ent,
                  split, T, K, k_prev, HAS_PREV: tl.constexpr, WRITE_R: tl.constexpr, D: tl.constexpr, BLOCK_E: tl.constexpr):
    r = tl.load(r_in_ptr + rows[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    if HAS_PREV:
        kp = tl.load(keys_prev_ptr + rows, mask=rmask, other=0)
        widx = (kp & 0xFFFF).to(tl.int32)
        r -= tl.load(cb32_prev_ptr + widx[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        if split == 0:
            tl.store(codes_prev_ptr + ((rows // T) * K + k_prev) * T + rows % T, widx.to(tl.int64), mask=rmask)
            if WRITE_R:
                tl.store(r_out_ptr + rows[:, None] * D + cols[None, :], r, mask=rmask[:, None])
    r16 = r.to(tl.float16)
    rn = tl.sum(r16.to(tl.float32) * r16.to(tl.float32), axis=1)
    d = tl.maximum(rn[:, None] + en[None, :] - 2.0 * tl.dot(r16, tl.trans(tile)), 0.0)
    loc = tl.arange(0, BLOCK_E)
    i1 = tl.argmin(d, axis=1)
    d2 = tl.where(loc[None, :] == i1[:, None], float("inf"), d)
    i2 = tl.argmin(d2, axis=1)
    g1 = split * BLOCK_E + i1
    g2 = split * BLOCK_E + i2
    row1 = tl.load(cb32_ptr + g1[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    row2 = tl.load(cb32_ptr + g2[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    df = r - row1
    e1 = tl.sum(df * df, axis=1)
    df = r - row2
    e2 = tl.sum(df * df, axis=1)
    take2 = (e2 < e1) | ((e2 == e1) & (g2 < g1))
    eb = tl.where(take2, e2, e1)
    gb = tl.where(take2, g2, g1)
    key = (eb.to(tl.int32, bitcast=True).to(tl.int64) << 32) | gb.to(tl.int64)
    tl.atomic_min(keys_ptr + rows, key, mask=rmask)


@triton.jit
def _rvq_ml_block_chunked(x_ptr, r_in_ptr, r_out_ptr, keys_prev_ptr, keys_ptr, cb16_ptr, cb32_prev_ptr, cb32_ptr, enorm_ptr, codes_prev_ptr,
                          N, T, K, k_prev, HAS_PREV: tl.constexpr, WRITE_R: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_E: tl.constexpr, DC: tl.constexpr):
    """Register-lean variant: everything streamed in DC-wide chunks of the 256-dim vectors (no full tiles live)."""
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    ent = split * BLOCK_E + tl.arange(0, BLOCK_E)
    en = tl.load(enorm_ptr + ent)
    if HAS_PREV:
        kp = tl.load(keys_prev_ptr + rows, mask=rmask, other=0)
        widx = (kp & 0xFFFF).to(tl.int32)
        if split == 0:
            tl.store(codes_prev_ptr + ((rows // T) * K + k_prev) * T + rows % T, widx.to(tl.int64), mask=rmask)
    else:
        widx = tl.zeros([BLOCK_N], dtype=tl.int32)
    # ---- coarse pass (fp16 tensor cores), chunked over D
    acc = tl.zeros([BLOCK_N, BLOCK_E], dtype=tl.float32)
    rn = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in tl.static_range(D // DC):
        cols = c * DC + tl.arange(0, DC)
        rc = tl.load(r_in_ptr + rows[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        if HAS_PREV:
            rc -= tl.load(cb32_prev_ptr + widx[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
            if WRITE_R:
                if split == 0:
                    tl.store(r_out_ptr + rows[:, None] * D + cols[None, :], rc, mask=rmask[:, None])
        r16 = rc.to(tl.float16)
        rn += tl.sum(r16.to(tl.float32) * r16.to(tl.float32), axis=1)
        tile = tl.load(cb16_ptr + ent[:, None] * D + cols[None, :])
        acc += tl.dot(r16, tl.trans(tile))
    d = tl.maximum(rn[:, None] + en[None, :] - 2.0 * acc, 0.0)
    loc = tl.arange(0, BLOCK_E)
    i1 = tl.argmin(d, axis=1)
    d2 = tl.where(loc[None, :] == i1[:, None], float("inf"), d)
    i2 = tl.argmin(d2, axis=1)
    g1 = split * BLOCK_E + i1
    g2 = split * BLOCK_E + i2
    # ---- exact fp32 distances of the two local candidates, chunked
    e1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    e2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in tl.static_range(D // DC):
        cols = c * DC + tl.arange(0, DC)
        rc = tl.load(r_in_ptr + rows[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        if HAS_PREV:
            rc -= tl.load(cb32_prev_ptr + widx[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        df = rc - tl.load(cb32_ptr + g1[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        e1 += tl.sum(df * df, axis=1)
        df = rc - tl.load(cb32_ptr + g2[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        e2 += tl.sum(df * df, axis=1)
    take2 = (e2 < e1) | ((e2 == e1) & (g2 < g1))
    eb = tl.where(take2, e2, e1)
    gb = tl.where(take2, g2, g1)
    key = (eb.to(tl.int32, bitcast=True).to(tl.int64) << 32) | gb.to(tl.int64)
    tl.atomic_min(keys_ptr + rows, key, mask=rmask)


@triton.jit
def _rvq_ml_final(keys_ptr, codes_ptr, N, T, K, k, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    kp = tl.load(keys_ptr + i, mask=m, other=0)
    tl.store(codes_ptr + ((i // T) * K + k) * T + i % T, kp & 0xFFFF, mask=m)


class RVQEncoderML:
    def __init__(self, sem_cb, ac_cb, block_n=16, block_e=64, num_warps=4, blocks_per_cta=None, chunked=True, dc=32):
        self.bpc = blocks_per_cta
        self.chunked, self.dc = chunked, dc
        self.D = sem_cb.shape[-1]
        self.E = sem_cb.shape[0]
        cb = torch.cat([sem_cb[None], ac_cb], 0).float().contiguous()
        self.cb32 = cb
        self.cb16 = cb.half().contiguous()
        self.enorm = self.cb16.float().pow(2).sum(-1).contiguous()
        self.K = cb.shape[0]
        self.block_n, self.block_e, self.num_warps = block_n, block_e, num_warps
        self.nsplit = self.E // block_e
        self._buf = {}

    def encode(self, xs, xa, B, T, num_quantizers=None):
        K = num_quantizers or self.K
        N = xs.shape[0]
        if N not in self._buf:
            self._buf[N] = (torch.empty(self.K, N, dtype=torch.int64, device=xs.device),
                            torch.empty(2, N, self.D, dtype=torch.float32, device=xs.device))
        keys, rbuf = self._buf[N]
        keys.fill_(torch.iinfo(torch.int64).max)
        codes = torch.empty(B, K, T, dtype=torch.int64, device=xs.device)
        n_blocks = triton.cdiv(N, self.block_n)
        bpc = self.bpc or 1                                        # measured: 1 block per CTA is fastest
        grid = (triton.cdiv(n_blocks, bpc), self.nsplit)
        if self.chunked:
            kern = _rvq_ml_block_chunked
            grid = (n_blocks, self.nsplit)
            kw = dict(D=self.D, BLOCK_N=self.block_n, BLOCK_E=self.block_e, DC=self.dc, num_warps=self.num_warps)
        else:
            kern = _rvq_ml_stage
            kw = dict(D=self.D, BLOCK_N=self.block_n, BLOCK_E=self.block_e, BPC=bpc, num_warps=self.num_warps)
        # semantic stage (keys[0]) reads xs; acoustic stage 1 reads xa (no prev)
        kern[grid](xs, xs, rbuf[0], keys[0], keys[0], self.cb16[0], self.cb32[0], self.cb32[0], self.enorm[0], codes, N, T, K, 0,
                   HAS_PREV=False, WRITE_R=False, **kw)
        kern[grid](xa, xa, rbuf[1], keys[1], keys[1], self.cb16[1], self.cb32[1], self.cb32[1], self.enorm[1], codes, N, T, K, 0,
                   HAS_PREV=False, WRITE_R=False, **kw)
        r_in = xa
        for k in range(2, K):
            r_out = rbuf[k % 2]
            kern[grid](xa, r_in, r_out, keys[k - 1], keys[k], self.cb16[k], self.cb32[k - 1], self.cb32[k], self.enorm[k], codes, N, T, K, k - 1,
                       HAS_PREV=True, WRITE_R=True, **kw)
            r_in = r_out
        _rvq_ml_final[(triton.cdiv(N, 256),)](keys[0], codes, N, T, K, 0, BLOCK=256)
        _rvq_ml_final[(triton.cdiv(N, 256),)](keys[K - 1], codes, N, T, K, K - 1, BLOCK=256)
        return codes
