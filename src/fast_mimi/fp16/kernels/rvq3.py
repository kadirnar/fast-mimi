"""Persistent two-stage RVQ search, v3: each CTA re-ranks its *own* local top-2 coarse candidates exactly
(fp32 rows gathered before the barrier), so the post-barrier critical path is: load keys -> min -> gather the
winner row (L2-hot) -> residual update -> coarse pass. Exact whenever the true nearest entry is within the
top-2 of its slice's fp16 coarse ranking (measured: always, on real Mimi embeddings).
grid = (NSPLIT, G groups of frame blocks); all CTAs co-resident.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _grid_barrier(bar_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release")
    cur = tl.atomic_add(bar_ptr, 0, sem="acquire")
    while cur < target:
        cur = tl.atomic_add(bar_ptr, 0, sem="acquire")
    tl.debug_barrier()


@triton.jit
def _stage(r, tile, escale, enorm, cb32_ptr, rows, rmask, cols, split, BLOCK_E: tl.constexpr, NOEXACT: tl.constexpr,
           INT8: tl.constexpr, NCAND: tl.constexpr):
    """coarse pass over this CTA's slice (fp16 or int8 tile) -> local top-NCAND -> exact fp32 distances ->
    packed key [BLOCK_N] (key = exact_dist_bits << 32 | global_idx; ties -> lowest index)"""
    r16 = r.to(tl.float16)
    rn = tl.sum(r16.to(tl.float32) * r16.to(tl.float32), axis=1)
    if INT8:
        t16 = (tile.to(tl.float32) * escale[:, None]).to(tl.float16)
    else:
        t16 = tile
    d = tl.maximum(rn[:, None] + enorm[None, :] - 2.0 * tl.dot(r16, tl.trans(t16)), 0.0)
    loc = tl.arange(0, BLOCK_E)
    i1 = tl.argmin(d, axis=1)
    g1 = split * BLOCK_E + i1
    if NOEXACT:
        d1 = tl.min(d, axis=1)
        return (d1.to(tl.int32, bitcast=True).to(tl.int64) << 32) | g1.to(tl.int64)
    d = tl.where(loc[None, :] == i1[:, None], float("inf"), d)
    i2 = tl.argmin(d, axis=1)
    g2 = split * BLOCK_E + i2
    D: tl.constexpr = cols.shape[0]
    row1 = tl.load(cb32_ptr + g1[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    row2 = tl.load(cb32_ptr + g2[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    if NCAND == 4:
        d = tl.where(loc[None, :] == i2[:, None], float("inf"), d)
        i3 = tl.argmin(d, axis=1)
        g3 = split * BLOCK_E + i3
        d = tl.where(loc[None, :] == i3[:, None], float("inf"), d)
        i4 = tl.argmin(d, axis=1)
        g4 = split * BLOCK_E + i4
        row3 = tl.load(cb32_ptr + g3[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        row4 = tl.load(cb32_ptr + g4[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
    df = r - row1
    e1 = tl.sum(df * df, axis=1)
    df = r - row2
    e2 = tl.sum(df * df, axis=1)
    take2 = (e2 < e1) | ((e2 == e1) & (g2 < g1))
    eb = tl.where(take2, e2, e1)
    gb = tl.where(take2, g2, g1)
    if NCAND == 4:
        df = r - row3
        e3 = tl.sum(df * df, axis=1)
        take3 = (e3 < eb) | ((e3 == eb) & (g3 < gb))
        eb = tl.where(take3, e3, eb)
        gb = tl.where(take3, g3, gb)
        df = r - row4
        e4 = tl.sum(df * df, axis=1)
        take4 = (e4 < eb) | ((e4 == eb) & (g4 < gb))
        eb = tl.where(take4, e4, eb)
        gb = tl.where(take4, g4, gb)
    return (eb.to(tl.int32, bitcast=True).to(tl.int64) << 32) | gb.to(tl.int64)


@triton.jit
def _rvq3_kernel(xs_ptr, xa_ptr, cb16_ptr, escale_ptr, cb32_ptr, enorm_ptr, keys_ptr, rpriv_ptr, codes_ptr, bar_ptr, N, T, K,
                 NSPLIT: tl.constexpr, G: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr, NOBAR: tl.constexpr, NOEXACT: tl.constexpr, NOKEYS: tl.constexpr,
                 INT8: tl.constexpr, NCAND: tl.constexpr):
    split = tl.program_id(0)
    g = tl.program_id(1)
    ent = split * BLOCK_E + tl.arange(0, BLOCK_E)
    cols = tl.arange(0, D)
    n_blocks = tl.cdiv(N, BLOCK_N)
    sidx = tl.arange(0, NSPLIT)
    bar = bar_ptr + g
    # keys layout: [K, NSPLIT, N] int64 ; rpriv: [NSPLIT, N, D] fp32 private residual copies
    # ---- stage 0 (semantic) and stage 1 (acoustic 0)
    tile = tl.load(cb16_ptr + ent[:, None] * D + cols[None, :])
    esc = tl.load(escale_ptr + ent)
    en = tl.load(enorm_ptr + ent)
    for blk in range(g, n_blocks, G):
        rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        r = tl.load(xs_ptr + rows[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        key = _stage(r, tile, esc, en, cb32_ptr, rows, rmask, cols, split, BLOCK_E, NOEXACT, INT8, NCAND)
        tl.store(keys_ptr + (0 * NSPLIT + split) * N + rows, key, mask=rmask)
    tile = tl.load(cb16_ptr + E * D + ent[:, None] * D + cols[None, :])
    esc = tl.load(escale_ptr + E + ent)
    en = tl.load(enorm_ptr + E + ent)
    for blk in range(g, n_blocks, G):
        rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        r = tl.load(xa_ptr + rows[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
        key = _stage(r, tile, esc, en, cb32_ptr + E * D, rows, rmask, cols, split, BLOCK_E, NOEXACT, INT8, NCAND)
        tl.store(keys_ptr + (1 * NSPLIT + split) * N + rows, key, mask=rmask)
        tl.store(rpriv_ptr + (split * N + rows[:, None]) * D + cols[None, :], r, mask=rmask[:, None])
    # ---- stages 2..K-1
    for k in range(2, K):
        tile = tl.load(cb16_ptr + k * E * D + ent[:, None] * D + cols[None, :])     # prefetch before barrier
        esc = tl.load(escale_ptr + k * E + ent)
        en = tl.load(enorm_ptr + k * E + ent)
        rows0 = g * BLOCK_N + tl.arange(0, BLOCK_N)
        r_pre = tl.load(rpriv_ptr + (split * N + rows0[:, None]) * D + cols[None, :], mask=(rows0 < N)[:, None], other=0.0)
        if not NOBAR:
            _grid_barrier(bar, (k - 1) * NSPLIT)
        for blk in range(g, n_blocks, G):
            rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
            rmask = rows < N
            if k == 2:   # emit the semantic code
                pk = tl.load(keys_ptr + (0 * NSPLIT + sidx[:, None]) * N + rows[None, :], mask=rmask[None, :], other=0, volatile=True)
                if split == 0:
                    tl.store(codes_ptr + ((rows // T) * K + 0) * T + rows % T, tl.min(pk, axis=0) & 0xFFFF, mask=rmask)
            if NOKEYS:
                pk1 = tl.load(keys_ptr + ((k - 1) * NSPLIT + split) * N + rows, mask=rmask, other=0)
                widx = (pk1 & 0xFFFF).to(tl.int32)
            else:
                pk = tl.load(keys_ptr + ((k - 1) * NSPLIT + sidx[:, None]) * N + rows[None, :], mask=rmask[None, :], other=0, volatile=True)
                widx = (tl.min(pk, axis=0) & 0xFFFF).to(tl.int32)
            if split == 0:
                tl.store(codes_ptr + ((rows // T) * K + (k - 1)) * T + rows % T, widx.to(tl.int64), mask=rmask)
            wrow = tl.load(cb32_ptr + (k - 1) * E * D + widx[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0)
            if blk == g:
                r = r_pre - wrow
            else:
                r = tl.load(rpriv_ptr + (split * N + rows[:, None]) * D + cols[None, :], mask=rmask[:, None], other=0.0) - wrow
            tl.store(rpriv_ptr + (split * N + rows[:, None]) * D + cols[None, :], r, mask=rmask[:, None])
            key = _stage(r, tile, esc, en, cb32_ptr + k * E * D, rows, rmask, cols, split, BLOCK_E, NOEXACT, INT8, NCAND)
            tl.store(keys_ptr + (k * NSPLIT + split) * N + rows, key, mask=rmask)
    # ---- finalize: last stage winner
    if not NOBAR:
        _grid_barrier(bar, (K - 1) * NSPLIT)
    if split == 0:
        for blk in range(g, n_blocks, G):
            rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
            rmask = rows < N
            pk = tl.load(keys_ptr + ((K - 1) * NSPLIT + sidx[:, None]) * N + rows[None, :], mask=rmask[None, :], other=0, volatile=True)
            tl.store(codes_ptr + ((rows // T) * K + (K - 1)) * T + rows % T, tl.min(pk, axis=0) & 0xFFFF, mask=rmask)


class RVQEncoder3:
    def __init__(self, sem_cb, ac_cb, block_n=16, block_e=64, num_warps=8, max_groups=4, num_stages=1, nobar=False, noexact=False, nokeys=False,
                 coarse="fp16", ncand=2):
        self.flags = dict(NOBAR=nobar, NOEXACT=noexact, NOKEYS=nokeys, INT8=coarse == "int8", NCAND=ncand)
        self.D = sem_cb.shape[-1]
        self.E = sem_cb.shape[0]
        cb = torch.cat([sem_cb[None], ac_cb], 0).float().contiguous()
        self.cb32 = cb
        if coarse == "int8":
            scale = cb.abs().amax(-1).clamp(min=1e-12) / 127.0                 # per-entry scale [K, E]
            self.cb16 = torch.round(cb / scale[..., None]).clamp(-127, 127).to(torch.int8).contiguous()
            self.escale = scale.float().contiguous()
            deq = self.cb16.float() * scale[..., None]
            self.enorm = deq.half().float().pow(2).sum(-1).contiguous()
        else:
            self.cb16 = cb.half().contiguous()
            self.escale = torch.ones(cb.shape[0], cb.shape[1], device=cb.device)
            self.enorm = self.cb16.float().pow(2).sum(-1).contiguous()
        self.K = cb.shape[0]
        self.block_n, self.block_e, self.num_warps, self.num_stages = block_n, block_e, num_warps, num_stages
        self.nsplit = self.E // block_e
        self.max_groups = max_groups
        self._buf = {}

    def encode(self, xs, xa, B, T, num_quantizers=None):
        K = num_quantizers or self.K
        N = xs.shape[0]
        n_blocks = triton.cdiv(N, self.block_n)
        G = max(1, min(self.max_groups, n_blocks))
        if N not in self._buf:
            self._buf[N] = (torch.empty(self.K * self.nsplit * N, dtype=torch.int64, device=xs.device),
                            torch.empty(self.nsplit * N * self.D, dtype=torch.float32, device=xs.device),
                            torch.zeros(self.max_groups, dtype=torch.int32, device=xs.device))
        keys, rpriv, bar = self._buf[N]
        bar.zero_()
        codes = torch.empty(B, K, T, dtype=torch.int64, device=xs.device)
        _rvq3_kernel[(self.nsplit, G)](xs, xa, self.cb16, self.escale, self.cb32, self.enorm, keys, rpriv, codes, bar, N, T, K,
                                       NSPLIT=self.nsplit, G=G, D=self.D, E=self.E, BLOCK_N=self.block_n,
                                       BLOCK_E=self.block_e, num_warps=self.num_warps, num_stages=self.num_stages, **self.flags)
        return codes
