"""Small fused kernels gluing the Mimi stages together:
  down_proj:      25 Hz transformer output -> 12.5 Hz (k=4, s=2, replicate pad) -> semantic/acoustic VQ projections
  rvq_decode_up:  codes -> sum of (output-projected) codebook rows -> 2x depthwise transposed-conv upsample
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _down_proj_kernel(x_ptr, w_ptr, out_ptr, T_in, T_out,
                      D: tl.constexpr, NOUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[t, n] += sum_k x[clamp(2t + j - 2), k] * w[j, k, n] for tap j = program_id(2)   (fp32 atomics over taps;
    w: [4, D, NOUT] bf16; x fp32 split hi/lo for ~fp32 accuracy)"""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    j = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < T_out
    ncols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    r = tl.minimum(tl.maximum(2 * rows + j - 2, 0), T_in - 1)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + r[:, None] * D + kk[None, :], mask=rmask[:, None], other=0.0)
        a_hi = a.to(tl.bfloat16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
        w = tl.load(w_ptr + (j * D + kk)[:, None] * NOUT + ncols[None, :])
        acc += tl.dot(a_hi, w) + tl.dot(a_lo, w)
    tl.atomic_add(out_ptr + rows[:, None] * NOUT + ncols[None, :], acc, mask=rmask[:, None], sem="relaxed")


@triton.jit
def _rvq_decode_up_kernel(codes_ptr, p_ptr, wu_ptr, y_ptr, T, K,
                          D: tl.constexpr, E: tl.constexpr, BT: tl.constexpr, KG: tl.constexpr):
    """y[2t+s, :] += sum_{k in group} (P[k, codes[k,t]] * wu[:, s] + P[k, codes[k,t-1]] * wu[:, s+2])   (fp32 atomics)"""
    pid = tl.program_id(0)
    grp = tl.program_id(1)
    t = pid * BT + tl.arange(0, BT)
    tmask = t < T
    tp = t - 1
    pmask = (tp >= 0) & tmask
    cols = tl.arange(0, D)
    e = tl.zeros([BT, D], dtype=tl.float32)
    ep = tl.zeros([BT, D], dtype=tl.float32)
    for kk in tl.static_range(KG):
        k = grp * KG + kk
        c = tl.load(codes_ptr + k * T + t, mask=tmask, other=0)
        cp = tl.load(codes_ptr + k * T + tp, mask=pmask, other=0)
        e += tl.load(p_ptr + (k * E + c)[:, None] * D + cols[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
        ep += tl.load(p_ptr + (k * E + cp)[:, None] * D + cols[None, :], mask=pmask[:, None], other=0.0).to(tl.float32)
    w0 = tl.load(wu_ptr + cols * 4 + 0)
    w1 = tl.load(wu_ptr + cols * 4 + 1)
    w2 = tl.load(wu_ptr + cols * 4 + 2)
    w3 = tl.load(wu_ptr + cols * 4 + 3)
    tl.atomic_add(y_ptr + (2 * t)[:, None] * D + cols[None, :], e * w0[None, :] + ep * w2[None, :], mask=tmask[:, None], sem="relaxed")
    tl.atomic_add(y_ptr + (2 * t + 1)[:, None] * D + cols[None, :], e * w1[None, :] + ep * w3[None, :], mask=tmask[:, None], sem="relaxed")


class DownProj:
    def __init__(self, down_w, sem_in, ac_in, bm=16, bn=32, bk=128, num_warps=4):
        # down_w: [512, 512, 4] (out, in, k) ; sem_in/ac_in: [256, 512]
        D = down_w.shape[1]
        w_in = torch.cat([sem_in, ac_in], 0).float()                     # [512, 512]
        wc = torch.stack([w_in @ down_w[:, :, j].float() for j in range(4)])   # [4, NOUT=512, K=D] (out, in)
        self.w = wc.permute(0, 2, 1).contiguous().bfloat16()              # [4, D, NOUT]
        self.D, self.NOUT = D, w_in.shape[0]
        self.bm, self.bn, self.bk, self.num_warps = bm, bn, bk, num_warps

    def __call__(self, x, out):
        T_in, T_out = x.shape[0], out.shape[0]
        out.zero_()
        _down_proj_kernel[(self.NOUT // self.bn, triton.cdiv(T_out, self.bm), 4)](x, self.w, out, T_in, T_out, D=self.D, NOUT=self.NOUT,
                                                                                BM=self.bm, BN=self.bn, BK=self.bk, num_warps=self.num_warps)
        return out


class RVQDecodeUp:
    def __init__(self, sem_cb, ac_cb, sem_out, ac_out, up_w, dtype=torch.bfloat16, bt=8, num_warps=4, kg=4):
        # projected codebooks P[k] = cb_k @ W_out^T  -> [K, E, 512]
        p = [sem_cb.float() @ sem_out.float().T] + [ac_cb[i].float() @ ac_out.float().T for i in range(ac_cb.shape[0])]
        self.p = torch.stack(p).to(dtype).contiguous()
        self.wu = up_w.float().reshape(-1, 4).contiguous()                # [512, 4]
        self.K, self.E, self.D = self.p.shape
        self.bt, self.num_warps, self.kg = bt, num_warps, kg

    def __call__(self, codes, y):
        """codes [K, T] int64 (one batch element) -> y [2T, 512] fp32 (accumulated with atomics)"""
        K, T = codes.shape
        assert K % self.kg == 0
        y.zero_()
        _rvq_decode_up_kernel[(triton.cdiv(T, self.bt), K // self.kg)](codes, self.p, self.wu, y, T, K, D=self.D, E=self.E, BT=self.bt, KG=self.kg, num_warps=self.num_warps)
        return y
