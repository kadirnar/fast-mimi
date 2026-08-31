"""Tune the SEANet conv tiles at long-form shapes and add them to `src/fast_mimi/v4/tuned/conv1d_fp32.json`.

    python benchmarks/v4/tune_conv1d.py --seconds 25 100 --write

The shipped table was measured on ~1 s inputs; `_lookup` reuses a layer's entry at any N, so a 100 s round trip runs
tiles chosen for a 10 000x smaller output length.  This walks a real encode + decode at the requested lengths, records
every (kind, B, M, N, K, stride) a conv plan is actually called with, sweeps tile configs for each, checks the result
against the incumbent, and keeps the winner when it beats it by more than the measurement noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch


def signal(sec: float) -> torch.Tensor:
    n = int(24000 * sec)
    g = torch.Generator(device="cpu").manual_seed(20260826)
    t = torch.arange(n) / 24000
    return (0.1 * torch.randn((1, 1, n), generator=g)
            + (0.5 * torch.sin(2 * math.pi * (200 + 3000 * t / sec) * t) * torch.exp(-t)).view(1, 1, n))


def bench(fn, n: int = 7) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


COLS_CAP = 64 << 20     # skip the im2col backend when its `cols` buffer would exceed this (it is linear in N)


def candidates(M: int, N: int, KK: int):
    """Triton tiles (with split-K where the output is small enough to want more CTAs) plus the cuBLAS im2col path.

    The last two used to be missing: `SPLIT` was pinned to 1 and the `cublas` entries were hand-placed, so the
    small-N / large-K layers -- which is where the Triton fp32 dot is 1.6-2.5x off a plain cuBLAS GEMM of the same
    shape -- were never offered either.
    """
    out = []
    for BM in (16, 32, 64, 128):
        if BM > 2 * M:
            continue
        for BN in (32, 64, 128, 256):
            if BN > 2 * N:
                continue
            for BK in (16, 32, 64):
                if BK > KK:
                    continue
                for warps, stages in ((4, 2), (4, 3), (8, 3)):
                    out.append(dict(BM=BM, BN=BN, BK=BK, SPLIT=1, num_warps=warps, num_stages=stages))
                    ctas = (M // BM + 1) * (N // BN + 1)
                    if ctas < 280:                      # fewer than ~4 waves: worth splitting K for more CTAs
                        for split in (2, 4, 8):
                            if KK // split >= BK:
                                out.append(dict(BM=BM, BN=BN, BK=BK, SPLIT=split, num_warps=warps, num_stages=stages))
    if (KK + 1) * N * 4 <= COLS_CAP:
        for variant in ("D", "E"):
            for kb_pad in (True, False):
                out.append(dict(backend="cublas", variant=variant, kb_pad=kb_pad))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[25, 100])
    ap.add_argument("--gain", type=float, default=0.03, help="minimum relative improvement to keep a new entry")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="src/fast_mimi/v4/tuned/conv1d_fp32.json")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from fast_mimi.v4 import build
    from fast_mimi.v4.kernels import conv1d_fp32 as C

    model = build()

    # --- record the shapes a real round trip drives through each plan -------------------------------------
    seen: dict[str, tuple] = {}
    orig_conv, orig_convt = C.Conv1dPlan.__call__, C.ConvT1dPlan.__call__

    def rec_conv(self, x, pad_left=0, pad_right=0, *args, **kw):
        B, _, T_x = x.shape
        N = (T_x + pad_left + pad_right - self.KS) // self.stride + 1
        kind = "conv1" if self.polyphase else "conv0"
        key = C._key(kind, B, self.Cout, N, self.Cin * self.KS, self.stride)
        seen.setdefault(key, ("conv", self, B, T_x, pad_left, pad_right, N))
        return orig_conv(self, x, pad_left, pad_right, *args, **kw)

    def rec_convt(self, x, *args, **kw):
        B, _, T_in = x.shape
        key = C._key("convt", B, self.Cout * self.stride, T_in, self.Cin * 2, self.stride)
        seen.setdefault(key, ("convt", self, B, T_in, 0, 0, T_in))
        return orig_convt(self, x, *args, **kw)

    C.Conv1dPlan.__call__, C.ConvT1dPlan.__call__ = rec_conv, rec_convt
    from fast_mimi.v4 import eager_mode
    with torch.inference_mode(), eager_mode():
        for sec in a.seconds:
            x = signal(sec).cuda()
            mask = torch.ones_like(x, dtype=torch.bool)
            model.decode(model.encode(x, mask).audio_codes, mask)
            del x, mask
            torch.cuda.empty_cache()
    C.Conv1dPlan.__call__, C.ConvT1dPlan.__call__ = orig_conv, orig_convt
    del model
    torch.cuda.empty_cache()

    tuned = dict(C._load_tuned())
    added = 0
    for key, (kind, plan, B, T, pl, pr, N) in sorted(seen.items(), key=lambda kv: -kv[1][6]):
        Cin = plan.Cin
        x = torch.randn((B, Cin, T), device="cuda")
        if kind == "conv":
            mode = C.MODE_S2C if plan.polyphase else C.MODE_DIRECT
            M, KK = plan.Cout, plan.Cin * plan.KS
            run = lambda cfg: plan(x, pl, pr, mode=mode, cfg=cfg)          # noqa: E731
            base_mode, base_cfg = plan._plan(B, N)
            base_run = lambda: plan(x, pl, pr, mode=base_mode, cfg=base_cfg)   # noqa: E731
        else:
            M, KK = plan.Cout * plan.stride, plan.Cin * 2
            run = lambda cfg: plan(x, cfg=cfg)                             # noqa: E731
            base_cfg = plan._plan(B, N)
            base_run = lambda: plan(x, cfg=base_cfg)                       # noqa: E731
        try:
            ref = base_run().clone()
            t_base = bench(base_run)
        except Exception:
            torch.cuda.empty_cache()
            continue
        best, best_t = None, t_base
        for cfg in candidates(M, N, KK):
            try:
                out = run(cfg)
                if not torch.isfinite(out).all() or (out - ref).abs().max().item() > 1e-3 * ref.abs().max().item():
                    continue
                t = bench(lambda: run(cfg), 5)
            except Exception:                       # OOM, out of shared memory, unsupported tile
                torch.cuda.empty_cache()
                continue
            if t < best_t:
                best, best_t = cfg, t
        # the incumbent is timed first, when the caching allocator is still growing into this shape's buffers;
        # time it again after the sweep and keep the better of the two, so the comparison is not order-biased
        try:
            t_base = min(t_base, bench(base_run))
        except Exception:
            pass
        mark = ""
        if best is not None and best_t < t_base * (1 - a.gain):
            ent = dict(best)
            ent.update(backend=best.get("backend", "triton"), incumbent_us=round(t_base, 2), us=round(best_t, 2),
                       tuned_at="long-form")
            tuned[key] = ent
            added += 1
            mark = f"  KEEP {best}"
        print(f"{key:44s} incumbent {t_base:9.1f} us -> best {best_t:9.1f} us ({t_base / best_t:4.2f}x){mark}", flush=True)
        del x, ref
        torch.cuda.empty_cache()

    print(f"\n{added} new entries")
    if a.write and added:
        with open(a.out, "w") as f:
            json.dump(dict(sorted(tuned.items())), f, indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
