"""Does batching pay?  Per-item cost of encode + decode at batch 1..N, v4 vs transformers.

    python benchmarks/fp32/bench_batch.py --seconds 1 5 25

Batch 1 is latency bound on streaming the 384 MB of fp32 weights once; that cost is shared across a batch, so
the interesting number is milliseconds per item, not per call.  Codes are checked against the reference at every
batch size (a batch must not change a single code).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time

import torch

SR = 24_000


def signal(seconds: float, batch: int) -> torch.Tensor:
    n = int(SR * seconds)
    g = torch.Generator(device="cpu").manual_seed(20260826)
    noise = torch.randn((batch, 1, n), generator=g)
    t = torch.arange(n) / SR
    sweep = (0.5 * torch.sin(2 * math.pi * (200 + 3000 * t / max(seconds, 1e-6)) * t) * torch.exp(-t)).view(1, 1, n)
    return 0.1 * noise + sweep


def bench(fn, repeats: int, ramp: float = 0.4) -> float:
    with torch.inference_mode():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp:
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 5, 25])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from fast_mimi.fp32 import build, load_reference

    ref = load_reference()
    fast = build()
    rows = []
    print(f"{'audio':>6} {'batch':>6} {'transformers':>13} {'v4 total':>10} {'v4 / item':>10} "
          f"{'speedup':>8} {'vs batch 1':>11} {'codes':>7}")
    for sec in a.seconds:
        base = None
        for B in a.batches:
            try:
                x = signal(sec, B).cuda()
                mask = torch.ones_like(x, dtype=torch.bool)
                with torch.inference_mode():
                    rc = ref.encode(x, mask).audio_codes
                    fc = fast.encode(x, mask).audio_codes
                    same = bool(torch.equal(rc, fc))
                rep = max(6, a.repeats // max(1, B // 2))
                t_ref = bench(lambda: ref.decode(ref.encode(x, mask).audio_codes, mask).audio_values, rep)
                t_fast = bench(lambda: fast.decode(fast.encode(x, mask).audio_codes, mask).audio_values, rep)
            except torch.OutOfMemoryError:
                print(f"{sec:5g}s {B:6d}   out of memory")
                del x, mask
                gc.collect(); torch.cuda.empty_cache()
                break
            per = t_fast / B
            if base is None:
                base = per
            rows.append(dict(seconds=sec, batch=B, transformers_ms=t_ref, v4_ms=t_fast, v4_per_item_ms=per,
                             speedup=t_ref / t_fast, gain_vs_batch1=base / per, codes_identical=same))
            print(f"{sec:5g}s {B:6d} {t_ref:12.2f}ms {t_fast:9.2f}ms {per:9.3f}ms "
                  f"{t_ref / t_fast:7.1f}x {base / per:10.2f}x {'yes' if same else 'NO':>7}", flush=True)
            del x, mask
            gc.collect(); torch.cuda.empty_cache()
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
