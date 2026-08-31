"""Benchmark fast-mimi backends with exactly the protocol used for the transformers baseline."""
from __future__ import annotations
import argparse, json, statistics, sys, time
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from fast_mimi.fp16 import FastMimi, load_mimi_state


def bench(fn, warmup, repeats, warm_seconds=1.0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warm_seconds:      # let GPU clocks ramp up
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


# transformers MimiModel fp32/SDPA, batch 1, median of 50 after 1 s warm-up (bench/baseline_transformers.py): (encode, decode, roundtrip) ms
BASELINE = {1: (12.39, 6.97, 19.34), 2: (12.80, 7.09, 19.87), 5: (13.41, 7.63, 21.02), 10: (15.49, 9.78, 25.25), 30: (25.46, 20.29, 45.99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 2, 5, 10, 30])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    state = load_mimi_state("kyutai/mimi")
    dtype = getattr(torch, args.dtype)
    if args.backend == "torch":
        model = FastMimi(state, dtype=dtype)
    else:
        from fast_mimi.fp16.backends import build
        model = build(args.backend, state, dtype=dtype)
    results = []
    with torch.inference_mode():
        for sec in args.seconds:
            n = int(24000 * sec)
            g = torch.Generator(device="cpu").manual_seed(20260826)
            x = torch.randn((1, 1, n), generator=g).cuda()
            codes = model.encode(x)
            enc_med, enc_min = bench(lambda: model.encode(x), args.warmup, args.repeats)
            dec_med, dec_min = bench(lambda: model.decode(codes, length=n), args.warmup, args.repeats)
            rt_med, rt_min = bench(lambda: model.decode(model.encode(x), length=n), args.warmup, args.repeats)
            r = dict(backend=args.backend, dtype=args.dtype, seconds=sec, frames=int(codes.shape[-1]),
                     encode_ms=enc_med, decode_ms=dec_med, roundtrip_ms=rt_med, roundtrip_min_ms=rt_min)
            b = BASELINE.get(int(sec) if float(sec).is_integer() else sec)
            if b:
                r.update(speedup_encode=b[0] / enc_med, speedup_decode=b[1] / dec_med, speedup_roundtrip=b[2] / rt_med)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            results.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
