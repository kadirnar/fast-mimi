"""fast-mimi FP32 timing rows in the FP16 benchmark schema (same protocol and seeded input as benchmarks/fp16/bench.py)."""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def bench(fn, warmup, repeats, warm_seconds=1.0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warm_seconds:
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 2, 5, 10, 30])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from fast_mimi.fp32 import build, load_reference
    ref = load_reference()
    model = build()
    results = []
    with torch.inference_mode():
        for sec in args.seconds:
            n = int(24000 * sec)
            g = torch.Generator(device="cpu").manual_seed(20260826)
            x = torch.randn((1, 1, n), generator=g).cuda()
            mask = torch.ones_like(x, dtype=torch.bool)
            codes = model.encode(x, mask).audio_codes.clone()
            ref_codes = ref.encode(x, mask).audio_codes
            match = float((codes == ref_codes).float().mean().item())
            enc_med, enc_min = bench(lambda: model.encode(x, mask), args.warmup, args.repeats)
            dec_med, dec_min = bench(lambda: model.decode(codes, mask), args.warmup, args.repeats)
            rt_med, rt_min = bench(lambda: model.decode(model.encode(x, mask).audio_codes, mask), args.warmup, args.repeats)
            r = dict(seconds=sec, samples=n, frames=int(codes.shape[-1]), encode_ms=enc_med, encode_min_ms=enc_min,
                     decode_ms=dec_med, decode_min_ms=dec_min, roundtrip_ms=rt_med, roundtrip_min_ms=rt_min, code_match=match)
            print(json.dumps(r), flush=True)
            results.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
