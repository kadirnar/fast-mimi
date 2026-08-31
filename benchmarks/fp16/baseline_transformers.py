"""Reference-speed measurement of the stock transformers MimiModel (fp32, default attention)."""
from __future__ import annotations
import argparse, json, statistics, time
import torch


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 2, 5, 10, 30])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from transformers import MimiModel
    kw = {}
    if args.attn:
        kw["attn_implementation"] = args.attn
    model = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=getattr(torch, args.dtype), **kw).cuda().eval()
    print("attn_implementation:", model.config._attn_implementation)
    results = []
    with torch.inference_mode():
        for sec in args.seconds:
            n = int(24000 * sec)
            g = torch.Generator(device="cpu").manual_seed(20260826)
            x = torch.randn((1, 1, n), generator=g).to("cuda", getattr(torch, args.dtype))
            mask = torch.ones_like(x, dtype=torch.bool)
            codes = model.encode(x, mask).audio_codes
            enc_med, enc_min = bench(lambda: model.encode(x, mask), args.warmup, args.repeats)
            dec_med, dec_min = bench(lambda: model.decode(codes, mask), args.warmup, args.repeats)
            rt_med, rt_min = bench(lambda: model.decode(model.encode(x, mask).audio_codes, mask), args.warmup, args.repeats)
            r = dict(seconds=sec, samples=n, frames=int(codes.shape[-1]),
                     encode_ms=enc_med, encode_min_ms=enc_min,
                     decode_ms=dec_med, decode_min_ms=dec_min,
                     roundtrip_ms=rt_med, roundtrip_min_ms=rt_min)
            print(json.dumps(r), flush=True)
            results.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
