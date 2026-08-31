"""transformers MimiModel (fp32, TF32 off) vs fast-mimi v4: timing, exactness, table and chart.

    python benchmarks/v4/bench_v4.py --seconds 1 2 5 10 25 50 100 \
        --out-json benchmarks/v4/results.json --out-md docs/v4/RESULTS.md --chart assets/v4-benchmark.svg

Protocol (identical for both models, per length): seeded synthetic 24 kHz audio (0.1 x white noise + a decaying
sine sweep), batch 1; encode / decode(reference codes) / round trip timed separately; 3 warm-up calls, a 0.5 s
clock ramp, then N CUDA-synchronised runs; the median is reported.  Exactness: discrete codes `torch.equal`,
decoded audio `allclose(rtol=2e-4, atol=2e-5)`, two runs bitwise identical.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch

SR = 24_000


def signal(seconds: float, seed: int, batch: int = 1) -> torch.Tensor:
    n = int(SR * seconds)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((batch, 1, n), generator=g)
    t = torch.arange(n) / SR
    sweep = (0.5 * torch.sin(2 * math.pi * (200 + 3000 * t / max(seconds, 1e-6)) * t) * torch.exp(-t)).view(1, 1, n)
    return 0.1 * noise + sweep


def time_callable(fn, warmup: int, repeats: int, ramp_seconds: float) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp_seconds:
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))]}


def bench_one(ref, fast, seconds: float, repeats: int, warmup: int, ramp: float) -> dict:
    x = signal(seconds, 20260826).cuda()
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.inference_mode():
        rc = ref.encode(x, mask).audio_codes.clone()
        ra = ref.decode(rc, mask).audio_values.clone()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    r_enc = time_callable(lambda: ref.encode(x, mask).audio_codes, warmup, repeats, ramp)
    r_dec = time_callable(lambda: ref.decode(rc, mask).audio_values, warmup, repeats, ramp)
    r_rt = time_callable(lambda: ref.decode(ref.encode(x, mask).audio_codes, mask).audio_values, warmup, repeats, ramp)
    ref_vram = torch.cuda.max_memory_allocated() / 1e9
    with torch.inference_mode():
        fc = fast.encode(x, mask).audio_codes.clone()
        fa = fast.decode(fc, mask).audio_values.clone()
        fc2 = fast.encode(x, mask).audio_codes.clone()
        fa2 = fast.decode(fc2, mask).audio_values.clone()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    f_enc = time_callable(lambda: fast.encode(x, mask).audio_codes, warmup, repeats, ramp)
    f_dec = time_callable(lambda: fast.decode(rc, mask).audio_values, warmup, repeats, ramp)
    f_rt = time_callable(lambda: fast.decode(fast.encode(x, mask).audio_codes, mask).audio_values, warmup, repeats, ramp)
    fast_vram = torch.cuda.max_memory_allocated() / 1e9
    diff = (ra - fa).abs()
    row = {
        "seconds": seconds, "frames": int(rc.shape[-1]), "repeats": repeats,
        "transformers": {"encode_ms": r_enc["median_ms"], "decode_ms": r_dec["median_ms"], "roundtrip_ms": r_rt["median_ms"], "vram_gb": ref_vram},
        "v4": {"encode_ms": f_enc["median_ms"], "decode_ms": f_dec["median_ms"], "roundtrip_ms": f_rt["median_ms"], "vram_gb": fast_vram},
        "codes_identical": bool(torch.equal(rc, fc)),
        "codes_differing": int((rc != fc).sum().item()), "codes_total": int(rc.numel()),
        "audio_within_tolerance": bool(torch.allclose(ra, fa, rtol=2e-4, atol=2e-5)),
        "audio_max_abs_diff": float(diff.max().item()),
        "deterministic": bool(torch.equal(fc, fc2) and torch.equal(fa, fa2)),
    }
    row["speedup"] = {k: row["transformers"][f"{k}_ms"] / row["v4"][f"{k}_ms"] for k in ("encode", "decode", "roundtrip")}
    row["rtf"] = seconds * 1000.0 / row["v4"]["roundtrip_ms"]
    return row


def write_markdown(rows: list[dict], path: str, gpu: str) -> str:
    f = lambda r: f"{r['encode_ms']:.2f} / {r['decode_ms']:.2f} / {r['roundtrip_ms']:.3f}"
    lines = [
        f"# fast-mimi v4 (fp32-exact) results ({datetime.date.today()})", "",
        f"GPU: {gpu}. Batch 1, seeded synthetic 24 kHz audio, all 32 codebooks, CUDA-synchronised wall clock, median of N runs after 3 warm-ups and a 0.5 s clock ramp, same inputs for both models.",
        "Baseline: `transformers` `MimiModel` fp32 with TF32 off (SDPA attention). v4: the same `MimiModel` object patched in place by `fast_mimi.v4.build()` (exact fp32 kernels + CUDA graphs).", "",
        "| audio | frames | transformers enc / dec / roundtrip (ms) | v4 enc / dec / roundtrip (ms) | speedup enc / dec / **roundtrip** | real-time factor | codes identical | audio within tol (max abs diff) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        s = r["speedup"]
        lines.append(f"| {r['seconds']:g} s | {r['frames']} | {f(r['transformers'])} | {f(r['v4'])} | {s['encode']:.1f}x / {s['decode']:.1f}x / **{s['roundtrip']:.1f}x** | {r['rtf']:,.0f}x | "
                     f"{'yes' if r['codes_identical'] else f'no ({r[chr(99)+chr(111)+chr(100)+chr(101)+chr(115)+chr(95)+chr(100)+chr(105)+chr(102)+chr(102)+chr(101)+chr(114)+chr(105)+chr(110)+chr(103)]}/{r[chr(99)+chr(111)+chr(100)+chr(101)+chr(115)+chr(95)+chr(116)+chr(111)+chr(116)+chr(97)+chr(108)]})'} | {'yes' if r['audio_within_tolerance'] else 'no'} ({r['audio_max_abs_diff']:.1e}) |")
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def chart(rows: list[dict], path: str, gpu: str) -> None:
    """One panel: encode + decode latency vs audio length, transformers vs v4, speedup labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    secs = [r["seconds"] for r in rows]
    tr = [r["transformers"]["roundtrip_ms"] for r in rows]
    fx = [r["v4"]["roundtrip_ms"] for r in rows]
    ax.plot(secs, tr, "o-", color="#8a94a6", label="transformers MimiModel (fp32, TF32 off)")
    ax.plot(secs, fx, "s-", color="#1f6feb", label="fast-mimi v4 (fp32, identical codes)")
    for s_, a, b in zip(secs, tr, fx):
        ax.annotate(f"{a / b:.1f}x", (s_, b), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#1f6feb")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("audio length (s)"); ax.set_ylabel("encode + decode latency (ms), batch 1")
    ax.set_title(f"fast-mimi v4 vs transformers - {gpu.split(',')[0]}", fontsize=11)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(secs); ax.set_xticklabels([f"{s_:g}" for s_ in secs])
    fig.tight_layout()
    fig.savefig(path, dpi=150)   # format from the extension (.svg for the repo asset, .png for a quick look)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 2, 5, 10, 25, 50, 100])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--long-repeats", type=int, default=20, help="repeats for inputs >= 25 s")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ramp", type=float, default=0.5)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--chart", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20260826)
    from fast_mimi.v4 import build, load_reference
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ref = load_reference()
    fast = build(log=lambda m: print(m, file=sys.stderr))
    rows = []
    for s in a.seconds:
        rep = a.repeats if s < 25 else a.long_repeats
        row = bench_one(ref, fast, s, rep, a.warmup, a.ramp)
        rows.append(row)
        print(json.dumps(row), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    if a.out_json:
        with open(a.out_json, "w") as fh:
            json.dump({"gpu": gpu, "date": str(datetime.date.today()), "torch": torch.__version__, "rows": rows}, fh, indent=1)
    print(write_markdown(rows, a.out_md, gpu))
    if a.chart:
        chart(rows, a.chart, gpu)
        print("chart:", a.chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
