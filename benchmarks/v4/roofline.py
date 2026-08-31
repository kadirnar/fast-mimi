"""Measured roofline for Mimi encode + decode: what the hardware allows, and how close v4 is.

    python benchmarks/v4/roofline.py

Reports, for this GPU: the achievable DRAM read bandwidth and fp32 (TF32 off) GEMM throughput; the fp32 weight
bytes one encode + decode has to touch; and the FLOPs per audio length, counted from the real module output
shapes.  From those it derives the two floors a bit-preserving fp32 implementation cannot go below --
weights / bandwidth and FLOPs / throughput -- and the resulting maximum speedup over `transformers`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch


FOOTER = """
## Reading this table

Both floors are hard. The first is the time to stream every fp32 weight through the memory system exactly once;
the second is the time to issue the model's FLOPs at the best fp32 throughput this GPU reaches on a large square
GEMM. A real implementation also pays for everything the floors ignore -- layer norms, ELUs, softmax, the
quantizer argmins, the layout changes between stages, kernel launch and tail effects, and the fact that a
64-channel SEANet convolution cannot be as efficient as an 8192-cube GEMM. The ceiling is an upper bound no
kernel set can reach, not a target.

What follows from it:

- **A 100x round trip is not reachable at any audio length.** At 1 s it would mean 0.175 ms, which is 2.6x less
  than the time it takes to read the weights once. The only way under that floor is to stop reading fp32
  weights -- quantization, or reduced-precision tensor cores -- which is exactly what "identical codes, no
  change in quality" rules out. That is the trade `fast_mimi.v3` makes: bf16 tensor cores, ~24x on 1 s, ~83% of
  the codes identical.
- Short inputs are bandwidth bound, long inputs are compute bound, and the crossover sits around 2 s of audio.
  That is why the speedup falls with length even though the implementation does not get worse.
- The remaining headroom is ~0.7 ms at 1 s and ~30 ms at 100 s, and it lives almost entirely in the SEANet
  convolutions and resnet blocks, which run at 25-40% of the fp32 FMA peak.
"""


def signal(sec: float) -> torch.Tensor:
    n = int(24000 * sec)
    g = torch.Generator(device="cpu").manual_seed(20260826)
    t = torch.arange(n) / 24000
    return (0.1 * torch.randn((1, 1, n), generator=g)
            + (0.5 * torch.sin(2 * math.pi * (200 + 3000 * t / sec) * t) * torch.exp(-t)).view(1, 1, n))


def bench(fn, n: int = 20, ramp: float = 0.4) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()                       # let the memory / SM clocks reach their steady state first
    while time.perf_counter() - t0 < ramp:
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def machine() -> tuple[float, float]:
    """(achievable DRAM read GB/s, achievable fp32 GEMM TFLOP/s)."""
    a = torch.empty((1 << 30) // 4, device="cuda", dtype=torch.float32).normal_()
    bw = a.numel() * 4 / bench(lambda: torch.max(a), 40) / 1e9      # single-pass read-only stream
    del a
    torch.cuda.empty_cache()
    x = torch.randn(8192, 8192, device="cuda")
    y = torch.randn(8192, 8192, device="cuda")
    z = torch.empty_like(x)
    tf = 2 * 8192 ** 3 / bench(lambda: torch.mm(x, y, out=z), 10) / 1e12
    del x, y, z
    torch.cuda.empty_cache()
    return bw, tf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 2, 5, 10, 25, 50, 100])
    ap.add_argument("--results", default="benchmarks/v4/results.json", help="bench_v4.py output, for the measured rows")
    ap.add_argument("--out-md", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import MimiModel

    bw, tf = machine()
    p = torch.cuda.get_device_properties(0)
    model = MimiModel.from_pretrained("kyutai/mimi", dtype=torch.float32).to("cuda").eval()

    params = sum(t.numel() for t in model.parameters()) * 4
    codebooks = sum(t.numel() for n, t in model.named_buffers() if n.endswith("embed_sum")) * 4
    weight_bytes = params + codebooks          # encode reads the encoder half + every codebook, decode the decoder half
    dram_ms = weight_bytes / (bw * 1e9) * 1e3

    macs = [0]

    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if isinstance(mod, torch.nn.Conv1d):
            macs[0] += o.numel() * mod.in_channels // mod.groups * mod.kernel_size[0]
        elif isinstance(mod, torch.nn.ConvTranspose1d):
            macs[0] += inp[0].numel() * mod.out_channels // mod.groups * mod.kernel_size[0]
        elif isinstance(mod, torch.nn.Linear):
            macs[0] += o.numel() * mod.in_features

    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)):
            mod.register_forward_hook(hook)

    try:
        measured = {r["seconds"]: r for r in json.load(open(a.results))["rows"]}
    except (OSError, ValueError, KeyError):
        measured = {}

    rows = []
    with torch.inference_mode():
        for sec in a.seconds:
            x = signal(sec).cuda()
            mask = torch.ones_like(x, dtype=torch.bool)
            macs[0] = 0
            codes = model.encode(x, mask).audio_codes
            model.decode(codes, mask)
            frames = int(codes.shape[-1])
            tt = frames * 2                                          # the transformers run at twice the frame rate
            attn = 2 * 8 * 8 * tt * min(tt, 250) * 64 * 2            # 2 models x 8 layers x 8 heads, QK and PV
            rvq = 32 * frames * 2048 * 258                           # the reference's augmented cdist matmul
            gflop = (2 * macs[0] + 2 * attn + 2 * rvq) / 1e9
            comp_ms = gflop / tf
            floor = max(dram_ms, comp_ms)
            m = measured.get(sec)
            rows.append(dict(seconds=sec, frames=frames, gflop=gflop, dram_ms=dram_ms, compute_ms=comp_ms,
                             floor_ms=floor,
                             transformers_ms=m["transformers"]["roundtrip_ms"] if m else None,
                             v4_ms=m["v4"]["roundtrip_ms"] if m else None))
            del x, mask, codes
            torch.cuda.empty_cache()

    head = [
        f"# What is actually reachable on this GPU ({torch.cuda.get_device_name(0)})", "",
        f"- {p.multi_processor_count} SMs, {p.L2_cache_size / 2**20:.0f} MB L2.",
        f"- Achievable DRAM read bandwidth, measured (1 GiB stream): **{bw:.0f} GB/s**.",
        f"- Achievable fp32 GEMM throughput, measured (8192^3, TF32 off): **{tf:.1f} TFLOP/s** "
        f"(the fp32 FMA peak is ~43.9; fp32 has no tensor-core path, and TF32/BF16 would not be fp32-exact).",
        f"- One encode + decode has to read **{weight_bytes / 1e6:.0f} MB** of fp32 weights "
        f"({params / 1e6:.0f} MB parameters + {codebooks / 1e6:.0f} MB codebooks), so no round trip can finish in "
        f"less than **{dram_ms:.3f} ms**, at any audio length.", "",
        "| audio | frames | GFLOP | weight/bandwidth floor | FLOP/throughput floor | transformers | **ceiling** | v4 today | of ceiling |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        tr, v4 = r["transformers_ms"], r["v4_ms"]
        ceil = f"**{tr / r['floor_ms']:.0f}x**" if tr else "-"
        cur = f"{tr / v4:.1f}x" if tr and v4 else "-"
        frac = f"{(tr / v4) / (tr / r['floor_ms']) * 100:.0f}%" if tr and v4 else "-"
        head.append(f"| {r['seconds']:g} s | {r['frames']} | {r['gflop']:.0f} | {r['dram_ms']:.2f} ms | "
                    f"{r['compute_ms']:.2f} ms | {tr:.2f} ms | {ceil} | {cur} | {frac} |")
    text = "\n".join(head) + "\n" + FOOTER
    print(text)
    if a.out_md:
        with open(a.out_md, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
