"""Compare FastMimi against the stock transformers MimiModel (the numerical oracle)."""
from __future__ import annotations
import argparse, json, sys, time
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from fast_mimi.fp16 import FastMimi, load_mimi_state


def make_inputs(sec, device):
    n = int(24000 * sec)
    g = torch.Generator(device="cpu").manual_seed(20260826)
    noise = torch.randn((1, 1, n), generator=g)
    t = torch.arange(n) / 24000.0
    sweep = (0.5 * torch.sin(2 * torch.pi * (200 + 3000 * t / max(sec, 1e-6)) * t) * torch.exp(-t)).view(1, 1, n)
    return {"noise": noise.to(device), "sweep": sweep.to(device), "mix": (0.1 * noise + sweep).to(device)}


def compare(ref, fast, x, name):
    with torch.inference_mode():
        mask = torch.ones_like(x, dtype=torch.bool)
        ref_codes = ref.encode(x, mask).audio_codes
        fast_codes = fast.encode(x).clone()
        assert ref_codes.shape == fast_codes.shape, (ref_codes.shape, fast_codes.shape)
        code_match = (ref_codes == fast_codes).float().mean().item()
        per_cb = (ref_codes == fast_codes).float().mean(dim=(0, 2))
        ref_audio = ref.decode(ref_codes, mask).audio_values
        fast_audio = fast.decode(ref_codes, length=x.shape[-1]).clone()
        fast_audio_own = fast.decode(fast_codes, length=x.shape[-1]).clone()
        assert ref_audio.shape == fast_audio.shape, (ref_audio.shape, fast_audio.shape)
        diff = (ref_audio.float() - fast_audio.float()).abs()
        rel = diff.max().item() / (ref_audio.abs().max().item() + 1e-12)
        snr = 10 * torch.log10(ref_audio.float().pow(2).mean() / ((ref_audio.float() - fast_audio_own.float()).pow(2).mean() + 1e-20)).item()
        def snr_vs_input(y):
            return 10 * torch.log10(x.float().pow(2).mean() / ((x.float() - y.float()).pow(2).mean() + 1e-20)).item()
        r = dict(name=name, samples=x.shape[-1], frames=ref_codes.shape[-1], code_match=code_match,
                 worst_codebook_match=per_cb.min().item(), max_abs_diff=diff.max().item(), rel_max_diff=rel,
                 snr_db_same_codes=10 * torch.log10(ref_audio.float().pow(2).mean() / (diff.pow(2).mean() + 1e-20)).item(),
                 snr_db_own_codes=snr,
                 recon_snr_ref=snr_vs_input(ref_audio), recon_snr_fast=snr_vs_input(fast_audio_own))
        print(json.dumps(r), flush=True)
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[1, 5])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sliding-window", action="store_true")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--tf32", default="default", choices=["default", "off", "on"])
    ap.add_argument("--ref-dtype", default="float32")
    args = ap.parse_args()
    if args.tf32 == "off":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    elif args.tf32 == "on":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import MimiModel
    ref = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=getattr(torch, args.ref_dtype)).cuda().eval()
    state = load_mimi_state("kyutai/mimi")
    if args.backend == "torch":
        fast = FastMimi(state, dtype=getattr(torch, args.dtype), sliding_window=args.sliding_window)
    elif args.backend == "transformers":  # reference vs itself (noise floor of the oracle)
        class _Wrap:
            def __init__(self, m): self.m = m
            def encode(self, x): return self.m.encode(x, torch.ones_like(x, dtype=torch.bool)).audio_codes
            def decode(self, c, length=None):
                y = self.m.decode(c).audio_values
                return y[..., :length] if length else y
        tf32 = args.tf32 != "off"
        torch.backends.cudnn.allow_tf32 = not tf32
        torch.backends.cuda.matmul.allow_tf32 = not tf32
        m2 = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=getattr(torch, args.dtype)).cuda().eval()
        fast = _Wrap(m2)
    else:
        from fast_mimi.fp16.backends import build
        fast = build(args.backend, state, dtype=getattr(torch, args.dtype))
    ok = True
    for sec in args.seconds:
        for name, x in make_inputs(sec, "cuda").items():
            r = compare(ref, fast, x, f"{name}-{sec}s")
            if args.dtype == "float32" and (r["code_match"] < 0.999 or r["rel_max_diff"] > 1e-3):
                ok = False
    print("OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
