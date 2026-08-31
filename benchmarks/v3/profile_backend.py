"""Per-kernel GPU profile of the Triton backend encode / decode (no graphs)."""
import sys, torch
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from fast_mimi.v3 import load_mimi_state
from fast_mimi.v3.backends import build
from torch.profiler import profile, ProfilerActivity
backend = sys.argv[1] if len(sys.argv) > 1 else "triton"
sec = float(sys.argv[2]) if len(sys.argv) > 2 else 1
state = load_mimi_state("kyutai/mimi")
m = build(backend, state, dtype=torch.bfloat16)
n = int(24000 * sec)
x = torch.randn(1, 1, n, device="cuda")
with torch.inference_mode():
    codes = m.encode(x); m.decode(codes, length=n); torch.cuda.synchronize()
    for name, fn in [("encode", lambda: m.encode(x)), ("decode", lambda: m.decode(codes, length=n))]:
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        evs = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        print(f"== {name}: {len(evs)} kernels, GPU sum {sum(e.time_range.elapsed_us() for e in evs):.1f} us")
        for e in evs:
            print(f"   {e.name[:58]:58s} {e.time_range.elapsed_us():8.2f}us")
