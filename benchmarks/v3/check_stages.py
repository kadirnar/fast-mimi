"""Stage-by-stage comparison of the Triton backend against the fp32 reference (TF32 off)."""
import sys, torch, math
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from fast_mimi.v3 import FastMimi, load_mimi_state
from fast_mimi.v3.model import causal_conv1d, causal_conv_transpose1d
from fast_mimi.v3.triton_backend import TritonMimi
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
state = load_mimi_state("kyutai/mimi")
ref = FastMimi(state, dtype=torch.float32)
tm = TritonMimi(state)

def snr(a, b):
    a = a.float(); b = b.float()
    return round(10 * math.log10(a.pow(2).mean().item() / ((a - b).pow(2).mean().item() + 1e-30)), 1)

n = 24000
g = torch.Generator(device="cpu").manual_seed(20260826)
x = torch.randn(1, 1, n, generator=g).cuda()
with torch.inference_mode():
    # ---------------- encode
    r1 = ref._seanet(x, ref.enc_layers)                       # [1, 512, 25]
    o1 = tm.enc_seanet(x)                                      # [25, 512]
    print("enc seanet     snr", snr(r1[0].T, o1))
    r2 = ref._transformer(r1.transpose(1, 2), ref.enc_tf)      # [1, 25, 512]
    xb = r1[0].T.contiguous().clone(); tm.enc_tf.forward(xb)
    print("enc transformer snr", snr(r2[0], xb))
    r3 = causal_conv1d(r2.transpose(1, 2), ref.down_w, None, stride=2, pad_mode="replicate")   # [1, 512, 13]
    e = r3[0].T
    r_xs = e @ ref.sem_in.T; r_xa = e @ ref.ac_in.T
    proj = torch.empty(13, 512, device="cuda"); tm.down_proj(r2[0].contiguous(), proj)
    print("down_proj sem   snr", snr(r_xs, proj[:, :256]), " ac snr", snr(r_xa, proj[:, 256:]))
    r_codes = ref._rvq_encode(r3, 32)
    o_codes = tm.rvq.encode(r_xs.contiguous(), r_xa.contiguous(), 1, 13)
    print("rvq (same input) code match", (r_codes == o_codes).float().mean().item())
    full_codes = tm.encode(x)
    print("full encode code match", (r_codes == full_codes).float().mean().item(), "per-codebook", [(r_codes[0, i] == full_codes[0, i]).float().mean().item() for i in range(0, 32, 4)])
    # ---------------- decode
    d1 = ref._rvq_decode(r_codes)                              # [1, 512, 13]
    d1u = causal_conv_transpose1d(d1, ref.up_w, None, stride=2, groups=512)   # [1, 512, 26]
    y = torch.empty(26, 512, device="cuda"); tm.rvq_dec(r_codes[0].contiguous(), y)
    print("rvq_decode_up   snr", snr(d1u[0].T, y))
    d2 = ref._transformer(d1u.transpose(1, 2), ref.dec_tf)     # [1, 26, 512]
    yb = d1u[0].T.contiguous().clone(); tm.dec_tf.forward(yb)
    print("dec transformer snr", snr(d2[0], yb))
    d3 = ref._seanet(d2.transpose(1, 2), ref.dec_layers)[0, 0]  # [L]
    o3 = tm.dec_seanet(d2[0].contiguous())
    print("dec seanet      snr", snr(d3, o3[: d3.shape[0]]))
    full = tm.decode(r_codes, length=n)[0, 0]
    print("full decode     snr", snr(d3[:n], full[:n]))
