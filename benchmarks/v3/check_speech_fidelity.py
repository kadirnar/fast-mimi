"""Fidelity on real speech: transformers fp32 reference vs fast-mimi, both compared to the original input with
waveform SNR and a log-mel spectral distance (the metric that matters for a neural codec)."""
import sys, glob, json, math, torch, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import soundfile as sf
from scipy.signal import resample_poly
from fast_mimi.v3 import load_mimi_state
from fast_mimi.v3.backends import build

torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
if len(sys.argv) < 2 or not sys.argv[1]:
    sys.exit("usage: check_speech_fidelity.py '<glob of wav files>' [backend] [max_seconds]")
files = sorted(glob.glob(sys.argv[1]))[:4]
backend = sys.argv[2] if len(sys.argv) > 2 else "triton"
max_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0


def load24k(path, max_sec):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != 24000:
        g = math.gcd(sr, 24000)
        wav = resample_poly(wav, 24000 // g, sr // g).astype(np.float32)
    wav = wav[: int(24000 * max_sec)]
    return torch.from_numpy(np.ascontiguousarray(wav))


def mel_fb(n_mels=80, n_fft=1024, sr=24000, device="cuda"):
    f = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=device)
    mel = lambda hz: 2595 * torch.log10(1 + hz / 700)
    m_pts = torch.linspace(mel(torch.tensor(0.0)), mel(torch.tensor(sr / 2.0)), n_mels + 2, device=device)
    hz_pts = 700 * (10 ** (m_pts / 2595) - 1)
    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for i in range(n_mels):
        lo, c, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        fb[i] = torch.clamp(torch.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), min=0)
    return fb


FB = None
def logmel(x, n_fft=1024, hop=256):
    global FB
    if FB is None:
        FB = mel_fb(device=x.device)
    spec = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft, device=x.device), return_complex=True).abs() ** 2
    return torch.log10(FB @ spec + 1e-8)


def snr(x, y):
    n = min(x.shape[-1], y.shape[-1]); x, y = x[..., :n], y[..., :n]
    return 10 * torch.log10(x.pow(2).mean() / ((x - y).pow(2).mean() + 1e-20)).item()


def main():
    from transformers import MimiModel
    ref = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
    state = load_mimi_state("kyutai/mimi")
    fast = build(backend, state, dtype=torch.bfloat16)
    rows = []
    with torch.inference_mode():
        for path in files:
            x = load24k(path, max_sec).cuda()
            n = x.shape[0]
            xin = x.view(1, 1, n)
            mask = torch.ones_like(xin, dtype=torch.bool)
            rc = ref.encode(xin, mask).audio_codes
            ra = ref.decode(rc, mask).audio_values[0, 0, :n]
            fc = fast.encode(xin).clone()
            fa = fast.decode(fc, length=n).clone()[0, 0, :n]
            fa_refcodes = fast.decode(rc, length=n).clone()[0, 0, :n]
            lm_x, lm_r, lm_f = logmel(x), logmel(ra), logmel(fa)
            rows.append(dict(file=path.split("/")[-1], seconds=round(n / 24000, 2), code_match=round((rc == fc).float().mean().item(), 4),
                             snr_same_codes_db=round(snr(ra, fa_refcodes), 1),
                             ref_vs_input_snr_db=round(snr(x, ra), 2), fast_vs_input_snr_db=round(snr(x, fa), 2),
                             ref_logmel_dist=round((lm_x - lm_r).abs().mean().item(), 4), fast_logmel_dist=round((lm_x - lm_f).abs().mean().item(), 4),
                             ref_vs_fast_logmel_dist=round((lm_r - lm_f).abs().mean().item(), 4)))
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
