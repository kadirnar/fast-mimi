# Fast-Mimi

Fast inference engine for [Mimi](https://huggingface.co/kyutai/mimi), Kyutai's neural audio codec (SEANet encoder/decoder, two 8-layer transformers, and a 32-codebook residual vector quantizer). This library accelerates Mimi encode + decode up to **15.3x** with bit-identical codes (v4, exact FP32) and up to **23.8x** with bf16 tensor cores (v3) on NVIDIA GPUs, through fused Triton/CUDA kernels and CUDA graphs — with no changes to model weights.

## Benchmark

NVIDIA RTX 5070 Ti | `kyutai/mimi` (96.2M params) | encode + decode, batch 1, 24 kHz mono, 32 codebooks | one session, median of 50 runs

### Full Precision (FP32) — v4, Identical Codes

| Audio | transformers | **fast-mimi v4** | Speedup | Real-time Factor |
|-------|:------------:|:----------------:|:-------:|:----------------:|
| 1 s | 18.17 ms | **1.18 ms** | **15.3x** | 844x |
| 2 s | 18.68 ms | **2.08 ms** | **9.0x** | 960x |
| 5 s | 19.76 ms | **3.89 ms** | **5.1x** | 1,284x |
| 10 s | 24.85 ms | **6.90 ms** | **3.6x** | 1,449x |
| 25 s | 39.77 ms | **16.55 ms** | **2.4x** | 1,510x |
| 50 s | 71.70 ms | **31.37 ms** | **2.3x** | 1,594x |
| 100 s | 140.79 ms | **61.25 ms** | **2.3x** | 1,633x |

### Half Precision (BF16 tensor cores) — v3

| Audio | transformers | **fast-mimi v3** | Speedup | Real-time Factor |
|-------|:------------:|:----------------:|:-------:|:----------------:|
| 1 s | 18.17 ms | **0.76 ms** | **23.8x** | 1,311x |
| 2 s | 18.68 ms | **1.14 ms** | **16.4x** | 1,751x |
| 5 s | 19.76 ms | **1.50 ms** | **13.1x** | 3,324x |
| 10 s | 24.85 ms | **2.45 ms** | **10.2x** | 4,087x |
| 25 s | 39.77 ms | **5.60 ms** | **7.1x** | 4,465x |
| 50 s | 71.70 ms | **10.91 ms** | **6.6x** | 4,582x |
| 100 s | 140.79 ms | **21.58 ms** | **6.5x** | 4,633x |

v4 keeps the discrete codes bit-identical to the fp32 `transformers` model and the waveform within `rtol 2e-4 / atol 2e-5`, deterministically. v3 trades that for bf16 tensor-core speed: equal reconstruction quality, ~80% of the codes identical.

## Quick Start

```bash
pip install "fast-mimi[v3,v4] @ git+https://github.com/kadirnar/fast-mimi.git"
```

```python
import torch

audio = torch.randn(1, 1, 24000, device="cuda")   # [batch, channels, samples] @ 24 kHz

# v4 — exact FP32, identical codes to transformers, ~1.2 ms
from fast_mimi.v4 import build

model = build()                                   # transformers MimiModel + exact kernels
codes = model.encode(audio).audio_codes           # [1, 32, frames] int64
wave = model.decode(codes).audio_values           # [1, 1, samples] fp32

# v3 — BF16 tensor cores, fastest, ~0.8 ms
from fast_mimi.v3 import load_mimi_state, build

codec = build("graph+triton", load_mimi_state("kyutai/mimi"))
codes = codec.encode(audio)
wave = codec.decode(codes, length=audio.shape[-1])
```

The public API matches the Hugging Face model, so `padding_mask` still works and truncates the decoded waveform to the original sample count:

```python
mask = torch.ones_like(audio, dtype=torch.bool)
wave = model.decode(model.encode(audio, mask).audio_codes, mask).audio_values
```

The first call for a new input length compiles the kernels and captures a CUDA graph; later calls reuse it.

## Requirements

- PyTorch 2.13 (CUDA 13) and Triton 3.7+
- `fast_mimi.v4`: `transformers >= 5.14` and an nvcc (the `v4` extra installs one)
- `fast_mimi.v3`: Triton only
- NVIDIA GPU (measured on RTX 5070 Ti, Blackwell)

## License

Apache 2.0
