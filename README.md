# Fast-Mimi

Fast inference engine for [Mimi](https://huggingface.co/kyutai/mimi), Kyutai's neural audio codec: a SEANet encoder/decoder, two 8-layer transformers and a 32-codebook residual vector quantizer. This library accelerates Mimi encode + decode up to **23.5x** on NVIDIA GPUs through fused Triton/CUDA kernels and CUDA graphs — as a drop-in for `transformers.MimiModel`, with bit-identical codes at FP32 and no changes to model weights.

## Benchmark

NVIDIA RTX 5070 Ti | `kyutai/mimi` (96.2M params) | encode + decode, batch 1, 24 kHz mono, 32 codebooks

### Full Precision (FP32) — Identical Codes

| Audio | transformers | fast-mimi | Speedup | Real-time Factor |
|-------|:------------:|:---------:|:-------:|:----------------:|
| 1 s | 18.16 ms | **1.18 ms** | **15.4x** | 847x |
| 2 s | 18.69 ms | **2.06 ms** | **9.1x** | 971x |
| 5 s | 20.31 ms | **3.87 ms** | **5.2x** | 1,292x |
| 10 s | 26.04 ms | **6.86 ms** | **3.8x** | 1,458x |
| 25 s | 42.78 ms | **16.43 ms** | **2.6x** | 1,522x |
| 50 s | 77.25 ms | **31.17 ms** | **2.5x** | 1,604x |
| 100 s | 151.89 ms | **60.99 ms** | **2.5x** | 1,640x |

### Half Precision (FP16 / BF16)

| Audio | transformers | fast-mimi | Speedup | RTF | SNR vs FP32 |
|-------|:------------:|:---------:|:-------:|:---:|:-----------:|
| 1 s | 18.16 ms | **0.77 ms** | **23.5x** | 1,299x | 14.0 dB |
| 2 s | 18.69 ms | **1.11 ms** | **16.9x** | 1,802x | 16.1 dB |
| 5 s | 20.31 ms | **1.51 ms** | **13.4x** | 3,311x | 10.6 dB |
| 10 s | 26.04 ms | **2.45 ms** | **10.6x** | 4,082x | 6.4 dB |
| 25 s | 42.78 ms | **5.46 ms** | **7.8x** | 4,579x | 6.2 dB |
| 50 s | 77.25 ms | **10.46 ms** | **7.4x** | 4,780x | 6.1 dB |
| 100 s | 151.89 ms | **20.07 ms** | **7.6x** | 4,983x | 4.7 dB |

FP32 keeps the discrete codes bit-identical to the reference and the waveform within `rtol 2e-4 / atol 2e-5`. Half precision keeps ~75-82% of the codes and the same reconstruction quality; FP16 and BF16 share every kernel and measure the same, so pick BF16 only for the wider exponent range.

## Quick Start

```bash
pip install "fast-mimi[fp32,fp16] @ git+https://github.com/kadirnar/fast-mimi.git"
```

```python
import torch
from transformers import MimiModel
import fast_mimi

model = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
audio = torch.randn(1, 1, 24000, device="cuda")

# FP32 — identical codes, ~1.18 ms
fast_mimi.optimize(model, dtype="fp32")
codes = model.encode(audio).audio_codes
wave = model.decode(codes).audio_values

# FP16 — fastest, ~0.77 ms
fast_mimi.optimize(model, dtype="fp16")

# BF16 — ~0.77 ms
fast_mimi.optimize(model, dtype="bf16")
```

`optimize` patches the model in place, so `encode` and `decode` keep their transformers signatures and output types. The first call for a new input length compiles the kernels and captures a CUDA graph; later calls reuse it.

## Requirements

- PyTorch 2.13 (CUDA 13), Triton 3.7+, `transformers` 5.14+
- NVIDIA GPU (measured on Blackwell)

## License

Apache 2.0
