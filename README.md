# Fast-Mimi

Fast inference for [Mimi](https://huggingface.co/kyutai/mimi), Kyutai's neural audio codec. Drop-in for `transformers.MimiModel`: same API, same output types, fused Triton/CUDA kernels and CUDA graphs underneath, model weights unchanged.

| precision | | Speedup @ 1 s | Codes vs `transformers` |
|---|---|:---:|---|
| **FP32** | exact kernels, no tensor cores | **15.4x** | bit-identical |
| **FP16** | tensor cores | **23.5x** | ~75-82% identical, same reconstruction quality |
| **BF16** | the same kernels, wider exponent | **23.5x** | ~75-82% identical |

## Quick Start

```bash
pip install "fast-mimi[fp32,fp16] @ git+https://github.com/kadirnar/fast-mimi.git"
```

```python
import torch
from transformers import MimiModel
import fast_mimi

model = MimiModel.from_pretrained("kyutai/mimi", dtype=torch.float32).cuda().eval()
fast_mimi.optimize(model)                        # FP32 - exact
# fast_mimi.optimize(model, dtype="fp16")        # FP16 - fastest
# fast_mimi.optimize(model, dtype="bf16")        # BF16 - same speed, wider exponent

audio = torch.randn(1, 1, 24000, device="cuda")  # [batch, channels, samples] @ 24 kHz
codes = model.encode(audio).audio_codes          # [1, 32, frames] int64
wave = model.decode(codes).audio_values          # [1, 1, samples] fp32
```

`optimize` patches the model in place and returns it. `encode` / `decode` keep the transformers signatures and return `MimiEncoderOutput` / `MimiDecoderOutput`, so `padding_mask` still works and still truncates the decoded waveform:

```python
mask = torch.ones_like(audio, dtype=torch.bool)
wave = model.decode(model.encode(audio, mask).audio_codes, mask).audio_values
```

Anything the fast path cannot serve falls back to the stock implementation instead of failing. The first call for a new input length compiles the kernels and captures a CUDA graph; later calls reuse it.

## Benchmark

RTX 5070 Ti | `kyutai/mimi` (96.2M params) | encode + decode, batch 1, 24 kHz mono, 32 codebooks | one session, median of 50 runs

| Audio | transformers | **FP32** | Speedup | **FP16** | Speedup |
|-------|:------------:|:--------:|:-------:|:--------:|:-------:|
| 1 s | 18.16 ms | **1.18 ms** | **15.4x** | **0.77 ms** | **23.5x** |
| 2 s | 18.69 ms | **2.06 ms** | **9.1x** | **1.11 ms** | **16.9x** |
| 5 s | 20.31 ms | **3.87 ms** | **5.2x** | **1.51 ms** | **13.4x** |
| 10 s | 26.04 ms | **6.86 ms** | **3.8x** | **2.45 ms** | **10.6x** |
| 25 s | 42.78 ms | **16.43 ms** | **2.6x** | **5.46 ms** | **7.8x** |
| 50 s | 77.25 ms | **31.17 ms** | **2.5x** | **10.46 ms** | **7.4x** |
| 100 s | 151.89 ms | **60.99 ms** | **2.5x** | **20.07 ms** | **7.6x** |

FP32 keeps the discrete codes bit-identical to the fp32 reference and the waveform within `rtol 2e-4 / atol 2e-5`, deterministically. FP16 and BF16 share every kernel and measure the same; pick BF16 only if you want the wider exponent range.

## Hugging Face `kernels`

The convolution kernel is also registered for [`kernels`](https://github.com/huggingface/kernels), so the standard entry point works:

```python
from kernels import Mode, kernelize
import fast_mimi.hub_kernels

fast_mimi.hub_kernels.register()
model = kernelize(model, mode=Mode.INFERENCE, device="cuda")
```

This is the per-layer path: it keeps the stock control flow and gives up what needs a whole-model view, so it is **1.18x** where `optimize` is 15.3x. Use it when you want `kernelize` to be the single entry point for every model in a pipeline; use `optimize` when you want the speed.

## Requirements

- PyTorch 2.13 (CUDA 13), Triton 3.7+, `transformers >= 5.14`
- FP32 also needs an nvcc (the `fp32` extra installs one)
- NVIDIA GPU (measured on RTX 5070 Ti, Blackwell)

## License

Apache 2.0
