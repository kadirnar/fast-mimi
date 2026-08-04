![fast-mimi](assets/fast-mimi-banner.png)

# Fast-Mimi

Fast-Mimi is an exact float32 inference runtime for
[Kyutai Mimi](https://huggingface.co/kyutai/mimi), a neural audio codec for
high-quality, low-bitrate audio. It loads the original checkpoint directly and
accelerates inference with PyTorch compilation, CUDA Graphs, CUDA C++, and
CUTLASS while preserving checkpoint keys, encoded tokens, and decoded
waveforms bit for bit.

No quantization, reduced precision, TF32, approximate math, pruning,
distillation, or model-weight changes are used.

## Benchmark

NVIDIA RTX 5070 Ti | `kyutai/mimi` | FP32 | 24 kHz mono

### Full Precision (FP32) — Bitwise Exact

| Workload | PyTorch baseline | Fast-Mimi | Speedup |
|---|---:|---:|---:|
| 1-second Q32 round trip | 17.159 ms | 4.466 ms | 3.84x |
| 100-second Q8 round trip | 149.077 ms | 87.091 ms | 1.71x |
| 100-second Q32 round trip | 152.412 ms | 90.475 ms | 1.68x |

Values are steady-state medians measured on the same machine and workload.
Actual performance depends on the GPU, PyTorch version, input shape, and
available CUDA toolchain.

## Optimizations

| Area | Exact acceleration path |
|---|---|
| RVQ encode | Fixed-shape `torch.compile` and Inductor graph |
| RVQ decode | Ordered float32 CUDA C++ gather/add kernel |
| Convolutions | Compiled encoder/decoder graphs and causal cuDNN paths |
| Attention | Native CUTLASS causal-window attention |
| Round trip | CUDA Graph replay for supported one-second Q32 inputs |
| Streaming | Bounded KV storage with exact cumulative positions |

Triton remains active inside the accepted Inductor graphs. Standalone Triton,
TileLang, and CuTe prototypes were parity-correct but are not dispatched because
they did not improve end-to-end latency. Every specialized path has capability
and shape guards with a pure-PyTorch fallback.

## Quick Start

```bash
pip install git+https://github.com/kadirnar/fast-mimi.git
```

```python
import torch

from fast_mimi import MimiModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MimiModel.from_pretrained(device=device)

audio = torch.randn(1, 1, 24_000, device=device) * 0.05
padding_mask = torch.ones_like(audio, dtype=torch.bool)

with torch.inference_mode():
    codes = model.encode(audio, num_quantizers=32).audio_codes
    reconstructed = model.decode(
        codes,
        padding_mask=padding_mask,
    ).audio_values

print(codes.shape)          # [1, 32, 13]
print(reconstructed.shape)  # [1, 1, 24000]
```

`model(audio)` performs the same encode/decode round trip. Streaming encoding
uses 1,920-sample chunks and reuses the returned `encoder_past_key_values` and
`padding_cache` on the next call.

## Requirements

- Python 3.10+
- PyTorch 2.5+
- `huggingface-hub` 0.28+
- `safetensors` 0.4+
- An NVIDIA GPU for CUDA acceleration; CPU execution uses PyTorch fallbacks

## License

Fast-Mimi is licensed under [Apache-2.0](LICENSE). The `kyutai/mimi` weights are
distributed separately under CC-BY-4.0. Mimi was introduced in Kyutai's
[Moshi repository](https://github.com/kyutai-labs/moshi) and
[Moshi paper](https://arxiv.org/abs/2410.00037).
