![fast-mimi](assets/fast-mimi-banner.png)

# Fast-Mimi

Fast-Mimi is a Transformers-free PyTorch inference runtime for the frozen
[`kyutai/mimi`](https://huggingface.co/kyutai/mimi) checkpoint. It provides a
portable PyTorch path and a guarded RTX 5070 Ti (SM120) path using Inductor,
Triton, cuDNN, CUDA Graphs, and native CUDA/CUTLASS kernels.

[![Optimization comparison](assets/optimization-progress.svg)](docs/optimization-comparison.html)

## Benchmarks

Measurements use FP32 audio, 24 kHz mono input, eight codebooks, an RTX 5070 Ti,
and exclude compilation and autotuning.

### Matched real-speech comparison

All implementations received the same five-second, hash-pinned LibriSpeech
tensor and the same checkpoint revision.

| Implementation | Median latency | Speedup vs. 15.508 ms Transformers baseline |
|---|---:|---:|
| Transformers reference | 15.508 ms | 1.0000x |
| fast-kernel native | 10.320 ms | 1.5028x |
| Fast-Mimi functional FP32 | 10.010 ms | 1.5492x |
| Fast-Mimi guarded SM120 | **5.675 ms** | **2.7327x** |

The cross-project values are cross-session medians. In 50 alternating pairs,
fast-kernel measured `1.5012x` against its Transformers reference. Fast-Mimi
measured `1.7786x` against its functional FP32 reference, with a 95% paired
bootstrap interval of `1.7581x–1.7992x`.

The recording is LibriSpeech `1272-128104-0004`, SHA-256
`07244790e9a8300bfcbf12c28ac5230792e75238d03b2ac167a72bf3943c5404`.
The five-second input is an exact prefix of the ten-second input.

| Real audio | PyTorch baseline | Fast-Mimi | Paired speedup vs. same-row baseline |
|---:|---:|---:|---:|
| 5 s | 10.010 ms | 5.675 ms | 1.7786x |
| 10 s | 12.961 ms | 9.587 ms | 1.3550x |

Codes and decoded waveforms were byte-identical on this fixed real-speech
input.

### Historical 100-second gate

This older gate used a deterministic generated waveform and is not part of the
real-speech comparison above.

| Result | Median latency | Paired speedup vs. 133.561 ms PyTorch baseline |
|---|---:|---:|
| PyTorch reference | 133.561 ms | 1.0000x |
| Fast-Mimi | 58.916 ms | 2.2669x |
| 5x target | ≤26.712 ms | 5.0000x |

The 95% paired bootstrap interval was `2.2638x–2.2852x`. The frozen quality
gate passed 20/20 inputs with exact code streams and no waveform-tolerance
violations.

## Installation

Portable runtime:

```bash
pip install "fast-mimi @ git+https://github.com/kadirnar/fast-mimi.git"
```

SM120 optimized runtime:

```bash
pip install "fast-mimi[optimized] @ git+https://github.com/kadirnar/fast-mimi.git"
```

## Usage

```python
import torch
from fast_mimi import MimiModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MimiModel.from_pretrained(device=device)

# audio: FP32 mono tensor shaped [batch, 1, samples] at 24 kHz
audio = audio.to(device)
mask = torch.ones_like(audio, dtype=torch.bool)
output = model(audio, padding_mask=mask, num_quantizers=8)
```

Unsupported devices, shapes, and toolchains use the portable PyTorch fallback.

## License

Fast-Mimi is licensed under [Apache-2.0](LICENSE). The Mimi weights are
distributed separately under CC-BY-4.0.
