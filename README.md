![fast-mimi](assets/fast-mimi-banner.png)

# Fast-Mimi

Fast-Mimi is an independent PyTorch inference runtime for the frozen
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural audio codec. Production
code under `src/fast_mimi` does not import or depend on `transformers`.
The original checkpoint is loaded directly with `huggingface-hub` and
`safetensors`.

The default portable path is pure PyTorch. The guarded RTX 5070 Ti/SM120 path
adds Inductor graphs, CUDA Graph replay, Triton memory kernels, cuDNN frontend
plan selection, and a native CUTLASS decoder-tail kernel. Unsupported shapes,
versions, devices, or toolchains fail closed to the portable implementation.

The declared model identity is locked to:

- Model: `kyutai/mimi`
- Revision: `89091b3e466eb6a9d11e537bf26b144f194978f7`
- Weights SHA-256: `bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e`
- Parameter fingerprint: `3feaa6168b191ffdebfd8f695b963f72c8d847a3966f7cc3283af6b38d437bb4`
- Parameter count: `79,308,609`

## Optimization results

RTX 5070 Ti, SM120, 24 kHz mono, eight codebooks, exactly 100 seconds
(2,400,000 samples). The first execution and all compilation/autotuning are
excluded. Accepted core statistics use 50 alternating paired samples and
10,000 bootstrap resamples. Every row below is an end-to-end result or states
why a failed component/correctness gate made an end-to-end number invalid. The
package contains only the independent PyTorch runtime and its optimized
PyTorch, Triton, cuDNN, CUDA Graph, and CUTLASS paths; it contains no production
Transformers import.

| Technique / path | Precision or backend | 100 s end-to-end median | Measured acceleration | Quality gate and decision |
|---|---|---:|---:|---|
| Independent scratch PyTorch | FP32 | 135.241 ms | 1.0000x | Frozen reference baseline |
| Previous Fast-Mimi release | FP32 | 87.091 ms | 1.5529x | Superseded |
| Compiled graphs, CUDA Graphs, local-window path, selected Triton/cuDNN bundle | FP32 | 65.848 ms | 2.0538x | Accepted intermediate |
| Monolithic Inductor bottleneck, including RVQ `cdist`/`argmin` | FP32 | 61.851 ms | 2.1866x | Rejected: seed 1103 crossed two nearest-code boundaries |
| Quality-safe split bottleneck; RVQ selection kept outside Inductor | FP32 | **62.235 ms** | **2.1731x; 95% CI 2.1718x–2.1744x** | **Accepted core; exact codes, 20/20 long-audio seeds passed** |
| Current public API, including caller-owned output clones | FP32 tensors; Triton/cuDNN/CUTLASS | **62.240 ms** | **2.1729x** | **Accepted; native CUTLASS loaded, zero code or audio-tolerance violations** |
| Fastest cuDNN plans without quality ranking | FP32 | 61.836 ms | 2.1871x | Rejected: 31 audio-tolerance violations |
| cuDNN attention in both attention stacks | FP16 | 58.624 ms | 2.3069x | Rejected: code and waveform quality failed |
| cuDNN attention in both attention stacks | BF16 | 58.653 ms | 2.3058x | Rejected: code and waveform quality failed |
| Corrected-accumulation cuDNN attention | FP32 | 59.452 ms | 2.2748x | Rejected: code/audio violations remained |
| Decoder-only cuDNN attention | FP32 | 60.427 ms | 2.2381x | Rejected: hundreds of audio violations |
| Global TF32 attention/MLP GEMMs | TF32 | 57.435 ms | 2.3547x | Rejected: 15–30 code mismatches and 358k–434k audio violations |
| Decoder-only TF32 attention/MLP GEMMs | TF32 | 59.648 ms | 2.2673x | Rejected: codes exact, but 10k–17k audio violations |
| Selective TF32 QKV/output/FC1/FC2 groups | TF32 | Not eligible: accuracy gate failed | 1.0023x–1.0273x incremental | Rejected: every grouping exceeded waveform tolerance |
| CuTe DSL, all six GEMMs in attention layer 7 | FP16 | 61.490 ms | 2.1994x; 1.0062x incremental | Rejected: 13,187 audio violations and <1% gain |
| CuTe DSL, all six GEMMs in attention layer 7 | BF16 | 61.492 ms | 2.1993x; 1.0065x incremental | Rejected: 335,855 audio violations |
| Three-term compensated CuTe DSL | FP16 with FP32 compensation | 62.033 ms | 2.1801x; 0.9974x incremental | Rejected: slower and 1,974 violations |
| Decoder residual block 6, both convolutions | FP16 | 60.875 ms | 2.2216x | Rejected: audio violations |
| Decoder residual block 6, first convolution | FP16 | 61.425 ms | 2.2017x | Rejected: audio violations |
| Decoder residual block 6, final convolution | FP16 | 61.950 ms | 2.1831x; 1.00015x incremental | Rejected alone: gain below 1% |
| Decoder residual block 6, first convolution | BF16 | 61.490 ms | 2.1994x | Rejected: 189,135–338,842 violations |
| Bit-exact QKV projection fusion | FP32 | 61.014 ms | 2.2166x; 1.0031x incremental | Rejected alone: gain below 1% |
| QKV fusion plus Triton scaled residual | FP32 | 62.828 ms | 2.1526x; 0.9948x incremental | Rejected: slower |
| Encoder NHWC/channels-last chain | FP32 | 1,388.750 ms | 0.0974x | Rejected: slower and incorrect |
| Full custom Triton encoder, decoder, attention, RoPE, RVQ, and norms | FP32 | 555.001 ms | 0.2437x | Rejected: 16 code and 492,355 audio violations |
| Native CUTLASS decoder layer 11, 22 SM120 variants | FP16 input, FP32 accumulation/output | Included in 62.235 ms core | 1.0645x over the accepted 65.848 ms bundle | Accepted with quality-ranked cuDNN recovery |
| cuDNN multi-MMA residual-branch graph | FP32 | Not eligible: cuDNN frontend returned no engine | — | Deferred; fixed-pointer CUDA Graph fallback improved full decoder only 17.2471→17.2045 ms (1.0025x) |
| Encoder suffix/stage-4 approximate fusion | FP32 | Not eligible: correctness gate failed | Encoder 26.631→22.122 ms (1.204x) | Rejected: 22 code mismatches; exact recovery fell to 1.036x encoder gain |
| Encoder stage-7 fusion | FP32 | Not eligible: correctness gate failed | Stage 4.731→4.576 ms; full encoder 26.852 ms | Rejected: seven code mismatches and no encoder-level gain |
| Encoder stage-10 fusion | FP32 | Not eligible: correctness gate failed | Stage 2.488→2.420 ms; full encoder 26.607 ms | Rejected: six code mismatches and no encoder-level gain |
| Alternative local-attention geometries (32/256 through 250/250) | FP32 | Not eligible: correctness gate failed | No credible gain | Rejected: seed 1103 failed; most variants were slower |
| TileLang Q projection | FP32 | Not eligible: component correctness failed (0.0473 ms) | 1.574x component | Rejected: max error 0.02073 |
| TileLang FC1 | FP32 | Not eligible: component correctness failed (0.1399 ms) | 1.750x component | Rejected: max error 0.01163 |
| ModelOpt encoder | FP8 | Not eligible: component correctness failed (81.796 ms) | 0.375x vs compiled FP32 | Rejected: slower and 3,015 code mismatches |
| HQQ plus GemLite | INT8 weights | Not eligible: component correctness failed (0.2214 ms) | 0.919x component | Rejected: slower and max error 0.1867 |
| HQQ plus GemLite | INT4 weights | Not eligible: component correctness failed (0.2370 ms) | 0.859x component | Rejected: slower and max error 2.016 |
| TensorRT encoder, best tested configuration | FP32 accumulation | Not eligible: code gate failed (20.901 ms encoder) | 1.467x encoder | Rejected: code mismatches on every tested seed |
| TensorRT decoder, best tested configuration | FP32 | Not eligible: component gate failed (154.457 ms decoder) | 0.111x decoder | Rejected: about 9x slower and quality failed |
| TF32 and 3xTF32 GEMM variants | Mixed | Not eligible: component correctness failed | Faster GEMMs | Rejected: code mismatches |
| Segmented cuDNN attention | FP32 | Not eligible: component gate failed (0.270 ms) | 0.937x component | Rejected: slower with the same error |
| FP16 channel-recovery search | Mixed FP16/FP32 | Not eligible: component correctness failed | At least 112/128 channels had to remain FP32 | Rejected: recovery erased the gain |
| Mixed FP32-input/FP16-weight cuDNN graph | Mixed | No executable engine | — | Deferred to a future SM120 cuDNN backend |
| ModelOpt/TensorRT-LLM FP8 export | FP8 | Export failed | — | Deferred to a CUDA 12-compatible target; eager FP8 already failed quality |
| Nsight Compute hardware counters | SM120 | `ERR_NVGPUCTRPERM` | — | Deferred to an administrator-enabled profiling target |

The current accepted path keeps all 79,308,609 parameters and all eight requested
codebooks. Long-form output codes are exact. Across 20 frozen 100-second seeds
there were zero audio-tolerance violations; the worst tolerance ratio was
`0.886597`, with maximum absolute error `0.000197306`. It is intentionally not described as bitwise waveform equality,
because quality-safe mixed execution can differ within the frozen
`atol=2e-4, rtol=1e-4` contract.

## Install

Portable PyTorch runtime:

```bash
pip install "fast-mimi @ git+https://github.com/kadirnar/fast-mimi.git"
```

RTX 5070 Ti/SM120 optimized runtime:

```bash
pip install "fast-mimi[optimized] @ git+https://github.com/kadirnar/fast-mimi.git"
```

The optimized extra installs the uv-distributed CUDA 13 `nvcc` and CCCL
packages; TileLang supplies the CUTLASS header tree. If Triton, cuDNN frontend,
TileLang, `nvcc`, CUDA 13, SM120, or the validated shape contract is
unavailable, Fast-Mimi uses the portable path. Set
`FAST_MIMI_DISABLE_OPTIMIZED_LONG=1` to force that fallback.

## Quick start

```python
import torch

from fast_mimi import MimiModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MimiModel.from_pretrained(device=device)

audio = torch.randn(1, 1, 2_400_000, device=device) * 0.05
padding_mask = torch.ones_like(audio, dtype=torch.bool)

with torch.inference_mode():
    output = model(audio, padding_mask=padding_mask, num_quantizers=8)

print(output.audio_codes.shape)
print(output.audio_values.shape)
```

Short clips, arbitrary supported shapes, streaming calls, CPU execution, and
non-SM120 GPUs continue to use the independent PyTorch implementation.

## Reproducing the accepted contract

The validated optimized path requires PyTorch `2.13.0+cu130`, Triton `3.7.1`,
cuDNN frontend `1.27.0`, TileLang `0.1.13`, CUDA runtime 13.0, the uv CUDA 13.3
compiler/CCCL packages, Linux, SM120,
`torch.backends.cuda.matmul.allow_tf32 == False`, and
`torch.backends.cudnn.allow_tf32 == True`. These are dispatch guards, not
silent global-setting mutations.

The CUTLASS implementation follows NVIDIA's
[SM120 functionality](https://docs.nvidia.com/cutlass/latest/overview.html),
[CuTe DSL](https://docs.nvidia.com/cutlass/latest/index.html), and
[autotuning](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/autotuning_gemm.html)
documentation.

## License

Fast-Mimi is licensed under [Apache-2.0](LICENSE). The `kyutai/mimi` weights
are distributed separately under CC-BY-4.0. Mimi was introduced in Kyutai's
[Moshi repository](https://github.com/kyutai-labs/moshi) and
[Moshi paper](https://arxiv.org/abs/2410.00037).
