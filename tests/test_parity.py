"""fp32-exact backend: identical codes and in-tolerance audio vs the stock transformers model (needs CUDA + weights)."""
from __future__ import annotations

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytestmark = [pytest.mark.integration, pytest.mark.cuda]

SR = 24_000


def _signal(seconds: float, seed: int, batch: int = 1, samples: int | None = None) -> torch.Tensor:
    n = samples or int(SR * seconds)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((batch, 1, n), generator=g)
    t = torch.arange(n) / SR
    sweep = (0.5 * torch.sin(2 * math.pi * (200 + 3000 * t / max(seconds, 1e-6)) * t) * torch.exp(-t)).view(1, 1, n)
    return 0.1 * noise + sweep


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seconds,batch,samples", [(1.0, 1, None), (0.5, 2, None), (1.0, 1, SR + 123), (0.05, 1, None)])
def test_v4_matches_reference(seconds, batch, samples):
    from fast_mimi.fp32 import build, load_reference
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = load_reference()
    fast = build()
    x = _signal(seconds, 20260826, batch, samples).cuda()
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.inference_mode():
        rc = ref.encode(x, mask).audio_codes
        ra = ref.decode(rc, mask).audio_values
        fc = fast.encode(x, mask).audio_codes
        fa = fast.decode(fc, mask).audio_values
        fc2 = fast.encode(x, mask).audio_codes
    assert torch.equal(rc, fc), "discrete codes must be identical to the fp32 reference"
    assert torch.equal(fc, fc2), "deterministic"
    torch.testing.assert_close(fa, ra, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_v4_long_form_matches_reference():
    """12 s of audio puts the transformers at T = 300, past the CUDA attention's 128-frame limit: the windowed
    Triton kernel and the cuBLAS O projection have to reproduce the reference layer."""
    from fast_mimi.fp32 import build, load_reference
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = load_reference()
    fast = build()
    x = _signal(12.0, 20260826).cuda()
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.inference_mode():
        rc = ref.encode(x, mask).audio_codes
        ra = ref.decode(rc, mask).audio_values
        fc = fast.encode(x, mask).audio_codes
        fa = fast.decode(fc, mask).audio_values
        fa2 = fast.decode(fc, mask).audio_values
    assert torch.equal(rc, fc), "discrete codes must be identical to the fp32 reference"
    assert torch.equal(fa, fa2), "deterministic"
    torch.testing.assert_close(fa, ra, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rvq_long_path_is_the_reference_expression():
    """Past REF_MIN_ROWS the quantizer search must be bit-identical to `transformers`' own cdist / argmin loop."""
    from fast_mimi.fp32._compat import ensure_cuda_home
    ensure_cuda_home()
    from fast_mimi.fp32.kernels.rvq_fp32 import REF_MIN_ROWS, rvq_encode

    torch.backends.cuda.matmul.allow_tf32 = False
    stages, size, dim = 8, 2048, 256
    g = torch.Generator(device="cpu").manual_seed(7)
    cb = (0.1 * torch.randn(stages, size, dim, generator=g)).cuda()
    frames = REF_MIN_ROWS + 37
    x = torch.randn((1, dim, frames), generator=g).cuda()

    residual, want = x, []
    with torch.inference_mode():
        for s in range(stages):
            rows = residual.permute(0, 2, 1).reshape(-1, dim)
            ind = torch.cdist(rows[None].float(), cb[s][None].float(), p=2)[0].argmin(dim=-1)
            want.append(ind.view(1, frames))
            residual = residual - torch.nn.functional.embedding(ind, cb[s]).view(1, frames, dim).permute(0, 2, 1)
        got = rvq_encode(x, cb, stages)
    assert torch.equal(got, torch.stack(want)), "long-form search must match the reference expression exactly"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rvq_tiled_path_matches_the_fused_chain():
    """The register-tiled search (rows >= TILED_MIN_ROWS) keeps the fused chain's 8 x 32 fmaf order: same codes."""
    from fast_mimi.fp32._compat import ensure_cuda_home
    ensure_cuda_home()
    from fast_mimi.fp32.kernels import rvq_fp32 as rvq

    torch.backends.cuda.matmul.allow_tf32 = False
    stages, size, dim = 8, 2048, 256
    g = torch.Generator(device="cpu").manual_seed(11)
    cb = (0.1 * torch.randn(stages, size, dim, generator=g)).cuda().contiguous()
    ref_min, tiled_min = rvq.REF_MIN_ROWS, rvq.TILED_MIN_ROWS
    try:
        for frames in (tiled_min, tiled_min + 51, ref_min - 1):
            x = (0.5 * torch.randn((1, dim, frames), generator=g)).cuda().contiguous()
            with torch.inference_mode():
                rvq.REF_MIN_ROWS, rvq.TILED_MIN_ROWS = 10**9, 10**9
                fused = rvq.rvq_encode(x, cb, stages).clone()
                rvq.TILED_MIN_ROWS = 0
                tiled = rvq.rvq_encode(x, cb, stages).clone()
            assert torch.equal(tiled, fused), f"tiled search must match the fused chain at {frames} rows"
    finally:
        rvq.REF_MIN_ROWS, rvq.TILED_MIN_ROWS = ref_min, tiled_min


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seconds", [1.0, 5.0])
def test_v4_without_padding_mask(seconds):
    """`model.encode(audio)` / `model.decode(codes)` -- the natural transformers call -- must take the graphed path
    and return exactly what the reference returns (encode ignores the mask; decode without one does not truncate)."""
    from fast_mimi.fp32 import build, load_reference, report

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = load_reference()
    fast = build()
    x = _signal(seconds, 20260826).cuda()
    with torch.inference_mode():
        rc = ref.encode(x).audio_codes
        ra = ref.decode(rc).audio_values
        before = report()["invocations"]
        fc = fast.encode(x).audio_codes
        fa = fast.decode(fc).audio_values
        after = report()["invocations"]
    assert torch.equal(rc, fc), "codes must be identical without a padding mask"
    assert fa.shape == ra.shape, "decode without a mask must not truncate"
    torch.testing.assert_close(fa, ra, rtol=2e-4, atol=2e-5)
    assert after == before + 2, "encode and decode without a mask must both take the graphed path"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
def test_optimize_keeps_the_transformers_api(dtype):
    """`fast_mimi.optimize(model, dtype=...)` patches a MimiModel in place and keeps its API and output types."""
    from transformers import MimiModel
    from transformers.models.mimi.modeling_mimi import (
        MimiDecoderOutput,
        MimiEncoderOutput,
    )

    import fast_mimi

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = MimiModel.from_pretrained("kyutai/mimi", dtype=torch.float32).cuda().eval()
    model = MimiModel.from_pretrained("kyutai/mimi", dtype=torch.float32).cuda().eval()
    assert fast_mimi.optimize(model, dtype=dtype) is model, "optimize patches in place"

    x = _signal(1.0, 20260826).cuda()
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.inference_mode():
        rc = ref.encode(x).audio_codes
        ra = ref.decode(rc).audio_values
        enc, dec = model.encode(x), model.decode(model.encode(x).audio_codes)
        trunc = model.decode(model.encode(x, mask).audio_codes, mask).audio_values
    assert isinstance(enc, MimiEncoderOutput) and isinstance(dec, MimiDecoderOutput)
    assert enc.audio_codes.shape == rc.shape and enc.audio_codes.dtype == rc.dtype
    assert dec.audio_values.shape == ra.shape
    assert trunc.shape[-1] == x.shape[-1], "a padding mask still truncates the waveform"
    if dtype == "fp32":
        assert torch.equal(rc, enc.audio_codes), "fp32 codes must be identical to the reference"
        torch.testing.assert_close(dec.audio_values, ra, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernelize_applies_the_conv_kernel():
    """The convolution kernel is reachable through `kernels.kernelize`, and does not change the output."""
    kernels = pytest.importorskip("kernels")
    from transformers import MimiModel

    from fast_mimi import hub_kernels

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = MimiModel.from_pretrained("kyutai/mimi", dtype=torch.float32).cuda().eval()
    x = _signal(1.0, 20260826).cuda()
    with torch.inference_mode():
        rc = ref.encode(x).audio_codes
        ra = ref.decode(rc).audio_values

    hub_kernels.register()
    model = kernels.kernelize(MimiModel.from_pretrained("kyutai/mimi", dtype=torch.float32).cuda().eval(),
                              mode=kernels.Mode.INFERENCE, device="cuda")
    with torch.inference_mode():
        codes = model.encode(x).audio_codes
        audio = model.decode(codes).audio_values
    assert torch.equal(rc, codes)
    torch.testing.assert_close(audio, ra, rtol=2e-4, atol=2e-5)
