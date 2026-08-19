"""Benchmark Fast-Mimi with one hash-pinned real recording and paired calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import soundfile
import torch

from fast_mimi import MimiModel
from fast_mimi._functional_runtime import PureTorchMimi


SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1_920
SUPPORTED_SECONDS = (5, 10)
FIXED_CROP_SECONDS = 10
REAL_AUDIO_ID = "1272-128104-0004"
REAL_AUDIO_DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
REAL_AUDIO_DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
REAL_AUDIO_SOURCE_SAMPLE_RATE = 16_000
REAL_AUDIO_SOURCE_FRAMES = 470_400
REAL_AUDIO_SHA256 = "07244790e9a8300bfcbf12c28ac5230792e75238d03b2ac167a72bf3943c5404"


def _default_audio_path() -> Path:
    """Return the shared content-addressed LibriSpeech cache path."""
    cache_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    return (
        cache_home
        / "fast-kernel"
        / "mimi"
        / f"{REAL_AUDIO_ID}-{REAL_AUDIO_SHA256}.flac"
    )


def _sha256(path: Path) -> str:
    """Hash an audio asset without loading it twice into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_real_audio(
    path: Path,
    *,
    samples: int,
    crop_seed: int,
) -> tuple[torch.Tensor, int]:
    """Verify, resample, and select one frame-aligned crop of real speech."""
    if not path.is_file():
        raise RuntimeError(
            f"real audio is missing: {path}; pass --audio-file with the pinned FLAC"
        )
    digest = _sha256(path)
    if digest != REAL_AUDIO_SHA256:
        raise RuntimeError(
            f"real-audio SHA-256 is {digest}, expected {REAL_AUDIO_SHA256}"
        )
    values, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    if sample_rate != REAL_AUDIO_SOURCE_SAMPLE_RATE:
        raise RuntimeError(
            f"real-audio sample rate is {sample_rate}, "
            f"expected {REAL_AUDIO_SOURCE_SAMPLE_RATE}"
        )
    if values.shape != (REAL_AUDIO_SOURCE_FRAMES, 1):
        raise RuntimeError(
            f"real-audio shape is {values.shape}, "
            f"expected {(REAL_AUDIO_SOURCE_FRAMES, 1)}"
        )
    waveform = torch.from_numpy(values[:, 0]).reshape(1, 1, -1)
    output_samples = values.shape[0] * SAMPLE_RATE // sample_rate
    resampled = torch.nn.functional.interpolate(
        waveform,
        size=output_samples,
        mode="linear",
        align_corners=False,
    ).reshape(-1)
    available = resampled.numel() - samples
    if available < 0:
        raise RuntimeError(
            "the pinned real recording is shorter than the requested crop"
        )
    crop_positions = available // FRAME_SAMPLES + 1
    crop_index = crop_seed % crop_positions
    crop_offset = crop_index * FRAME_SAMPLES
    crop = resampled.narrow(0, crop_offset, samples).clone()
    return crop.reshape(1, 1, -1).contiguous(), crop_offset


@dataclass(frozen=True)
class DurationResult:
    """Store one fixed-duration paired benchmark and quality result."""

    audio_seconds: int
    samples: int
    pairs: int
    seed: int
    quantizers: int
    runtime_path: str
    reference_median_ms: float
    candidate_median_ms: float
    speedup: float
    paired_median_speedup: float
    ci95_low: float
    ci95_high: float
    code_mismatches: int
    waveform_violations: int
    waveform_max_abs: float
    waveform_worst_tolerance_ratio: float


def _percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile from non-empty values."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(
    ratios: list[float], *, seed: int, repetitions: int
) -> tuple[float, float]:
    """Bootstrap the median paired speedup with a deterministic RNG."""
    generator = random.Random(seed)
    medians = []
    for _ in range(repetitions):
        sample = [ratios[generator.randrange(len(ratios))] for _ in ratios]
        medians.append(statistics.median(sample))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _synchronized_latency_ms(function: Callable[[], object]) -> float:
    """Measure one complete call, including dispatch and materialized outputs."""
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    function()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / 1_000_000


def _benchmark_duration(
    model: MimiModel,
    reference: PureTorchMimi,
    audio: torch.Tensor,
    *,
    seconds: int,
    pairs: int,
    seed: int,
    quantizers: int,
    bootstrap_repetitions: int,
) -> DurationResult:
    """Benchmark one prefix of the shared fixed waveform."""
    samples = seconds * SAMPLE_RATE
    input_values = audio[..., :samples].contiguous()
    padding_mask = torch.ones_like(input_values, dtype=torch.bool)

    def run_reference():
        return reference.forward(input_values, padding_mask, quantizers)

    def run_candidate():
        return model(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=quantizers,
        )

    with torch.inference_mode():
        reference_output = run_reference()
        run_reference()
        candidate_output = run_candidate()
        candidate_output = run_candidate()
        torch.cuda.synchronize()

        if model._optimized_long_runtime_failed:
            raise RuntimeError(model._optimized_long_runtime_error)

        code_mismatches = int(
            (reference_output.audio_codes != candidate_output.audio_codes)
            .sum()
            .item()
        )
        delta = (reference_output.audio_values - candidate_output.audio_values).abs()
        tolerance = 2e-4 + 1e-4 * reference_output.audio_values.abs()
        waveform_violations = int((delta > tolerance).sum().item())
        waveform_max_abs = float(delta.max().item())
        waveform_worst_tolerance_ratio = float((delta / tolerance).max().item())

        reference_times: list[float] = []
        candidate_times: list[float] = []
        for pair in range(pairs):
            order = (
                ((run_reference, reference_times), (run_candidate, candidate_times))
                if pair % 2 == 0
                else (
                    (run_candidate, candidate_times),
                    (run_reference, reference_times),
                )
            )
            for function, destination in order:
                destination.append(_synchronized_latency_ms(function))

    ratios = [
        reference_ms / candidate_ms
        for reference_ms, candidate_ms in zip(
            reference_times, candidate_times, strict=True
        )
    ]
    ci_low, ci_high = _bootstrap_ci(
        ratios,
        seed=seed + seconds,
        repetitions=bootstrap_repetitions,
    )
    reference_median = statistics.median(reference_times)
    candidate_median = statistics.median(candidate_times)
    return DurationResult(
        audio_seconds=seconds,
        samples=samples,
        pairs=pairs,
        seed=seed,
        quantizers=quantizers,
        runtime_path=(
            "guarded_sm120_optimized"
            if model._optimized_long_runtime is not None
            else "portable_pytorch"
        ),
        reference_median_ms=reference_median,
        candidate_median_ms=candidate_median,
        speedup=reference_median / candidate_median,
        paired_median_speedup=statistics.median(ratios),
        ci95_low=ci_low,
        ci95_high=ci_high,
        code_mismatches=code_mismatches,
        waveform_violations=waveform_violations,
        waveform_max_abs=waveform_max_abs,
        waveform_worst_tolerance_ratio=waveform_worst_tolerance_ratio,
    )


def _parse_args() -> argparse.Namespace:
    """Parse the fixed-input benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-seconds",
        type=int,
        nargs="+",
        choices=SUPPORTED_SECONDS,
        default=list(SUPPORTED_SECONDS),
    )
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=_default_audio_path(),
        help="Hash-pinned LibriSpeech FLAC used as the fixed real input.",
    )
    parser.add_argument("--quantizers", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow a missing frozen checkpoint to be downloaded.",
    )
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.quantizers != 8:
        parser.error("the matched real-speech benchmark requires 8 quantizers")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")
    return args


def main() -> None:
    """Load the frozen model once and print JSON plus a Markdown table."""
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the fixed optimized benchmark requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    maximum_samples = max(args.audio_seconds) * SAMPLE_RATE
    fixed_crop_samples = FIXED_CROP_SECONDS * SAMPLE_RATE
    fixed_audio_cpu, crop_offset = _load_real_audio(
        args.audio_file,
        samples=fixed_crop_samples,
        crop_seed=args.seed,
    )
    if maximum_samples > fixed_audio_cpu.shape[-1]:
        raise RuntimeError("the canonical fixed crop is shorter than the benchmark input")
    fixed_audio = fixed_audio_cpu.to(device="cuda", dtype=torch.float32)
    model = MimiModel.from_pretrained(
        device="cuda",
        local_files_only=not args.allow_download,
    )
    reference = PureTorchMimi(model)

    results = [
        _benchmark_duration(
            model,
            reference,
            fixed_audio,
            seconds=seconds,
            pairs=args.pairs,
            seed=args.seed,
            quantizers=args.quantizers,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
        for seconds in args.audio_seconds
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "sample_rate": SAMPLE_RATE,
        "input_kind": "real_speech",
        "real_audio_id": REAL_AUDIO_ID,
        "real_audio_dataset": REAL_AUDIO_DATASET_ID,
        "real_audio_dataset_revision": REAL_AUDIO_DATASET_REVISION,
        "real_audio_sha256": REAL_AUDIO_SHA256,
        "real_audio_file": str(args.audio_file.resolve()),
        "fixed_crop_seed": args.seed,
        "fixed_crop_offset_samples": crop_offset,
        "fixed_crop_offset_seconds": crop_offset / SAMPLE_RATE,
        "fixed_crop_seconds": FIXED_CROP_SECONDS,
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2))
    print()
    print(
        "| Audio | Reference | Fast-Mimi | Paired median speedup "
        "| 95% paired CI | Quality |"
    )
    print("|---:|---:|---:|---:|---:|---|")
    for result in results:
        quality = (
            f"codes {result.code_mismatches}, wave {result.waveform_violations}; "
            f"{result.runtime_path}"
        )
        print(
            f"| {result.audio_seconds} s | {result.reference_median_ms:.3f} ms "
            f"| {result.candidate_median_ms:.3f} ms "
            f"| {result.paired_median_speedup:.4f}x "
            f"| {result.ci95_low:.4f}x–{result.ci95_high:.4f}x | {quality} |"
        )


if __name__ == "__main__":
    main()
