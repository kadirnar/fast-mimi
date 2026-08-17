"""Build and launch the accepted fused decoder-12/final SM120 WMMA kernel."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("decoder12_final_wmma.cu")
_LOADED: Decoder12FinalWmma | None = None
_LOAD_ATTEMPTED = False


def _find_nvcc() -> Path:
    executable = shutil.which("nvcc")
    if executable is not None:
        return Path(executable).resolve()
    root = Path(torch.__file__).resolve().parent.parent
    candidates = sorted(root.glob("nvidia/**/bin/nvcc"), reverse=True)
    if not candidates:
        raise RuntimeError("nvcc unavailable")
    return candidates[0].resolve()


def _build() -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"unvalidated CUDA capability {capability}")
    nvcc = _find_nvcc()
    version = subprocess.run(
        [str(nvcc), "--version"], check=True, capture_output=True, text=True
    ).stdout
    digest = hashlib.sha256(
        _SOURCE.read_bytes()
        + str(nvcc).encode()
        + version.encode()
        + repr(capability).encode()
    ).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / "fast-kernel-mimi-native"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = cache / f"decoder12_final_wmma_sm120_{digest}.so"
    lock_path = cache / f"decoder12_final_wmma_sm120_{digest}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        temporary = cache / f".{output.name}.{os.getpid()}.tmp"
        result = subprocess.run(
            [
                str(nvcc),
                "--allow-unsupported-compiler",
                "-O3",
                "-std=c++17",
                "-shared",
                "-Xcompiler=-fPIC",
                "-gencode=arch=compute_120,code=sm_120",
                str(_SOURCE),
                "-o",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError((result.stderr or result.stdout)[-6000:])
        os.replace(temporary, output)
    return output


class Decoder12FinalWmma:
    """Fused half pointwise, FP32 residual ELU, and final TF32 convolution."""

    def __init__(self, path: Path | None = None) -> None:
        self.library = ctypes.CDLL(str(_build() if path is None else path))
        self.function = self.library.mimi_decoder12_final_wmma
        self.function.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 2
            + [ctypes.c_void_p]
        )
        self.function.restype = ctypes.c_int

    def __call__(
        self,
        branch: torch.Tensor,
        pointwise_weight: torch.Tensor,
        residual: torch.Tensor,
        pointwise_bias: torch.Tensor,
        final_weight: torch.Tensor,
        final_bias: torch.Tensor,
        output: torch.Tensor,
        *,
        config: int = 0,
    ) -> torch.Tensor:
        length = branch.shape[0]
        if (
            branch.shape != (length, 32)
            or branch.dtype != torch.float16
            or pointwise_weight.shape != (64, 32)
            or pointwise_weight.dtype != torch.float16
            or residual.shape != (length, 64)
            or residual.dtype != torch.float32
            or pointwise_bias.shape != (64,)
            or pointwise_bias.dtype != torch.float32
            or final_weight.shape != (1, 64, 3)
            or final_weight.dtype != torch.float32
            or final_bias.shape != (1,)
            or final_bias.dtype != torch.float32
            or output.shape != (1, 1, length)
            or output.dtype != torch.float32
            or not all(
                value.is_cuda and value.is_contiguous()
                for value in (
                    branch,
                    pointwise_weight,
                    residual,
                    pointwise_bias,
                    final_weight,
                    final_bias,
                    output,
                )
            )
        ):
            raise RuntimeError("unexpected decoder-12 geometry")
        status = self.function(
            branch.data_ptr(),
            pointwise_weight.data_ptr(),
            residual.data_ptr(),
            pointwise_bias.data_ptr(),
            final_weight.data_ptr(),
            final_bias.data_ptr(),
            output.data_ptr(),
            length,
            config,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"decoder-12/final launch failed: {status}")
        return output


def load_decoder12_final_wmma() -> Decoder12FinalWmma | None:
    """Load once; preserve the existing decoder tail if native build fails."""
    global _LOADED, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOADED
    _LOAD_ATTEMPTED = True
    try:
        _LOADED = Decoder12FinalWmma(_build())
    except (OSError, RuntimeError, subprocess.SubprocessError):
        _LOADED = None
    return _LOADED


if sys.platform != "linux":
    _LOAD_ATTEMPTED = True
