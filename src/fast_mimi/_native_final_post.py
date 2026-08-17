"""Build and launch the accepted frozen SM120 Mimi final decoder post."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from ._native_cutlass import _find_nvcc

_SOURCE = Path(__file__).with_name("final_post_cuda.cu")
_LOADED: NativeFinalPost | None = None
_LOAD_ATTEMPTED = False


def _build_library() -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"unvalidated CUDA capability {capability}")
    nvcc = _find_nvcc()
    version = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    identity = hashlib.sha256()
    identity.update(_SOURCE.read_bytes())
    identity.update(str(nvcc).encode())
    identity.update(version.encode())
    identity.update(repr(capability).encode())
    digest = identity.hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / "fast-kernel-mimi-native"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = cache / f"final_post_sm120_{digest}.so"
    lock_path = cache / f"final_post_sm120_{digest}.lock"
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
            diagnostics = (result.stderr or result.stdout)[-4000:]
            raise RuntimeError(f"native final-post build failed: {diagnostics}")
        os.replace(temporary, output)
    return output


class NativeFinalPost:
    """Callable TF32-input/FP32-accumulate fused final decoder operation."""

    def __init__(self) -> None:
        self.library = ctypes.CDLL(str(_build_library()))
        self.function = self.library.mimi_final_post_cuda
        self.function.argtypes = (
            [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
        )
        self.function.restype = ctypes.c_int

    def __call__(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        *,
        reduction_order: int,
        exp_mode: int,
        config: int,
    ) -> torch.Tensor:
        if (
            residual.shape != branch.shape
            or residual.ndim != 4
            or residual.shape[0] != 1
            or residual.shape[1] != 64
            or residual.shape[2] != 1
            or residual.dtype != torch.float32
            or branch.dtype != torch.float32
            or weight.shape != (1, 64, 3)
            or weight.dtype != torch.float32
            or bias.shape != (1,)
            or bias.dtype != torch.float32
            or output.shape != (1, 1, residual.shape[-1])
            or output.dtype != torch.float32
            or not residual.is_cuda
            or not branch.is_cuda
            or not weight.is_cuda
            or not bias.is_cuda
            or not output.is_cuda
            or len(
                {
                    residual.device,
                    branch.device,
                    weight.device,
                    bias.device,
                    output.device,
                }
            )
            != 1
            or residual.stride(1) != 1
            or residual.stride(3) != 64
            or branch.stride(1) != 1
            or branch.stride(3) != 64
            or not weight.is_contiguous()
            or not bias.is_contiguous()
            or not output.is_contiguous()
        ):
            raise RuntimeError("unexpected frozen Mimi final-post geometry")
        status = self.function(
            reduction_order,
            exp_mode,
            config,
            residual.data_ptr(),
            branch.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            residual.shape[-1],
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"native final-post launch failed with status {status}")
        return output


def load_native_final_post() -> NativeFinalPost | None:
    """Load once; retain the quality-safe cuDNN route when CUDA build is unavailable."""
    global _LOADED, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOADED
    _LOAD_ATTEMPTED = True
    try:
        _LOADED = NativeFinalPost()
    except (OSError, RuntimeError, subprocess.SubprocessError):
        _LOADED = None
    return _LOADED


if sys.platform != "linux":  # fcntl and the measured native path are Linux-only.
    _LOAD_ATTEMPTED = True
