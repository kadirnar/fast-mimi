"""Build and launch the accepted SM120 fused decoder-9 WMMA kernel."""

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

_SOURCE = Path(__file__).with_name("decoder9_fused_wmma.cu")
_LOADED: FusedDecoder9Wmma | None = None
_LOAD_ATTEMPTED = False


def _find_nvcc() -> Path:
    executable = shutil.which("nvcc")
    if executable is not None:
        return Path(executable).resolve()
    site_packages = Path(torch.__file__).resolve().parent.parent
    candidates = sorted(site_packages.glob("nvidia/**/bin/nvcc"), reverse=True)
    if not candidates:
        raise RuntimeError("nvcc is unavailable")
    return candidates[0].resolve()


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
    output = cache / f"decoder9_fused_wmma_sm120_{digest}.so"
    lock_path = cache / f"decoder9_fused_wmma_sm120_{digest}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        temporary = cache / f".{output.name}.{os.getpid()}.tmp"
        command = [
            str(nvcc),
            "--allow-unsupported-compiler",
            "-O3",
            "-std=c++17",
            "-shared",
            "-Xcompiler=-fPIC",
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            "-gencode=arch=compute_120,code=sm_120",
            str(_SOURCE),
            "-o",
            str(temporary),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            temporary.unlink(missing_ok=True)
            diagnostics = (result.stderr or result.stdout)[-4000:]
            raise RuntimeError(f"decoder-9 WMMA build failed: {diagnostics}")
        os.replace(temporary, output)
    return output


class FusedDecoder9Wmma:
    """FP16 branch GEMM with FP32 residual+bias and fused ELU output."""

    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.mimi_decoder9_fused_wmma
        self.function.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_void_p]
        )
        self.function.restype = ctypes.c_int

    def __call__(
        self,
        branch: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        rows = branch.shape[0]
        if (
            rows != 600_000
            or branch.dtype != torch.float16
            or weight.dtype != torch.float16
            or residual.dtype != torch.float32
            or bias.dtype != torch.float32
            or output.dtype != torch.float16
            or branch.shape != (rows, 64)
            or weight.shape != (128, 64)
            or residual.shape != (rows, 128)
            or bias.shape != (128,)
            or output.shape != (rows, 128)
            or not all(
                tensor.is_cuda and tensor.is_contiguous()
                for tensor in (branch, weight, residual, bias, output)
            )
        ):
            raise RuntimeError("unexpected frozen 100-second Mimi decoder-9 geometry")
        status = self.function(
            branch.data_ptr(),
            weight.data_ptr(),
            residual.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            rows,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"decoder-9 WMMA launch failed with status {status}")
        return output


def load_fused_decoder9_wmma() -> FusedDecoder9Wmma | None:
    """Load once; preserve the measured cuDNN path if the native build fails."""
    global _LOADED, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOADED
    _LOAD_ATTEMPTED = True
    try:
        _LOADED = FusedDecoder9Wmma(_build_library())
    except (OSError, RuntimeError, subprocess.SubprocessError):
        _LOADED = None
    return _LOADED


if sys.platform != "linux":
    _LOAD_ATTEMPTED = True
