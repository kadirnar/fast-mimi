"""Build and launch the guarded SM120 CUTLASS decoder-tail kernel."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("cutlass_bias_dgrad.cu")
_LOADED: CutlassBiasDgrad | None = None
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


def _find_cutlass_include() -> Path:
    specification = importlib.util.find_spec("tilelang")
    if specification is None or specification.origin is None:
        raise RuntimeError("TileLang's CUTLASS headers are unavailable")
    include = (
        Path(specification.origin).resolve().parent / "3rdparty" / "cutlass" / "include"
    )
    if not (include / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError("TileLang's CUTLASS include tree is incomplete")
    return include


def _build_library() -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"unvalidated CUDA capability {capability}")
    nvcc = _find_nvcc()
    include = _find_cutlass_include()
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
    identity.update(str(include).encode())
    identity.update(repr(capability).encode())
    digest = identity.hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / "fast-mimi-native"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = cache / f"cutlass_bias_dgrad_sm120_{digest}.so"
    lock_path = cache / f"cutlass_bias_dgrad_sm120_{digest}.lock"
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
            "-gencode=arch=compute_120,code=sm_120",
            f"-I{include}",
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
            raise RuntimeError(f"CUTLASS build failed: {diagnostics}")
        os.replace(temporary, output)
    return output


class CutlassBiasDgrad:
    """Callable FP16-input/FP32-output layer-11 causal transposed convolution."""

    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.mimi_cutlass_bias_dgrad
        self.function.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        self.function.restype = ctypes.c_int

    def __call__(
        self,
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        if (
            values.dtype != torch.float16
            or weight.dtype != torch.float16
            or bias.dtype != torch.float32
            or output.dtype != torch.float32
            or values.ndim != 4
            or weight.ndim != 4
            or output.ndim != 4
            or values.shape[0] != 1
            or values.shape[2] != 1
            or weight.shape[2] != 1
            or output.shape[0] != 1
            or output.shape[2] != 1
            or weight.shape[1] != 64
            or output.shape[1] != 64
            or weight.shape[-1] != 8
            or stride != 4
            or output.shape[-1] != values.shape[-1] * stride
            or not values.is_contiguous(memory_format=torch.channels_last)
            or not weight.is_contiguous(memory_format=torch.channels_last)
            or not output.is_contiguous(memory_format=torch.channels_last)
            or not bias.is_contiguous()
        ):
            raise RuntimeError("unexpected frozen Mimi layer-11 geometry")
        status = self.function(
            values.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            values.shape[-1],
            values.shape[1],
            weight.shape[1],
            weight.shape[-1],
            stride,
            output.shape[-1],
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"CUTLASS dgrad launch failed with status {status}")
        return output


def load_cutlass_bias_dgrad() -> CutlassBiasDgrad | None:
    """Load once; keep the existing cuDNN route when this toolchain is unavailable."""
    global _LOADED, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOADED
    _LOAD_ATTEMPTED = True
    try:
        _LOADED = CutlassBiasDgrad(_build_library())
    except (OSError, RuntimeError, subprocess.SubprocessError):
        _LOADED = None
    return _LOADED


if sys.platform != "linux":  # fcntl and the measured native path are Linux-only.
    _LOAD_ATTEMPTED = True
