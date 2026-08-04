"""Optional exact-float32 CUDA backend for the profiled Q32 RVQ decode path.

The source is compiled lazily into an external cache. The Python package keeps
no generated binary and callers automatically fall back to PyTorch when the
compiler, shape, device, or launch contract is unavailable.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

_QUANTIZERS = 32
_CODEBOOK_SIZE = 2048
_DIMENSION = 256
_FRAMES = 13
_COMPILE_LOCK = threading.Lock()
_LIBRARIES: dict[tuple[str, str, str], ctypes.CDLL] = {}

# The kernel mirrors PyTorch's left-to-right float32 RVQ accumulation. Explicit
# round-to-nearest additions plus the compile flags below prevent contraction
# or fast-math rewrites from changing checkpoint output bits.
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kQuantizers = 32;
constexpr int kCodebookDimension = 256;
constexpr int kFrames = 13;
constexpr int kOutputElements = kCodebookDimension * kFrames;

__global__ void split_rvq_decode_q32_kernel(
    const int64_t* __restrict__ codes,
    const float* const* __restrict__ codebooks,
    float* __restrict__ semantic,
    float* __restrict__ acoustic) {
    const int output_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output_index >= kOutputElements) {
        return;
    }

    const int channel = output_index / kFrames;
    const int frame = output_index - channel * kFrames;
    const int64_t semantic_code = codes[frame];
    const int64_t semantic_offset =
        semantic_code * kCodebookDimension + channel;
    semantic[output_index] =
        __fadd_rn(0.0f, codebooks[0][semantic_offset]);

    float acoustic_value = 0.0f;
#pragma unroll 1
    for (int quantizer = 1; quantizer < kQuantizers; ++quantizer) {
        const int64_t code = codes[quantizer * kFrames + frame];
        const int64_t codebook_offset =
            code * kCodebookDimension + channel;
        acoustic_value = __fadd_rn(
            acoustic_value, codebooks[quantizer][codebook_offset]);
    }
    acoustic[output_index] = acoustic_value;
}

}  // namespace

extern "C" int launch_split_rvq_decode_q32(
    const int64_t* codes,
    const float* const* codebooks,
    float* semantic,
    float* acoustic,
    void* stream_pointer) {
    constexpr int threads = 256;
    constexpr int blocks = (kOutputElements + threads - 1) / threads;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    split_rvq_decode_q32_kernel<<<blocks, threads, 0, stream>>>(
        codes, codebooks, semantic, acoustic);
    return static_cast<int>(cudaGetLastError());
}
"""


def _nvcc_path() -> Path:
    """Resolve and validate the optional CUDA compiler.

    Returns:
        Absolute path to the configured or discoverable ``nvcc`` executable.

    Raises:
        RuntimeError: If ``nvcc`` cannot be found or is not a regular file.
    """
    configured = os.environ.get("FAST_MIMI_CUDA_NVCC")
    discovered = configured or shutil.which("nvcc")
    if discovered is None:
        raise RuntimeError("nvcc is unavailable")
    compiler = Path(discovered).expanduser().resolve()
    if not compiler.is_file():
        raise RuntimeError(f"nvcc does not exist: {compiler}")
    return compiler


def _host_compiler() -> str | None:
    """Resolve the host compiler used by ``nvcc``.

    Returns:
        The explicit compiler path, a discovered ``g++-15`` path, or ``None``
        to let ``nvcc`` select its default host compiler.

    Raises:
        RuntimeError: If an explicitly configured compiler is not a file.
    """
    configured = os.environ.get("FAST_MIMI_CUDA_HOST_COMPILER")
    if configured:
        compiler = Path(configured).expanduser().resolve()
        if not compiler.is_file():
            raise RuntimeError(f"CUDA host compiler does not exist: {compiler}")
        return str(compiler)
    return shutil.which("g++-15")


def _cache_directory() -> Path:
    """Choose the external cache for lazily compiled native libraries.

    Returns:
        An absolute configured cache path, or the platform-style
        ``$XDG_CACHE_HOME/fast-mimi/cuda`` fallback.
    """
    configured = os.environ.get("FAST_MIMI_CUDA_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_root = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
    return root / "fast-mimi" / "cuda"


def _load_library(architecture: str) -> ctypes.CDLL:
    """Build and load the native library for one CUDA architecture.

    Args:
        architecture: Compact CUDA architecture string such as ``"120"``.

    Returns:
        Loaded shared-library handle cached for this compiler and architecture.

    Raises:
        RuntimeError: If compiler discovery or validation fails.
        subprocess.SubprocessError: If compiler probing or compilation fails.
        OSError: If the generated library cannot be written or loaded.
    """
    compiler = _nvcc_path()
    host_compiler = _host_compiler()
    version = subprocess.run(
        [str(compiler), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    host_fingerprint = host_compiler or "nvcc-default-host-compiler"
    key = (str(compiler), architecture, host_fingerprint)
    with _COMPILE_LOCK:
        loaded = _LIBRARIES.get(key)
        if loaded is not None:
            return loaded
        digest = hashlib.sha256(
            (
                _CUDA_SOURCE + str(compiler) + version + architecture + host_fingerprint
            ).encode()
        ).hexdigest()[:20]
        cache = _cache_directory()
        cache.mkdir(parents=True, exist_ok=True)
        library_path = cache / f"rvq_decode_{digest}.so"
        if not library_path.is_file():
            source_path = cache / f"rvq_decode_{digest}.{os.getpid()}.cu"
            temporary_library = cache / f"rvq_decode_{digest}.{os.getpid()}.so"
            source_path.write_text(_CUDA_SOURCE, encoding="utf-8")
            command = [
                str(compiler),
                "-O3",
                f"-arch=sm_{architecture}",
                "--fmad=false",
                "--cudart=shared",
                "-shared",
                "-Xcompiler=-fPIC",
                "-Xcompiler=-fno-fast-math",
                "-Xcompiler=-ffp-contract=off",
            ]
            if host_compiler is not None:
                command[1:1] = ["-ccbin", host_compiler]
            command.extend((str(source_path), "-o", str(temporary_library)))
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                os.replace(temporary_library, library_path)
            finally:
                source_path.unlink(missing_ok=True)
                temporary_library.unlink(missing_ok=True)
        library = ctypes.CDLL(str(library_path))
        _LIBRARIES[key] = library
        return library


class CudaRVQDecodeBackend:
    """Launch the exact ordered Q32 split-RVQ gather/add kernel.

    The backend stores device pointers to all checkpoint codebooks. It accepts
    only the profiled batch-one, 32-codebook, 13-frame workload so unsupported
    calls cannot silently use a numerically unverified specialization.
    """

    def __init__(self, codebooks: Sequence[Tensor]) -> None:
        """Validate codebooks and bind the architecture-specific launcher.

        Args:
            codebooks: Ordered sequence of 32 contiguous float32 CUDA tensors,
                each with shape ``[2048, 256]``.

        Raises:
            ValueError: If the count, shape, dtype, device, or layout differs
                from the verified Q32 contract.
            RuntimeError: If no usable CUDA compiler is available.
            OSError: If the cached native library cannot be loaded.
        """
        if len(codebooks) != _QUANTIZERS:
            raise ValueError(f"expected {_QUANTIZERS} codebooks")
        device = codebooks[0].device
        for codebook in codebooks:
            if (
                codebook.shape != (_CODEBOOK_SIZE, _DIMENSION)
                or codebook.dtype != torch.float32
                or codebook.device != device
                or not codebook.is_contiguous()
            ):
                raise ValueError("unsupported CUDA RVQ codebook")
        capability = torch.cuda.get_device_capability(device)
        architecture = f"{capability[0]}{capability[1]}"
        self.device = device
        self.codebook_pointers = torch.tensor(
            [codebook.data_ptr() for codebook in codebooks],
            dtype=torch.int64,
            device=device,
        )
        self.library = _load_library(architecture)
        self.launch = self.library.launch_split_rvq_decode_q32
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def __call__(self, codes: Tensor) -> tuple[Tensor, Tensor]:
        """Decode fixed Q32 codes into semantic and acoustic embeddings.

        Args:
            codes: Contiguous int64 CUDA tensor with shape ``[1, 32, 13]`` on
                the same device as the codebooks.

        Returns:
            A pair of float32 tensors with shape ``[1, 256, 13]``. The first
            contains the semantic codebook and the second contains the ordered
            sum of all acoustic codebooks.

        Raises:
            ValueError: If ``codes`` does not match the fixed kernel contract.
            RuntimeError: If CUDA reports a kernel launch error.
        """
        if (
            codes.shape != (1, _QUANTIZERS, _FRAMES)
            or codes.dtype != torch.int64
            or codes.device != self.device
            or not codes.is_contiguous()
        ):
            raise ValueError("unsupported CUDA RVQ codes")
        semantic = torch.empty(
            (1, _DIMENSION, _FRAMES), dtype=torch.float32, device=codes.device
        )
        acoustic = torch.empty_like(semantic)
        stream = torch.cuda.current_stream(codes.device).cuda_stream
        status = self.launch(
            codes.data_ptr(),
            self.codebook_pointers.data_ptr(),
            semantic.data_ptr(),
            acoustic.data_ptr(),
            stream,
        )
        if status != 0:
            raise RuntimeError(f"CUDA RVQ launch failed with status {status}")
        return semantic, acoustic
