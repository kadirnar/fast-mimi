"""Small stand-ins for the fast-kernel harness helpers the kernels were written against.

`Graphed` captures a callable with static shapes into one CUDA graph; `ensure_cuda_home()` finds a pip-installed
nvcc (torch's cu13 wheels depend on `nvidia-cuda-nvcc`) and exports CUDA_HOME/PATH so `torch.utils.cpp_extension`
can build the CUDA C++ kernels; `build_dir()` is the extension cache directory.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Any

EAGER = False   # when True, Graphed.__call__ runs the captured function eagerly (profiling / debugging)


class eager_mode:
    """Context manager: run every Graphed callable eagerly."""

    def __enter__(self):
        global EAGER
        self._prev = EAGER
        EAGER = True
        return self

    def __exit__(self, *exc):
        global EAGER
        EAGER = self._prev
        return False


class Graphed:
    """Capture `fn(*example_args)` into a CUDA graph; calls copy into the static inputs, replay, return static outputs."""

    def __init__(self, fn, example_args: tuple[Any, ...] = (), example_kwargs: dict[str, Any] | None = None,
                 warmup: int = 3, pool=None):
        import torch
        self.fn = fn
        self.args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in example_args)
        self.kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in (example_kwargs or {}).items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup):
                self.fn(*self.args, **self.kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, pool=pool):
            self.out = self.fn(*self.args, **self.kwargs)
        torch.cuda.synchronize()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        if EAGER:
            with torch.inference_mode():
                return self.fn(*args, **kwargs)
        for dst, src in zip(self.args, args, strict=False):
            if isinstance(dst, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        for key, src in kwargs.items():
            dst = self.kwargs.get(key)
            if isinstance(dst, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        self.graph.replay()
        return self.out


def _site_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        import site
        dirs += site.getsitepackages()
        user = site.getusersitepackages()
        if user:
            dirs.append(user)
    except Exception:  # noqa: BLE001
        pass
    dirs += [p for p in sys.path if p.endswith("site-packages")]
    return list(dict.fromkeys(d for d in dirs if d and os.path.isdir(d)))


def find_nvcc() -> tuple[str | None, str | None]:
    """(nvcc, cuda_home): FAST_MIMI_CUDA_HOME, the newest toolchain under ~/.cache/fast-kernel/toolchains, CUDA_HOME/CUDA_PATH,
    PATH, pip wheels in the venv (nvidia/cu*/bin/nvcc), /usr/local/cuda*, /opt/cuda*."""
    forced = os.environ.get("FAST_MIMI_CUDA_HOME") or os.environ.get("FAST_KERNEL_CUDA_HOME")
    if forced and Path(forced, "bin", "nvcc").exists():
        return str(Path(forced, "bin", "nvcc")), forced
    # newest self-contained toolchain first (`fast-kernel toolchain install --cuda 13.3` / a newer pip wheel set):
    # the CUDA release bundled with torch may be older than the host compiler supports
    cache = Path(os.environ.get("FAST_KERNEL_TOOLCHAINS") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fast-kernel" / "toolchains")
    tool = sorted(glob.glob(str(cache / "cuda-*" / "nvidia" / "cu*" / "bin" / "nvcc")),
                  key=lambda q: tuple(int(x) if x.isdigit() else 0 for x in Path(q).parents[3].name[len("cuda-"):].split(".")), reverse=True)
    for path in tool:
        if os.access(path, os.X_OK):
            return path, str(Path(path).parent.parent)
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home and Path(env_home, "bin", "nvcc").exists():
        return str(Path(env_home, "bin", "nvcc")), env_home
    on_path = shutil.which("nvcc")
    if on_path:
        return on_path, str(Path(on_path).resolve().parent.parent)
    candidates: list[str] = []
    for site_dir in _site_dirs():
        candidates += glob.glob(os.path.join(site_dir, "nvidia", "cu*", "bin", "nvcc"))
        candidates += glob.glob(os.path.join(site_dir, "nvidia", "cuda_nvcc", "bin", "nvcc"))
    candidates += glob.glob("/usr/local/cuda*/bin/nvcc") + glob.glob("/opt/cuda*/bin/nvcc")
    for path in candidates:
        if os.access(path, os.X_OK):
            return path, str(Path(path).parent.parent)
    return None, None


def _ensure_link_layout(home: str) -> None:
    """pip CUDA wheels ship lib/libcudart.so.13 but linkers want -lcudart (libcudart.so) and torch looks in lib64/."""
    root = Path(home)
    lib = root / "lib"
    if not lib.is_dir():
        return
    try:
        for versioned in lib.glob("lib*.so.*"):
            unversioned = lib / (versioned.name.split(".so.")[0] + ".so")
            if not unversioned.exists():
                unversioned.symlink_to(versioned.name)
        lib64 = root / "lib64"
        if not lib64.exists():
            lib64.symlink_to("lib")
    except OSError:
        pass


def ensure_cuda_home() -> str | None:
    """Export CUDA_HOME/PATH for a discovered nvcc (call before importing torch.utils.cpp_extension)."""
    for venv_bin in {str(Path(sys.executable).parent), str(Path(sys.prefix) / "bin")}:
        if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    nvcc, home = find_nvcc()
    if home:
        flags = os.environ.get("NVCC_APPEND_FLAGS", "")
        if "-allow-unsupported-compiler" not in flags:
            os.environ["NVCC_APPEND_FLAGS"] = (flags + " -allow-unsupported-compiler").strip()
        os.environ["CUDA_HOME"] = home
        os.environ["CUDA_PATH"] = home
        _ensure_link_layout(home)
        bin_dir = str(Path(nvcc).parent)
        if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            import torch
            if torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:  # noqa: BLE001
            pass
    return home


def build_dir(_root: Path | None = None) -> Path:
    """Cache directory for the compiled CUDA extensions (FAST_MIMI_BUILD_DIR or ~/.cache/fast-mimi/fp32)."""
    env = os.environ.get("FAST_MIMI_BUILD_DIR")
    path = Path(env) if env else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fast-mimi" / "fp32"
    path.mkdir(parents=True, exist_ok=True)
    return path
