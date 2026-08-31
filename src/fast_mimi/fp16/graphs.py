"""CUDA-graph wrapper: removes all CPU/launch overhead for fixed shapes (one graph per input shape)."""
from __future__ import annotations
import torch


class GraphedMimi:
    """Wraps any object exposing encode(audio)->codes and decode(codes, length)->audio.

    Shapes are captured lazily; the returned tensors are the graph's static output buffers (clone if you
    need to keep them across calls).
    """

    def __init__(self, model, warmup: int = 3, pool=None):
        self.m = model
        self.warmup = warmup
        self._enc = {}
        self._dec = {}
        self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()

    def _capture(self, fn, *static_args):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                out = fn(*static_args)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            out = fn(*static_args)
        return g, out

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, num_quantizers: int | None = None):
        key = (tuple(audio.shape), num_quantizers)
        if key not in self._enc:
            static = audio.clone()
            g, out = self._capture(lambda a: self.m.encode(a, num_quantizers), static)
            self._enc[key] = (g, static, out)
        g, static, out = self._enc[key]
        static.copy_(audio)
        g.replay()
        return out

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, length: int | None = None):
        key = (tuple(codes.shape), length)
        if key not in self._dec:
            static = codes.clone()
            g, out = self._capture(lambda c: self.m.decode(c, length), static)
            self._dec[key] = (g, static, out)
        g, static, out = self._dec[key]
        static.copy_(codes)
        g.replay()
        return out
