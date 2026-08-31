"""SEANet encoder/decoder built from the Triton conv kernels (channels-last, bf16 activations)."""
from __future__ import annotations
import math
import torch
from .conv import Conv1d, ConvT1d, conv_first, conv_first_res, conv_last


class TritonSEANetEncoder:
    """audio [L] fp32 -> [T25, 512] fp32.  Activations are applied by the producing kernel (ELU epilogue), so
    every conv streams plain tiles into the tensor cores; resnet inputs get a dual (raw, ELU) output."""

    def __init__(self, state, cfg, tune=True, fp32_max_T=512, wdtype="bf16"):
        g = lambda k: state[k].float()
        self.tune, self.fp32_max_T = tune, fp32_max_T
        wd = dict(wdtype=wdtype)
        self.w0, self.b0 = g("encoder.layers.0.conv.weight").contiguous(), g("encoder.layers.0.conv.bias")
        self.blocks = []
        idx = 1
        for i, ratio in enumerate(reversed(cfg.upsampling_ratios)):
            p = f"encoder.layers.{idx}."
            c1 = Conv1d(g(p + "block.1.conv.weight"), g(p + "block.1.conv.bias"), name=f"enc.res{i}.c1", **wd)
            c2 = Conv1d(g(p + "block.3.conv.weight"), g(p + "block.3.conv.bias"), res=True, name=f"enc.res{i}.c2", **wd)
            idx += 2
            down = Conv1d(g(f"encoder.layers.{idx}.conv.weight"), g(f"encoder.layers.{idx}.conv.bias"), stride=ratio, name=f"enc.down{i}", **wd)
            idx += 1
            self.blocks.append((c1, c2, down, ratio))
        idx += 1
        self.last = Conv1d(g(f"encoder.layers.{idx}.conv.weight"), g(f"encoder.layers.{idx}.conv.bias"), name="enc.last", **wd)
        self._buf = {}

    def _buffers(self, L, dev):
        if L in self._buf:
            return self._buf[L]
        b = {}
        T = L
        dt = lambda T: torch.float32 if T <= self.fp32_max_T else torch.bfloat16
        C = 64
        b["x0a"] = torch.empty(T, C, dtype=torch.bfloat16, device=dev)     # ELU(conv0); the raw residual is recomputed by c2
        for i, (c1, c2, down, ratio) in enumerate(self.blocks):
            b[f"h{i}"] = torch.empty(T, C // 2, dtype=dt(T), device=dev)     # ELU(c1 out)
            b[f"r{i}"] = torch.empty(T, C, dtype=dt(T), device=dev)          # ELU(resnet out)
            T = math.ceil(T / ratio)
            C *= 2
            b[f"d{i}"] = torch.empty(T, C, dtype=dt(T), device=dev)          # raw down output (residual of next block)
            b[f"d{i}a"] = torch.empty(T, C, dtype=dt(T), device=dev)         # ELU(down output)
            if T <= 64:
                b[f"d{i}32"] = torch.empty(T, C, dtype=torch.float32, device=dev)
        b["out"] = torch.empty(T, 512, dtype=torch.float32, device=dev)
        b["out32"] = torch.empty(T, 512, dtype=torch.float32, device=dev)
        self._buf[L] = b
        if self.tune:
            self._run(b, torch.zeros(L, device=dev), tune=True)
        return b

    def _run(self, b, audio, tune=False):
        conv_first(audio, self.w0, self.b0, None, b["x0a"])
        x_raw, x_act = None, b["x0a"]
        for i, (c1, c2, down, ratio) in enumerate(self.blocks):
            kw1 = dict(y=None, y_act=b[f"h{i}"])
            kw2 = dict(y=None, res=x_raw, y_act=b[f"r{i}"]) if i > 0 else dict(y=None, y_act=b["r0"], res_conv0=(audio, self.w0, self.b0))
            kw3 = dict(y=b[f"d{i}"], y_act=b[f"d{i}a"], y32=b.get(f"d{i}32"))
            if tune:
                c1.tune(x_act, **kw1); c2.tune(b[f"h{i}"], **kw2); down.tune(b[f"r{i}"], **kw3)
            c1(x_act, **kw1)
            c2(b[f"h{i}"], **kw2)
            down(b[f"r{i}"], **kw3)
            x_raw, x_act = b[f"d{i}"], b[f"d{i}a"]
        kw = dict(y=b["out"], y32=b["out32"])
        if tune:
            self.last.tune(x_act, **kw)
        self.last(x_act, **kw)
        return b["out"]

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        L = audio.shape[-1]
        b = self._buffers(L, audio.device)
        return self._run(b, audio.reshape(-1).float().contiguous())


class TritonSEANetDecoder:
    """[T25, 512] fp32 -> audio [T25*960] fp32"""

    def __init__(self, state, cfg, tune=True, fp32_max_T=512, wdtype="bf16"):
        g = lambda k: state[k].float()
        self.tune, self.fp32_max_T = tune, fp32_max_T
        wd = dict(wdtype=wdtype)
        self.first = Conv1d(g("decoder.layers.0.conv.weight"), g("decoder.layers.0.conv.bias"), name="dec.first", **wd)
        self.blocks = []
        idx = 1
        for i, ratio in enumerate(cfg.upsampling_ratios):
            idx += 1
            up = ConvT1d(g(f"decoder.layers.{idx}.conv.weight"), g(f"decoder.layers.{idx}.conv.bias"), stride=ratio, elu_in=False, name=f"dec.up{i}", **wd)
            idx += 1
            p = f"decoder.layers.{idx}."
            c1 = Conv1d(g(p + "block.1.conv.weight"), g(p + "block.1.conv.bias"), name=f"dec.res{i}.c1", **wd)
            c2 = Conv1d(g(p + "block.3.conv.weight"), g(p + "block.3.conv.bias"), res=True, name=f"dec.res{i}.c2", **wd)
            idx += 1
            self.blocks.append((up, c1, c2, ratio))
        idx += 1
        self.wl, self.bl = g(f"decoder.layers.{idx}.conv.weight").contiguous(), g(f"decoder.layers.{idx}.conv.bias")
        self._buf = {}

    def _buffers(self, T, dev):
        if T in self._buf:
            return self._buf[T]
        T0 = T
        dt = lambda T: torch.float32 if T <= self.fp32_max_T else torch.bfloat16
        b = {"fa": torch.empty(T, 1024, dtype=dt(T), device=dev), "f32": torch.empty(T, 1024, dtype=torch.float32, device=dev)}
        C = 1024
        for i, (up, c1, c2, ratio) in enumerate(self.blocks):
            T = T * ratio
            C //= 2
            b[f"u{i}"] = torch.empty(T, C, dtype=dt(T), device=dev)        # raw convT output (residual)
            b[f"u{i}a"] = torch.empty(T, C, dtype=dt(T), device=dev)       # ELU(convT output)
            if T <= 64:
                b[f"u{i}32"] = torch.empty(T, C, dtype=torch.float32, device=dev)
            b[f"h{i}"] = torch.empty(T, C // 2, dtype=dt(T), device=dev)
            b[f"r{i}"] = torch.empty(T, C, dtype=dt(T), device=dev)        # ELU(resnet out)
        b["out"] = torch.empty(T, dtype=torch.float32, device=dev)
        self._buf[T0] = b
        if self.tune:
            self._run(b, torch.zeros(T0, 512, dtype=torch.float32, device=dev), tune=True)
        return b

    def _run(self, b, x, tune=False):
        kw = dict(y=None, y_act=b["fa"], y32=b["f32"])
        if tune:
            self.first.tune(x, **kw)
        self.first(x, **kw)
        x_act = b["fa"]
        for i, (up, c1, c2, ratio) in enumerate(self.blocks):
            kwu = dict(y=b[f"u{i}"], y_act=b[f"u{i}a"], y32=b.get(f"u{i}32"))
            kw1 = dict(y=None, y_act=b[f"h{i}"])
            kw2 = dict(y=None, res=b[f"u{i}"], y_act=b[f"r{i}"])
            if tune:
                up.tune(x_act, **kwu); c1.tune(b[f"u{i}a"], **kw1); c2.tune(b[f"h{i}"], **kw2)
            up(x_act, **kwu)
            c1(b[f"u{i}a"], **kw1)
            c2(b[f"h{i}"], **kw2)
            x_act = b[f"r{i}"]
        conv_last(x_act, self.wl, self.bl, b["out"])
        return b["out"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b = self._buffers(x.shape[0], x.device)
        return self._run(b, x)
