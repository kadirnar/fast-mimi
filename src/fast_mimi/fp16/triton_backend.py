"""fast-mimi Triton backend: SEANet (implicit-GEMM convs) + fused transformer + two-stage RVQ, bf16 tensor cores,
fp32 residual stream. Batch size 1 (the latency-critical case); see FastMimi for the generic reference."""
from __future__ import annotations
import math
import torch
from .model import MimiConfig, FastMimi
from .kernels.seanet import TritonSEANetEncoder, TritonSEANetDecoder
from .kernels.transformer import TritonTransformer
from .kernels.rvq3 import RVQEncoder3
from .kernels.rvq_ml import RVQEncoderML
from .kernels.glue import DownProj, RVQDecodeUp


class TritonMimi:
    def __init__(self, state, dtype=torch.bfloat16, variant="triton", tune=True, config: MimiConfig | None = None,
                 rvq_coarse="fp16", **kw):
        """variant: 'triton' (bf16 weights, default) | 'triton-int8' (INT8 weights: no faster here, lower fidelity).
        rvq_coarse: 'fp16' (exact two-stage search, default) | 'int8' (~20 % faster RVQ, exact in tests but no margin)."""
        self.cfg = cfg = config or MimiConfig()
        wdtype = "int8" if "int8" in variant else "bf16"
        if "rvqi8" in variant:
            rvq_coarse = "int8"
        ref = FastMimi(state, cfg, dtype=torch.float32)          # for weights in a convenient form
        self.ref = ref
        self.enc_seanet = TritonSEANetEncoder(state, cfg, tune=tune, wdtype=wdtype)
        self.dec_seanet = TritonSEANetDecoder(state, cfg, tune=tune, wdtype=wdtype)
        tf_kw = dict(bm=16, attn="batched", wdtype=wdtype, max_bt=64, window=cfg.sliding_window)
        self.enc_tf = TritonTransformer(ref.enc_tf, **tf_kw)
        self.dec_tf = TritonTransformer(ref.dec_tf, **tf_kw)
        self.down_proj = DownProj(ref.down_w, ref.sem_in, ref.ac_in)
        self.rvq = RVQEncoder3(ref.sem_cb, ref.ac_cb, block_e=64, num_warps=8, coarse=rvq_coarse, ncand=2 if rvq_coarse == "fp16" else 4, max_groups=2)
        self.rvq_ml = RVQEncoderML(ref.sem_cb, ref.ac_cb)                       # for frame counts too large to keep co-resident
        self.rvq_persistent_max_frames = 32
        self.rvq_dec = RVQDecodeUp(ref.sem_cb, ref.ac_cb, ref.sem_out, ref.ac_out, ref.up_w)
        self.device = ref.device
        self._buf = {}

    def _bufs(self, key, **shapes):
        if key not in self._buf:
            self._buf[key] = {k: torch.empty(*v[0], dtype=v[1], device=self.device) for k, v in shapes.items()}
        return self._buf[key]

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        assert audio.shape[0] == 1, "batch size 1 only"
        L = audio.shape[-1]
        T25 = math.ceil(L / 960)
        T12 = math.ceil(T25 / 2)
        b = self._bufs(("enc", L), x=((T25, 512), torch.float32), proj=((T12, 512), torch.float32))
        feats = self.enc_seanet(audio)                       # [T25, 512] fp32
        b["x"].copy_(feats)
        self.enc_tf.forward(b["x"])
        self.down_proj(b["x"], b["proj"])                     # [T12, 512] = [sem 256 | ac 256]
        xs = b["proj"][:, :256]
        xa = b["proj"][:, 256:]
        rvq = self.rvq if T12 <= self.rvq_persistent_max_frames else self.rvq_ml
        return rvq.encode(xs.contiguous(), xa.contiguous(), 1, T12, num_quantizers)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, length: int | None = None) -> torch.Tensor:
        assert codes.shape[0] == 1
        K, T = codes.shape[1], codes.shape[2]
        b = self._bufs(("dec", T), x=((2 * T, 512), torch.float32))
        self.rvq_dec(codes[0].contiguous(), b["x"])
        self.dec_tf.forward(b["x"])
        audio = self.dec_seanet(b["x"])                        # [2T*960] fp32
        audio = audio.view(1, 1, -1)
        if length is not None:
            audio = audio[..., :length]
        return audio
