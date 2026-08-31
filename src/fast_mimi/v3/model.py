"""Pure-PyTorch re-implementation of the Kyutai Mimi codec (numerically matching transformers.MimiModel).

This is the *reference* implementation of fast-mimi: every optimized backend (Triton kernels, CUDA graphs,
low precision) is validated against this module, and this module is validated against transformers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ----------------------------------------------------------------------------- config

@dataclass(frozen=True)
class MimiConfig:
    sampling_rate: int = 24000
    frame_rate: float = 12.5
    audio_channels: int = 1
    num_filters: int = 64
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    compress: int = 2
    dilation_growth_rate: int = 2
    num_residual_layers: int = 1
    upsampling_ratios: tuple[int, ...] = (8, 6, 5, 4)
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    head_dim: int = 64
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 250
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_quantizers: int = 32
    num_semantic_quantizers: int = 1
    upsample_groups: int = 512
    codebook_eps: float = 1e-5

    @property
    def hop_length(self) -> int:  # samples per 12.5 Hz frame
        return int(math.prod(self.upsampling_ratios)) * 2  # 1920

    @property
    def encodec_hop(self) -> int:  # samples per 25 Hz transformer token
        return int(math.prod(self.upsampling_ratios))  # 960


# ----------------------------------------------------------------------------- weights

def _resolve_checkpoint(repo_or_path: str) -> tuple[str, str]:
    import os
    if os.path.isdir(repo_or_path):
        return os.path.join(repo_or_path, "model.safetensors"), os.path.join(repo_or_path, "config.json")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_or_path, "model.safetensors"), hf_hub_download(repo_or_path, "config.json")


def load_mimi_state(repo_or_path: str = "kyutai/mimi", device="cuda") -> dict[str, Tensor]:
    """Load the raw fp32 safetensors state dict of a Mimi checkpoint."""
    from safetensors.torch import load_file
    weights_path, _ = _resolve_checkpoint(repo_or_path)
    return {k: v.to(device) for k, v in load_file(weights_path).items()}


# ----------------------------------------------------------------------------- primitives

def _extra_right_pad(length: int, k_eff: int, stride: int) -> int:
    """transformers' `_get_extra_padding_for_conv1d` for a causal conv (padding_total = k_eff - stride)."""
    padding_total = k_eff - stride
    n_frames = math.ceil((length - k_eff + padding_total) / stride + 1) - 1
    ideal = n_frames * stride + k_eff - padding_total
    return ideal - length


def causal_conv1d(x: Tensor, w: Tensor, b: Tensor | None, stride: int = 1, dilation: int = 1,
                  pad_mode: str = "constant") -> Tensor:
    k = w.shape[-1]
    k_eff = (k - 1) * dilation + 1
    left = k_eff - stride
    right = _extra_right_pad(x.shape[-1], k_eff, stride)
    if left or right:
        x = F.pad(x, (left, right), mode=pad_mode)
    return F.conv1d(x, w, b, stride=stride, dilation=dilation)


def causal_conv_transpose1d(x: Tensor, w: Tensor, b: Tensor | None, stride: int, groups: int = 1) -> Tensor:
    k = w.shape[-1]
    y = F.conv_transpose1d(x, w, b, stride=stride, groups=groups)
    return y[..., : y.shape[-1] - (k - stride)]  # causal: trim everything on the right


def elu(x: Tensor) -> Tensor:
    return F.elu(x)


# ----------------------------------------------------------------------------- model

class FastMimi(torch.nn.Module):
    """Mimi codec with the same numerics as transformers' MimiModel (no KV-cache / non-streaming path).

    Notes on fidelity: transformers (4.48) does not pass an attention mask to the encoder/decoder transformer,
    so with the default SDPA attention the transformer is *causal without sliding window*. We reproduce exactly
    that. Set `sliding_window=True` to get the behaviour described in the Mimi paper (window of 250 tokens).
    """

    def __init__(self, state: dict[str, Tensor], config: MimiConfig | None = None, dtype: torch.dtype = torch.float32,
                 sliding_window: bool = False, attn_impl: str = "sdpa"):
        super().__init__()
        self.cfg = cfg = config or MimiConfig()
        self.dtype = dtype
        self.sliding_window = sliding_window
        self.attn_impl = attn_impl
        dev = next(iter(state.values())).device
        self.device = dev

        def g(name):
            return state[name].to(dtype)

        # --- SEANet encoder: (kind, params)
        enc = []
        enc.append(("conv", g("encoder.layers.0.conv.weight"), g("encoder.layers.0.conv.bias"), 1, 1))
        idx = 1
        for ratio in reversed(cfg.upsampling_ratios):
            enc.append(("res", g(f"encoder.layers.{idx}.block.1.conv.weight"), g(f"encoder.layers.{idx}.block.1.conv.bias"),
                        g(f"encoder.layers.{idx}.block.3.conv.weight"), g(f"encoder.layers.{idx}.block.3.conv.bias")))
            idx += 1
            enc.append(("elu",))
            idx += 1
            enc.append(("conv", g(f"encoder.layers.{idx}.conv.weight"), g(f"encoder.layers.{idx}.conv.bias"), ratio, 1))
            idx += 1
        enc.append(("elu",))
        idx += 1
        enc.append(("conv", g(f"encoder.layers.{idx}.conv.weight"), g(f"encoder.layers.{idx}.conv.bias"), 1, 1))
        self.enc_layers = enc

        # --- SEANet decoder
        dec = []
        dec.append(("conv", g("decoder.layers.0.conv.weight"), g("decoder.layers.0.conv.bias"), 1, 1))
        idx = 1
        for ratio in cfg.upsampling_ratios:
            dec.append(("elu",))
            idx += 1
            dec.append(("convT", g(f"decoder.layers.{idx}.conv.weight"), g(f"decoder.layers.{idx}.conv.bias"), ratio))
            idx += 1
            dec.append(("res", g(f"decoder.layers.{idx}.block.1.conv.weight"), g(f"decoder.layers.{idx}.block.1.conv.bias"),
                        g(f"decoder.layers.{idx}.block.3.conv.weight"), g(f"decoder.layers.{idx}.block.3.conv.bias")))
            idx += 1
        dec.append(("elu",))
        idx += 1
        dec.append(("conv", g(f"decoder.layers.{idx}.conv.weight"), g(f"decoder.layers.{idx}.conv.bias"), 1, 1))
        self.dec_layers = dec

        # --- transformers
        def tf_layers(prefix):
            layers = []
            for i in range(cfg.num_hidden_layers):
                p = f"{prefix}.layers.{i}."
                layers.append(dict(
                    ln1_w=g(p + "input_layernorm.weight"), ln1_b=g(p + "input_layernorm.bias"),
                    wq=g(p + "self_attn.q_proj.weight"), wk=g(p + "self_attn.k_proj.weight"),
                    wv=g(p + "self_attn.v_proj.weight"), wo=g(p + "self_attn.o_proj.weight"),
                    ls1=g(p + "self_attn_layer_scale.scale"),
                    ln2_w=g(p + "post_attention_layernorm.weight"), ln2_b=g(p + "post_attention_layernorm.bias"),
                    w1=g(p + "mlp.fc1.weight"), w2=g(p + "mlp.fc2.weight"),
                    ls2=g(p + "mlp_layer_scale.scale"),
                ))
            return layers
        self.enc_tf = tf_layers("encoder_transformer")
        self.dec_tf = tf_layers("decoder_transformer")
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=dev, dtype=torch.int64).float() / cfg.head_dim))
        self.inv_freq = inv_freq

        # --- down/up sampling between 25 Hz and 12.5 Hz
        self.down_w = g("downsample.conv.weight")          # [512, 512, 4], stride 2, replicate pad
        self.up_w = g("upsample.conv.weight")              # [512, 1, 4], stride 2, groups 512

        # --- quantizer
        def codebook(p):
            embed_sum = state[p + "embed_sum"]
            usage = state[p + "cluster_usage"]
            return (embed_sum / usage.clamp(min=cfg.codebook_eps)[:, None]).to(dtype)
        self.sem_in = g("quantizer.semantic_residual_vector_quantizer.input_proj.weight")[:, :, 0]   # [256, 512]
        self.sem_out = g("quantizer.semantic_residual_vector_quantizer.output_proj.weight")[:, :, 0] # [512, 256]
        self.ac_in = g("quantizer.acoustic_residual_vector_quantizer.input_proj.weight")[:, :, 0]
        self.ac_out = g("quantizer.acoustic_residual_vector_quantizer.output_proj.weight")[:, :, 0]
        self.sem_cb = codebook("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.")   # [2048, 256]
        self.ac_cb = torch.stack([codebook(f"quantizer.acoustic_residual_vector_quantizer.layers.{i}.codebook.")
                                  for i in range(cfg.num_quantizers - cfg.num_semantic_quantizers)])  # [31, 2048, 256]

    # ------------------------------------------------------------------ SEANet
    def _seanet(self, x: Tensor, layers) -> Tensor:
        for layer in layers:
            kind = layer[0]
            if kind == "conv":
                _, w, b, stride, dilation = layer
                x = causal_conv1d(x, w, b, stride=stride, dilation=dilation)
            elif kind == "convT":
                _, w, b, stride = layer
                x = causal_conv_transpose1d(x, w, b, stride=stride)
            elif kind == "res":
                _, w1, b1, w2, b2 = layer
                h = causal_conv1d(elu(x), w1, b1, dilation=1)
                h = causal_conv1d(elu(h), w2, b2)
                x = x + h
            elif kind == "elu":
                x = elu(x)
            else:
                raise ValueError(kind)
        return x

    # ------------------------------------------------------------------ transformer
    def _rope(self, T: int, dtype):
        pos = torch.arange(T, device=self.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)               # [T, 32]
        emb = torch.cat((freqs, freqs), dim=-1)                # [T, 64]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _transformer(self, x: Tensor, layers) -> Tensor:
        """x: [B, T, D]"""
        cfg = self.cfg
        B, T, D = x.shape
        H, hd = cfg.num_attention_heads, cfg.head_dim
        cos, sin = self._rope(T, x.dtype)
        mask = None
        if self.sliding_window:
            i = torch.arange(T, device=x.device)
            allowed = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - cfg.sliding_window)
            mask = allowed[None, None]
        for L in layers:
            h = F.layer_norm(x, (D,), L["ln1_w"], L["ln1_b"], cfg.norm_eps)
            q = (h @ L["wq"].T).view(B, T, H, hd).transpose(1, 2)
            k = (h @ L["wk"].T).view(B, T, H, hd).transpose(1, 2)
            v = (h @ L["wv"].T).view(B, T, H, hd).transpose(1, 2)
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin
            if self.attn_impl == "eager":
                att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(hd))
                causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril() if mask is None else mask
                att = att.masked_fill(~causal, float("-inf"))
                att = torch.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
                o = att @ v
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
            o = o.transpose(1, 2).reshape(B, T, D) @ L["wo"].T
            x = x + L["ls1"] * o
            h = F.layer_norm(x, (D,), L["ln2_w"], L["ln2_b"], cfg.norm_eps)
            h = F.gelu(h @ L["w1"].T) @ L["w2"].T
            x = x + L["ls2"] * h
        return x

    # ------------------------------------------------------------------ quantizer
    @staticmethod
    def _nearest(x: Tensor, cb: Tensor) -> Tensor:
        """x: [N, D] fp32, cb: [K, D] -> [N] argmin of euclidean distance (same formulation as torch.cdist)."""
        x = x.float()
        cb = cb.float()
        d = torch.cdist(x[None], cb[None], p=2)[0]
        return d.argmin(dim=-1)

    def _rvq_encode(self, emb: Tensor, num_quantizers: int) -> Tensor:
        """emb: [B, 512, T] -> codes [B, K, T]"""
        B, _, T = emb.shape
        e = emb.transpose(1, 2).reshape(B * T, -1)                     # [N, 512]
        xs = e @ self.sem_in.T                                          # [N, 256]
        codes = [self._nearest(xs, self.sem_cb)]
        if num_quantizers > 1:
            r = e @ self.ac_in.T
            for i in range(num_quantizers - 1):
                idx = self._nearest(r, self.ac_cb[i])
                codes.append(idx)
                r = r - self.ac_cb[i][idx]
        return torch.stack(codes, dim=1).view(B, T, -1).transpose(1, 2)  # [B, K, T]

    def _rvq_decode(self, codes: Tensor) -> Tensor:
        """codes: [B, K, T] -> [B, 512, T]"""
        B, K, T = codes.shape
        sem = F.embedding(codes[:, 0], self.sem_cb) @ self.sem_out.T       # [B, T, 512]
        if K > 1:
            ac = torch.zeros(B, T, self.cfg.codebook_dim, device=codes.device, dtype=self.sem_cb.dtype)
            for i in range(K - 1):
                ac = ac + F.embedding(codes[:, i + 1], self.ac_cb[i])
            sem = sem + ac @ self.ac_out.T
        return sem.transpose(1, 2)

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def encode(self, audio: Tensor, num_quantizers: int | None = None) -> Tensor:
        """audio: [B, 1, L] float -> codes [B, K, ceil(L/1920)] int64"""
        nq = num_quantizers or self.cfg.num_quantizers
        x = audio.to(self.dtype)
        x = self._seanet(x, self.enc_layers)                           # [B, 512, T25]
        x = self._transformer(x.transpose(1, 2), self.enc_tf).transpose(1, 2)
        x = causal_conv1d(x, self.down_w, None, stride=2, pad_mode="replicate")   # [B, 512, T12]
        return self._rvq_encode(x, nq)

    @torch.no_grad()
    def decode(self, codes: Tensor, length: int | None = None) -> Tensor:
        """codes: [B, K, T] -> audio [B, 1, T*1920] (truncated to `length` if given)"""
        x = self._rvq_decode(codes)                                    # [B, 512, T]
        x = causal_conv_transpose1d(x, self.up_w, None, stride=2, groups=self.cfg.upsample_groups)  # [B, 512, 2T]
        x = self._transformer(x.transpose(1, 2), self.dec_tf).transpose(1, 2)
        x = self._seanet(x, self.dec_layers)
        if length is not None:
            x = x[..., :length]
        return x
