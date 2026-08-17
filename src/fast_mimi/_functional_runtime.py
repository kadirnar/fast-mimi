"""Functional PyTorch operations used by the optimized independent Mimi runtime.

Checkpoint loading and identity validation remain owned by ``MimiModel``. This
module reads its frozen tensors and implements the optimized offline graph with
PyTorch operations.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


class PureTorchMimi:
    """Functional offline operations backed by the independent model tensors."""

    hidden_size = 512
    intermediate_size = 2048
    num_heads = 8
    head_dim = 64
    sliding_window = 250
    norm_eps = 1e-5

    def __init__(self, model: Any) -> None:
        self.model = model
        self.state = model.state_dict(keep_vars=True)
        self._position_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mask_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._embeds: dict[str, torch.Tensor] = {}

    @staticmethod
    def _causal_conv1d(
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
        groups: int = 1,
    ) -> torch.Tensor:
        effective_kernel = (weight.shape[-1] - 1) * dilation + 1
        padding_total = effective_kernel - stride
        length = values.shape[-1]
        numerator = length - effective_kernel + padding_total
        frame_count = (numerator + stride - 1) // stride
        ideal_length = frame_count * stride + effective_kernel - padding_total
        extra_padding = ideal_length - length
        if pad_mode == "constant":
            values = F.pad(values, (padding_total, extra_padding))
        else:
            values = F.pad(values, (padding_total, extra_padding), mode=pad_mode)
        return F.conv1d(
            values,
            weight,
            bias,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )

    @staticmethod
    def _causal_conv_transpose1d(
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        stride: int,
        groups: int = 1,
    ) -> torch.Tensor:
        values = F.conv_transpose1d(
            values,
            weight,
            bias,
            stride=stride,
            groups=groups,
        )
        padding_right = weight.shape[-1] - stride
        if padding_right:
            values = values[..., :-padding_right]
        return values

    def _conv(
        self,
        values: torch.Tensor,
        name: str,
        *,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ) -> torch.Tensor:
        return self._causal_conv1d(
            values,
            self.state[f"{name}.weight"],
            self.state.get(f"{name}.bias"),
            stride=stride,
            dilation=dilation,
            pad_mode=pad_mode,
        )

    def _conv_transpose(
        self,
        values: torch.Tensor,
        name: str,
        *,
        stride: int,
        groups: int = 1,
    ) -> torch.Tensor:
        return self._causal_conv_transpose1d(
            values,
            self.state[f"{name}.weight"],
            self.state.get(f"{name}.bias"),
            stride=stride,
            groups=groups,
        )

    def _residual(self, values: torch.Tensor, name: str) -> torch.Tensor:
        hidden = self._elu(values)
        hidden = self._conv(hidden, f"{name}.block.1.conv")
        hidden = self._elu(hidden)
        hidden = self._conv(hidden, f"{name}.block.3.conv")
        return self._add(values, hidden)

    @staticmethod
    def _elu(values: torch.Tensor) -> torch.Tensor:
        return F.elu(values)

    @staticmethod
    def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right

    @staticmethod
    def _linear(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Linear dispatch point used by precision and backend experiments."""
        return F.linear(values, weight)

    def encoder(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self._conv(input_values, "encoder.layers.0.conv")
        hidden = self._residual(hidden, "encoder.layers.1")
        hidden = self._elu(hidden)
        hidden = self._conv(hidden, "encoder.layers.3.conv", stride=4)
        hidden = self._residual(hidden, "encoder.layers.4")
        hidden = self._elu(hidden)
        hidden = self._conv(hidden, "encoder.layers.6.conv", stride=5)
        hidden = self._residual(hidden, "encoder.layers.7")
        hidden = self._elu(hidden)
        hidden = self._conv(hidden, "encoder.layers.9.conv", stride=6)
        hidden = self._residual(hidden, "encoder.layers.10")
        hidden = self._elu(hidden)
        hidden = self._conv(hidden, "encoder.layers.12.conv", stride=8)
        hidden = self._elu(hidden)
        return self._conv(hidden, "encoder.layers.14.conv")

    def decoder(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = self._conv(embeddings, "decoder.layers.0.conv")
        hidden = self._elu(hidden)
        hidden = self._conv_transpose(hidden, "decoder.layers.2.conv", stride=8)
        hidden = self._residual(hidden, "decoder.layers.3")
        hidden = self._elu(hidden)
        hidden = self._conv_transpose(hidden, "decoder.layers.5.conv", stride=6)
        hidden = self._residual(hidden, "decoder.layers.6")
        hidden = self._elu(hidden)
        hidden = self._conv_transpose(hidden, "decoder.layers.8.conv", stride=5)
        hidden = self._residual(hidden, "decoder.layers.9")
        hidden = self._elu(hidden)
        hidden = self._conv_transpose(hidden, "decoder.layers.11.conv", stride=4)
        hidden = self._residual(hidden, "decoder.layers.12")
        hidden = self._elu(hidden)
        return self._conv(hidden, "decoder.layers.14.conv")

    def _position_embeddings(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (length, device, dtype)
        cached = self._position_cache.get(key)
        if cached is not None:
            return cached
        inv_freq = 1.0 / (
            10_000.0
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(
                    dtype=torch.float32
                )
                / self.head_dim
            )
        )
        inv_freq = inv_freq.to(device)
        position_ids = torch.arange(length, device=device).unsqueeze(0)
        inv_freq_expanded = inv_freq[None, :, None].expand(1, -1, 1)
        frequencies = (
            inv_freq_expanded.float() @ position_ids[:, None, :].float()
        ).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        result = (embedding.cos().to(dtype), embedding.sin().to(dtype))
        self._position_cache[key] = result
        return result

    def _attention_mask(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (length, device, dtype)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        position = torch.arange(length, device=device)
        query = position[:, None]
        key_position = position[None, :]
        allowed = (key_position <= query) & (key_position > query - self.sliding_window)
        mask = allowed[None, None]
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def transformer(self, hidden: torch.Tensor, prefix: str) -> torch.Tensor:
        length = hidden.shape[1]
        cosine, sine = self._position_embeddings(length, hidden.device, hidden.dtype)
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        mask = self._attention_mask(length, hidden.device, hidden.dtype)
        scaling = 1.0 / math.sqrt(self.head_dim)

        for layer_index in range(8):
            name = f"{prefix}.layers.{layer_index}"
            residual = hidden
            normalized = F.layer_norm(
                hidden,
                (self.hidden_size,),
                self.state[f"{name}.input_layernorm.weight"],
                self.state[f"{name}.input_layernorm.bias"],
                self.norm_eps,
            )
            query = self._linear(
                normalized, self.state[f"{name}.self_attn.q_proj.weight"]
            )
            key = self._linear(
                normalized, self.state[f"{name}.self_attn.k_proj.weight"]
            )
            value = self._linear(
                normalized, self.state[f"{name}.self_attn.v_proj.weight"]
            )
            query = query.view(1, length, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(1, length, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(1, length, self.num_heads, self.head_dim).transpose(1, 2)
            query = query * cosine + self._rotate_half(query) * sine
            key = key * cosine + self._rotate_half(key) * sine
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scaling,
            )
            attended = (
                attended.transpose(1, 2).contiguous().view(1, length, self.hidden_size)
            )
            attended = self._linear(
                attended, self.state[f"{name}.self_attn.o_proj.weight"]
            )
            hidden = (
                residual + attended * self.state[f"{name}.self_attn_layer_scale.scale"]
            )

            residual = hidden
            normalized = F.layer_norm(
                hidden,
                (self.hidden_size,),
                self.state[f"{name}.post_attention_layernorm.weight"],
                self.state[f"{name}.post_attention_layernorm.bias"],
                self.norm_eps,
            )
            mlp = self._linear(normalized, self.state[f"{name}.mlp.fc1.weight"])
            mlp = F.gelu(mlp)
            mlp = self._linear(mlp, self.state[f"{name}.mlp.fc2.weight"])
            hidden = residual + mlp * self.state[f"{name}.mlp_layer_scale.scale"]
        return hidden

    def downsample(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self._conv(
            embeddings,
            "downsample.conv",
            stride=2,
            pad_mode="replicate",
        )

    def upsample(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self._conv_transpose(
            embeddings,
            "upsample.conv",
            stride=2,
            groups=512,
        )

    def _embed(self, codebook: str) -> torch.Tensor:
        cached = self._embeds.get(codebook)
        if cached is not None:
            return cached
        usage = self.state[f"{codebook}.cluster_usage"].clamp(min=1e-5)
        embed = self.state[f"{codebook}.embed_sum"] / usage[:, None]
        self._embeds[codebook] = embed
        return embed

    def _rvq_encode(
        self,
        embeddings: torch.Tensor,
        prefix: str,
        count: int,
    ) -> torch.Tensor:
        projected = F.conv1d(
            embeddings,
            self.state[f"{prefix}.input_proj.weight"],
        )
        residual = projected
        indices = []
        for index in range(count):
            codebook = f"{prefix}.layers.{index}.codebook"
            embed = self._embed(codebook)
            flattened = residual.permute(0, 2, 1).reshape(-1, embed.shape[1])
            distances = torch.cdist(flattened[None].float(), embed[None].float(), p=2)[
                0
            ]
            selected = distances.argmin(dim=-1).view(
                residual.shape[0], residual.shape[-1]
            )
            quantized = F.embedding(selected, embed).permute(0, 2, 1)
            residual = residual - quantized
            indices.append(selected)
        return torch.stack(indices)

    def quantizer_encode(self, embeddings: torch.Tensor, count: int) -> torch.Tensor:
        semantic_prefix = "quantizer.semantic_residual_vector_quantizer"
        acoustic_prefix = "quantizer.acoustic_residual_vector_quantizer"
        codes = [self._rvq_encode(embeddings, semantic_prefix, 1)]
        if count > 1:
            codes.append(self._rvq_encode(embeddings, acoustic_prefix, count - 1))
        return torch.cat(codes, dim=0).transpose(0, 1)

    def _rvq_decode(
        self,
        codes: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        decoded = None
        for index in range(codes.shape[1]):
            embed = self._embed(f"{prefix}.layers.{index}.codebook")
            value = F.embedding(codes[:, index], embed).permute(0, 2, 1)
            decoded = value if decoded is None else decoded + value
        assert decoded is not None
        return F.conv1d(decoded, self.state[f"{prefix}.output_proj.weight"])

    def quantizer_decode(self, codes: torch.Tensor) -> torch.Tensor:
        semantic = self._rvq_decode(
            codes[:, :1],
            "quantizer.semantic_residual_vector_quantizer",
        )
        if codes.shape[1] == 1:
            return semantic
        acoustic = self._rvq_decode(
            codes[:, 1:],
            "quantizer.acoustic_residual_vector_quantizer",
        )
        return semantic + acoustic

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
        num_quantizers: int,
    ) -> SimpleNamespace:
        embeddings = self.encoder(input_values)
        embeddings = self.transformer(embeddings.transpose(1, 2), "encoder_transformer")
        embeddings = self.downsample(embeddings.transpose(1, 2))
        codes = self.quantizer_encode(embeddings, num_quantizers)

        decoded = self.quantizer_decode(codes)
        decoded = self.upsample(decoded)
        decoded = self.transformer(decoded.transpose(1, 2), "decoder_transformer")
        audio_values = self.decoder(decoded.transpose(1, 2))
        if padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]
        return SimpleNamespace(
            audio_codes=codes,
            audio_values=audio_values,
            encoder_past_key_values=None,
            decoder_past_key_values=None,
        )
