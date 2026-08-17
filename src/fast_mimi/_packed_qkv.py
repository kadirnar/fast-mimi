"""Bit-exact packed QKV projection and RoPE for the optimized Mimi path."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _packed_qkv_rope_kernel(
    packed_pointer,
    cosine_pointer,
    sine_pointer,
    query_pointer,
    key_pointer,
    value_pointer,
    length: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < count
    dimension = offset % head_dim
    token_head = offset // head_dim
    token = token_head % length
    head = token_head // length
    feature = head * head_dim + dimension
    hidden = heads * head_dim
    packed_row = token * hidden * 3
    rotated_dimension = tl.where(
        dimension < head_dim // 2,
        dimension + head_dim // 2,
        dimension - head_dim // 2,
    )
    rotated_feature = head * head_dim + rotated_dimension
    query = tl.load(
        packed_pointer + packed_row + feature,
        mask=valid,
        other=0.0,
    )
    key = tl.load(
        packed_pointer + packed_row + hidden + feature,
        mask=valid,
        other=0.0,
    )
    value = tl.load(
        packed_pointer + packed_row + 2 * hidden + feature,
        mask=valid,
        other=0.0,
    )
    rotated_query = tl.load(
        packed_pointer + packed_row + rotated_feature,
        mask=valid,
        other=0.0,
    )
    rotated_key = tl.load(
        packed_pointer + packed_row + hidden + rotated_feature,
        mask=valid,
        other=0.0,
    )
    sign = tl.where(dimension < head_dim // 2, -1.0, 1.0)
    rotated_query *= sign
    rotated_key *= sign
    cosine = tl.load(
        cosine_pointer + token * head_dim + dimension,
        mask=valid,
        other=0.0,
    )
    sine = tl.load(
        sine_pointer + token * head_dim + dimension,
        mask=valid,
        other=0.0,
    )
    query = libdevice.add_rn(
        libdevice.mul_rn(query, cosine),
        libdevice.mul_rn(rotated_query, sine),
    )
    key = libdevice.add_rn(
        libdevice.mul_rn(key, cosine),
        libdevice.mul_rn(rotated_key, sine),
    )
    output_offset = token * hidden + feature
    tl.store(query_pointer + output_offset, query, mask=valid)
    tl.store(key_pointer + output_offset, key, mask=valid)
    tl.store(value_pointer + output_offset, value, mask=valid)


def packed_qkv_rope(
    packed: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = packed.shape[1]
    hidden = heads * head_dim
    query_nlc = torch.empty(
        (1, length, hidden),
        device=packed.device,
        dtype=packed.dtype,
    )
    key_nlc = torch.empty_like(query_nlc)
    value_nlc = torch.empty_like(query_nlc)
    count = length * hidden
    block = 256
    _packed_qkv_rope_kernel[(triton.cdiv(count, block),)](
        packed,
        cosine,
        sine,
        query_nlc,
        key_nlc,
        value_nlc,
        length,
        heads,
        head_dim,
        count,
        BLOCK=block,
        num_warps=4,
    )
    return (
        query_nlc.view(1, length, heads, head_dim).transpose(1, 2),
        key_nlc.view(1, length, heads, head_dim).transpose(1, 2),
        value_nlc.view(1, length, heads, head_dim).transpose(1, 2),
    )


class PackedQkvTransformer:
    """Run the frozen transformer with one QKV GEMM per layer."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.weights = {
            f"{prefix}.layers.{index}": torch.cat(
                (
                    runtime.state[f"{prefix}.layers.{index}.self_attn.q_proj.weight"],
                    runtime.state[f"{prefix}.layers.{index}.self_attn.k_proj.weight"],
                    runtime.state[f"{prefix}.layers.{index}.self_attn.v_proj.weight"],
                ),
                dim=0,
            ).contiguous()
            for prefix in ("encoder_transformer", "decoder_transformer")
            for index in range(8)
        }

    def __call__(self, hidden: torch.Tensor, prefix: str) -> torch.Tensor:
        runtime = self.runtime
        length = hidden.shape[1]
        cosine, sine = runtime._position_embeddings(
            length,
            hidden.device,
            hidden.dtype,
        )
        for layer_index in range(8):
            name = f"{prefix}.layers.{layer_index}"
            residual = hidden
            normalized = F.layer_norm(
                hidden,
                (runtime.hidden_size,),
                runtime.state[f"{name}.input_layernorm.weight"],
                runtime.state[f"{name}.input_layernorm.bias"],
                runtime.norm_eps,
            )
            packed = F.linear(normalized, self.weights[name])
            query, key, value = packed_qkv_rope(
                packed,
                cosine,
                sine,
                runtime.num_heads,
                runtime.head_dim,
            )
            attended = runtime._window_attention_nlc(query, key, value)
            attended = F.linear(
                attended,
                runtime.state[f"{name}.self_attn.o_proj.weight"],
            )
            hidden = (
                residual
                + attended * runtime.state[f"{name}.self_attn_layer_scale.scale"]
            )
            residual = hidden
            normalized = F.layer_norm(
                hidden,
                (runtime.hidden_size,),
                runtime.state[f"{name}.post_attention_layernorm.weight"],
                runtime.state[f"{name}.post_attention_layernorm.bias"],
                runtime.norm_eps,
            )
            mlp = F.linear(
                normalized,
                runtime.state[f"{name}.mlp.fc1.weight"],
            )
            mlp = F.gelu(mlp)
            mlp = F.linear(mlp, runtime.state[f"{name}.mlp.fc2.weight"])
            hidden = residual + mlp * runtime.state[f"{name}.mlp_layer_scale.scale"]
        return hidden
