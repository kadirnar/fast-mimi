"""Accepted SM120 long-form optimizations for the independent Mimi runtime.

This module contains only the quality-gated production path: exact functional
PyTorch operations, selected Triton memory kernels, cuDNN frontend plan
selection, CUDA Graph replay, and the native CUTLASS decoder-tail kernel.  It
has no Transformers dependency and fails closed when its measured contract is
not available.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

try:
    import cudnn
except ImportError:  # pragma: no cover - optional optimized extra.
    cudnn = None

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ._functional_runtime import PureTorchMimi
from ._native_cutlass import CutlassBiasDgrad, load_cutlass_bias_dgrad
from ._native_decoder9_fused import FusedDecoder9Wmma, load_fused_decoder9_wmma
from ._native_decoder12_final import (
    Decoder12FinalWmma,
    load_decoder12_final_wmma,
)
from ._native_final_post import NativeFinalPost, load_native_final_post
from ._packed_qkv import PackedQkvTransformer


@triton.jit
def _add_elu_half_kernel(
    residual_pointer,
    branch_pointer,
    output_pointer,
    count: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    valid = offset < count
    value = tl.load(residual_pointer + offset, mask=valid, other=0.0)
    value += tl.load(branch_pointer + offset, mask=valid, other=0.0)
    value = tl.where(value > 0.0, value, libdevice.expm1(value))
    tl.store(output_pointer + offset, value, mask=valid)


def _triton_add_elu_half_into(
    residual: torch.Tensor,
    branch: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if (
        residual.shape != branch.shape
        or residual.shape != output.shape
        or residual.dtype != torch.float32
        or branch.dtype != torch.float32
        or output.dtype != torch.float16
        or residual.stride() != branch.stride()
        or residual.stride() != output.stride()
    ):
        raise RuntimeError("unexpected fused decoder pre-activation layout")
    block = 256
    _add_elu_half_kernel[(triton.cdiv(output.numel(), block),)](
        residual,
        branch,
        output,
        output.numel(),
        block,
        num_warps=4,
    )
    return output


@triton.jit
def _rope_kernel(
    query_pointer,
    key_pointer,
    cosine_pointer,
    sine_pointer,
    query_output_pointer,
    key_output_pointer,
    length: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    count: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    dimension = offset % head_dim
    token_head = offset // head_dim
    token = token_head % length
    head = token_head // length
    source = token * heads * head_dim + head * head_dim + dimension
    rotated_dimension = tl.where(
        dimension < head_dim // 2,
        dimension + head_dim // 2,
        dimension - head_dim // 2,
    )
    rotated_source = token * heads * head_dim + head * head_dim + rotated_dimension
    valid = offset < count
    query = tl.load(query_pointer + source, mask=valid, other=0.0)
    key = tl.load(key_pointer + source, mask=valid, other=0.0)
    rotated_query = tl.load(query_pointer + rotated_source, mask=valid, other=0.0)
    rotated_key = tl.load(key_pointer + rotated_source, mask=valid, other=0.0)
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
    query_result = libdevice.add_rn(
        libdevice.mul_rn(query, cosine),
        libdevice.mul_rn(rotated_query, sine),
    )
    key_result = libdevice.add_rn(
        libdevice.mul_rn(key, cosine),
        libdevice.mul_rn(rotated_key, sine),
    )
    tl.store(query_output_pointer + source, query_result, mask=valid)
    tl.store(key_output_pointer + source, key_result, mask=valid)


def _triton_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = query.shape[1]
    heads = 8
    head_dim = 64
    count = heads * length * head_dim
    query_layout = query.view(1, length, heads, head_dim).transpose(1, 2)
    key_layout = key.view(1, length, heads, head_dim).transpose(1, 2)
    query_output = torch.empty_like(query_layout)
    key_output = torch.empty_like(key_layout)
    block = 256
    _rope_kernel[(triton.cdiv(count, block),)](
        query.contiguous(),
        key.contiguous(),
        cosine.contiguous(),
        sine.contiguous(),
        query_output,
        key_output,
        length,
        heads,
        head_dim,
        count,
        block,
        num_warps=4,
    )
    return query_output, key_output


class CompiledSeanetsMimi(PureTorchMimi):
    """Compile only the exact SEANet encoder and decoder into Inductor/Triton graphs."""

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self._compiled_encoder = torch.compile(
            self._encoder_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_decoder = torch.compile(
            self._decoder_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )

    def _encoder_graph(self, input_values: torch.Tensor) -> torch.Tensor:
        return PureTorchMimi.encoder(self, input_values)

    def _decoder_graph(self, embeddings: torch.Tensor) -> torch.Tensor:
        return PureTorchMimi.decoder(self, embeddings)

    def encoder(self, input_values: torch.Tensor) -> torch.Tensor:
        return self._compiled_encoder(input_values)

    def decoder(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self._compiled_decoder(embeddings)


@triton.jit
def _attention_pack_nlc_kernel(
    early_pointer,
    middle_pointer,
    tail_pointer,
    output_pointer,
    length: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
    early_span: tl.constexpr,
    tail_start: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack the three FMHA batches directly into contiguous (batch, time, channels)."""
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < length * heads * width
    feature = indices % width
    row = indices // width
    head = row % heads
    time = row // heads

    early_block = time // block
    early_row = time % block
    early_offset = ((early_block * block + early_row) * heads + head) * width + feature

    middle_time = time - early_span
    middle_block = middle_time // block
    middle_row = middle_time % block
    middle_offset = (
        (middle_block * block + middle_row) * heads + head
    ) * width + feature

    tail_row = time - tail_start
    tail_offset = (tail_row * heads + head) * width + feature

    value = tl.load(
        early_pointer + early_offset,
        mask=valid & (time < early_span),
        other=0.0,
    )
    value = tl.where(
        (time >= early_span) & (time < tail_start),
        tl.load(
            middle_pointer + middle_offset,
            mask=valid & (time >= early_span) & (time < tail_start),
            other=0.0,
        ),
        value,
    )
    value = tl.where(
        time >= tail_start,
        tl.load(
            tail_pointer + tail_offset,
            mask=valid & (time >= tail_start),
            other=0.0,
        ),
        value,
    )
    tl.store(output_pointer + indices, value, mask=valid)


def _channels_last_stride(dimensions: list[int]) -> list[int]:
    _, channels, height, width = dimensions
    return [channels * height * width, 1, channels * width, channels]


class _CudnnHalfInputConv:
    """Fuse FP32 ELU, FP16 tensor-core convolution, and FP32 bias/output."""

    def __init__(
        self,
        example: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        input_dtype: torch.dtype = torch.float16,
        weight_dtype: torch.dtype = torch.float16,
        output_elu_half: bool = False,
    ) -> None:
        if cudnn is None:
            raise RuntimeError("cuDNN frontend is unavailable")
        self.handle = cudnn.create_handle()
        dtype_map = {
            torch.float16: cudnn.data_type.HALF,
            torch.bfloat16: cudnn.data_type.BFLOAT16,
            torch.float32: cudnn.data_type.FLOAT,
        }
        if input_dtype not in dtype_map or weight_dtype not in dtype_map:
            raise RuntimeError("unsupported cuDNN mixed convolution dtype")
        self.weight = weight.to(weight_dtype).contiguous(
            memory_format=torch.channels_last
        )
        self.bias = bias.reshape(1, -1, 1, 1)
        output_dtype = torch.float16 if output_elu_half else torch.float32
        self.output = torch.empty(
            (example.shape[0], self.weight.shape[0], 1, example.shape[-1]),
            device=example.device,
            dtype=output_dtype,
        ).contiguous(memory_format=torch.channels_last)

        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        self.graph = graph
        self.input_tensor = graph.tensor_like(example, name="input")
        self.weight_tensor = graph.tensor_like(self.weight, name="weight")
        self.bias_tensor = graph.tensor_like(self.bias, name="bias")
        activated = graph.elu(
            input=self.input_tensor,
            name="elu",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        activated.set_data_type(dtype_map[input_dtype])
        kernel = self.weight.shape[-1]
        convolved = graph.conv_fprop(
            image=activated,
            weight=self.weight_tensor,
            pre_padding=[0, kernel - 1],
            post_padding=[0, 0],
            stride=[1, 1],
            dilation=[1, 1],
            name="convolution",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        dimensions = list(self.output.shape)
        stride = _channels_last_stride(dimensions)
        convolved.set_dim(dimensions).set_stride(stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        result = graph.bias(
            input=convolved,
            bias=self.bias_tensor,
            name="bias_add",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        result.set_dim(dimensions).set_stride(stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        if output_elu_half:
            result = graph.elu(
                input=result,
                name="output_elu",
                compute_data_type=cudnn.data_type.FLOAT,
            )
            result.set_dim(dimensions).set_stride(stride).set_data_type(
                cudnn.data_type.HALF
            )
        result.set_output(True)
        self.output_tensor = result

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK]
        )
        plans: list[tuple[int, torch.Tensor]] = []
        for index in range(graph.get_execution_plan_count()):
            try:
                graph.build_plan_at_index(index)
                workspace = torch.empty(
                    graph.get_workspace_size(),
                    device=example.device,
                    dtype=torch.uint8,
                )
                plans.append((index, workspace))
            except Exception:
                continue
        if not plans:
            raise RuntimeError("cuDNN found no executable convolution plan")
        self.plan = self._autotune(example, plans)

    def _select(self, plan: tuple[int, torch.Tensor]) -> None:
        self.graph.build_plan_at_index(plan[0])
        self.plan = plan

    def _execute(self, values: torch.Tensor) -> torch.Tensor:
        cudnn.set_stream(
            handle=self.handle,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        self.graph.execute(
            {
                self.input_tensor: values,
                self.weight_tensor: self.weight,
                self.bias_tensor: self.bias,
                self.output_tensor: self.output,
            },
            self.plan[1],
            handle=self.handle,
        )
        return self.output

    def _autotune(
        self,
        values: torch.Tensor,
        plans: list[tuple[int, torch.Tensor]],
    ) -> tuple[int, torch.Tensor]:
        timings: list[tuple[float, tuple[int, torch.Tensor]]] = []
        repetitions = 7
        for plan in plans:
            self._select(plan)
            self._execute(values)
            self._execute(values)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                self._execute(values)
            end.record()
            end.synchronize()
            timings.append((start.elapsed_time(end) / repetitions, plan))
        selected = min(timings, key=lambda item: item[0])[1]
        self._select(selected)
        return selected

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return self._execute(values)


class _FusedDecoder9:
    """cuDNN half branch feeding an SM120 WMMA residual epilogue."""

    def __init__(
        self,
        native: FusedDecoder9Wmma,
        residual: torch.Tensor,
        branch_weight: torch.Tensor,
        branch_bias: torch.Tensor,
        pointwise_weight: torch.Tensor,
        pointwise_bias: torch.Tensor,
    ) -> None:
        if residual.shape[-1] != 600_000:
            raise RuntimeError("decoder-9 native path is validated for 100 seconds")
        self.native = native
        self.branch = _CudnnHalfInputConv(
            residual,
            branch_weight,
            branch_bias,
            output_elu_half=True,
        )
        self.weight = (
            pointwise_weight.squeeze(-1).squeeze(-1).to(torch.float16).contiguous()
        )
        self.bias = pointwise_bias.contiguous()
        self.output_matrix = torch.empty(
            (residual.shape[-1], residual.shape[1]),
            device=residual.device,
            dtype=torch.float16,
        )
        self.output = self.output_matrix.view(
            1, 1, residual.shape[-1], residual.shape[1]
        ).permute(0, 3, 1, 2)

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        branch = self.branch(residual)
        rows = residual.shape[-1]
        branch_matrix = branch.permute(0, 2, 3, 1).reshape(rows, 64)
        residual_matrix = residual.permute(0, 2, 3, 1).reshape(rows, 128)
        self.native(
            branch_matrix,
            self.weight,
            residual_matrix,
            self.bias,
            self.output_matrix,
        )
        return self.output


class _FusedDecoder12Final:
    """Fuse the validated 100-second decoder-12 branch and final convolution."""

    def __init__(
        self,
        native: Decoder12FinalWmma,
        residual: torch.Tensor,
        branch_weight: torch.Tensor,
        branch_bias: torch.Tensor,
        pointwise_weight: torch.Tensor,
        pointwise_bias: torch.Tensor,
        final_weight: torch.Tensor,
        final_bias: torch.Tensor,
    ) -> None:
        if residual.shape[-1] != 2_400_000:
            raise RuntimeError("decoder-12 fusion is validated for 100 seconds")
        self.native = native
        self.branch = _CudnnHalfInputConv(
            residual,
            branch_weight,
            branch_bias,
            output_elu_half=True,
        )
        self.pointwise_weight = (
            pointwise_weight.squeeze(-1).squeeze(-1).to(torch.float16).contiguous()
        )
        self.pointwise_bias = pointwise_bias.contiguous()
        self.final_weight = final_weight.contiguous()
        self.final_bias = final_bias.contiguous()
        self.output = torch.empty(
            (1, 1, residual.shape[-1]),
            device=residual.device,
            dtype=torch.float32,
        )

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        branch = self.branch(residual)
        length = residual.shape[-1]
        branch_matrix = branch.permute(0, 2, 3, 1).reshape(length, 32)
        residual_matrix = residual.permute(0, 2, 3, 1).reshape(length, 64)
        return self.native(
            branch_matrix,
            self.pointwise_weight,
            residual_matrix,
            self.pointwise_bias,
            self.final_weight,
            self.final_bias,
            self.output,
            config=3,
        )


class _CutlassDecoderLayer11:
    """Fuse residual+ELU+FP16 conversion with bias-aware CUTLASS dgrad."""

    def __init__(
        self,
        native: CutlassBiasDgrad,
        residual: torch.Tensor,
        branch: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        self.native = native
        self.input = torch.empty_like(
            residual,
            dtype=torch.float16,
            memory_format=torch.preserve_format,
        )
        self.weight = weight.to(torch.float16).contiguous(
            memory_format=torch.channels_last
        )
        self.bias = bias.contiguous()
        dimensions = (
            residual.shape[0],
            weight.shape[1],
            1,
            residual.shape[-1] * 4,
        )
        self.output = torch.empty(
            dimensions,
            device=residual.device,
            dtype=torch.float32,
        ).contiguous(memory_format=torch.channels_last)
        self.decoder9: _FusedDecoder9 | None = None
        self.decoder12_final: _FusedDecoder12Final | None = None
        self(residual, branch)

    def __call__(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        _triton_add_elu_half_into(residual, branch, self.input)
        return self.from_half(self.input)

    def from_half(self, values: torch.Tensor) -> torch.Tensor:
        return self.native(values, self.weight, self.bias, self.output, 4)


class _CudnnDeconv:
    """Autotune an FP32 cuDNN dgrad plan for a causal transposed convolution."""

    def __init__(
        self,
        example: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: int,
        *,
        require_exact: bool = False,
        quality_first: bool = False,
    ) -> None:
        if cudnn is None:
            raise RuntimeError("cuDNN frontend is unavailable")
        self.handle = cudnn.create_handle()
        self.weight = weight
        self.bias = bias.reshape(1, -1, 1, 1)
        self.output = torch.empty(
            (example.shape[0], weight.shape[1], 1, example.shape[-1] * stride),
            device=example.device,
            dtype=torch.float32,
        ).contiguous(memory_format=torch.channels_last)

        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        self.graph = graph
        self.input_tensor = graph.tensor_like(example, name="input")
        self.weight_tensor = graph.tensor_like(weight, name="weight")
        self.bias_tensor = graph.tensor_like(self.bias, name="bias")
        kernel = weight.shape[-1]
        deconvolved = graph.conv_dgrad(
            loss=self.input_tensor,
            filter=self.weight_tensor,
            pre_padding=[0, 0],
            post_padding=[0, kernel - stride],
            stride=[1, stride],
            dilation=[1, 1],
            out_dims=list(self.output.shape),
            name="deconvolution",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        dimensions = list(self.output.shape)
        output_stride = list(self.output.stride())
        deconvolved.set_dim(dimensions).set_stride(output_stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        result = graph.bias(
            input=deconvolved,
            bias=self.bias_tensor,
            name="bias_add",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        result.set_dim(dimensions).set_stride(output_stride).set_data_type(
            cudnn.data_type.FLOAT
        ).set_output(True)
        self.output_tensor = result

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK]
        )
        plans: list[tuple[int, torch.Tensor]] = []
        for index in range(graph.get_execution_plan_count()):
            try:
                graph.build_plan_at_index(index)
                workspace_size = graph.get_workspace_size()
                if workspace_size > 512 * 1024 * 1024:
                    continue
                plans.append(
                    (
                        index,
                        torch.empty(
                            workspace_size,
                            device=example.device,
                            dtype=torch.uint8,
                        ),
                    )
                )
            except Exception:
                continue
        if not plans:
            raise RuntimeError("cuDNN found no executable deconvolution plan")
        self.plans = tuple(plans)
        reference = None
        if require_exact or quality_first:
            reference = F.conv_transpose2d(
                example,
                weight,
                bias,
                stride=(1, stride),
            )[..., : self.output.shape[-1]]
        self.plan = self._autotune(
            example,
            plans,
            reference,
            quality_first=quality_first,
        )

    def _select(self, plan: tuple[int, torch.Tensor]) -> None:
        self.graph.build_plan_at_index(plan[0])
        self.plan = plan

    def _execute(self, values: torch.Tensor) -> torch.Tensor:
        cudnn.set_stream(
            handle=self.handle,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        self.graph.execute(
            {
                self.input_tensor: values,
                self.weight_tensor: self.weight,
                self.bias_tensor: self.bias,
                self.output_tensor: self.output,
            },
            self.plan[1],
            handle=self.handle,
        )
        return self.output

    def _autotune(
        self,
        values: torch.Tensor,
        plans: list[tuple[int, torch.Tensor]],
        reference: torch.Tensor | None,
        *,
        quality_first: bool,
    ) -> tuple[int, torch.Tensor]:
        timings: list[tuple[float, float, float, tuple[int, torch.Tensor]]] = []
        repetitions = 7
        for plan in plans:
            self._select(plan)
            self._execute(values)
            self._execute(values)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                self._execute(values)
            end.record()
            end.synchronize()
            maximum = 0.0
            mean = 0.0
            if reference is not None:
                delta = (self.output - reference).abs()
                maximum = float(delta.max().item())
                mean = float(delta.mean().item())
                if not quality_first and maximum != 0.0:
                    continue
            timings.append(
                (
                    maximum,
                    mean,
                    start.elapsed_time(end) / repetitions,
                    plan,
                )
            )
        if not timings:
            raise RuntimeError("cuDNN found no bit-exact deconvolution plan")
        if quality_first:
            selected_entry = min(
                timings,
                key=lambda item: (item[0], item[1], item[2]),
            )
        else:
            selected_entry = min(timings, key=lambda item: item[2])
        self.selection = {
            "plan_index": selected_entry[3][0],
            "max_abs_error": selected_entry[0],
            "mean_abs_error": selected_entry[1],
            "latency_ms": selected_entry[2],
        }
        selected = selected_entry[3]
        self._select(selected)
        return selected

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return self._execute(values)


class _CudnnFusedDecoderPost:
    """Fuse the final residual, ELU, causal padding, convolution, and bias."""

    def __init__(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if cudnn is None:
            raise RuntimeError("cuDNN frontend is unavailable")
        self.handle = cudnn.create_handle()
        self.weight = weight
        self.bias = bias.reshape(1, -1, 1, 1)
        output_dimensions = [
            residual.shape[0],
            weight.shape[0],
            1,
            residual.shape[-1],
        ]
        self.output = torch.empty_strided(
            output_dimensions,
            _channels_last_stride(output_dimensions),
            device=residual.device,
            dtype=torch.float32,
        )

        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        self.graph = graph
        self.residual_tensor = graph.tensor_like(residual, name="residual")
        self.branch_tensor = graph.tensor_like(branch, name="branch")
        self.weight_tensor = graph.tensor_like(weight, name="weight")
        self.bias_tensor = graph.tensor_like(self.bias, name="bias")
        hidden = graph.add(
            a=self.residual_tensor,
            b=self.branch_tensor,
            name="residual_add",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        dimensions = list(residual.shape)
        stride = _channels_last_stride(dimensions)
        hidden.set_dim(dimensions).set_stride(stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        hidden = graph.elu(
            input=hidden,
            name="elu",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        hidden.set_dim(dimensions).set_stride(stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        kernel = weight.shape[-1]
        convolved = graph.conv_fprop(
            image=hidden,
            weight=self.weight_tensor,
            pre_padding=[0, kernel - 1],
            post_padding=[0, 0],
            stride=[1, 1],
            dilation=[1, 1],
            name="final_convolution",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        output_stride = list(self.output.stride())
        convolved.set_dim(output_dimensions).set_stride(output_stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        result = graph.bias(
            input=convolved,
            bias=self.bias_tensor,
            name="bias_add",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        result.set_dim(output_dimensions).set_stride(output_stride).set_data_type(
            cudnn.data_type.FLOAT
        ).set_output(True)
        self.output_tensor = result

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK]
        )
        plans: list[tuple[int, torch.Tensor]] = []
        for index in range(graph.get_execution_plan_count()):
            try:
                graph.build_plan_at_index(index)
                workspace_size = graph.get_workspace_size()
                if workspace_size > 512 * 1024 * 1024:
                    continue
                plans.append(
                    (
                        index,
                        torch.empty(
                            workspace_size,
                            device=residual.device,
                            dtype=torch.uint8,
                        ),
                    )
                )
            except Exception:
                continue
        if not plans:
            raise RuntimeError("cuDNN found no executable fused decoder post plan")
        self.plan = self._autotune(residual, branch, plans, reference)

    def _select(self, plan: tuple[int, torch.Tensor]) -> None:
        self.graph.build_plan_at_index(plan[0])
        self.plan = plan

    def _execute(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        cudnn.set_stream(
            handle=self.handle,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        self.graph.execute(
            {
                self.residual_tensor: residual,
                self.branch_tensor: branch,
                self.weight_tensor: self.weight,
                self.bias_tensor: self.bias,
                self.output_tensor: self.output,
            },
            self.plan[1],
            handle=self.handle,
        )
        return self.output.squeeze(2)

    def _autotune(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        plans: list[tuple[int, torch.Tensor]],
        reference: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        timings: list[tuple[float, tuple[int, torch.Tensor]]] = []
        repetitions = 7
        for plan in plans:
            self._select(plan)
            self._execute(residual, branch)
            self._execute(residual, branch)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                self._execute(residual, branch)
            end.record()
            end.synchronize()
            if not torch.allclose(
                self.output.squeeze(2),
                reference,
                rtol=0.0,
                atol=5e-6,
            ):
                continue
            timings.append((start.elapsed_time(end) / repetitions, plan))
        if not timings:
            raise RuntimeError("cuDNN found no quality-safe fused decoder post plan")
        selected = min(timings, key=lambda item: item[0])[1]
        self._select(selected)
        return selected

    def __call__(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        return self._execute(residual, branch)


class _CudnnExactEncoderConv:
    """Autotune an FP32 encoder convolution, retaining only exact plans."""

    def __init__(
        self,
        example: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        stride: int = 1,
        reference: torch.Tensor | None = None,
        require_exact: bool = True,
    ) -> None:
        if cudnn is None:
            raise RuntimeError("cuDNN frontend is unavailable")
        self.handle = cudnn.create_handle()
        weight3 = weight.contiguous()
        self.weight = weight3.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        self.bias = bias.reshape(1, -1, 1, 1)
        kernel = weight.shape[-1]
        left = kernel - stride
        frames = (example.shape[-1] - kernel + left + stride - 1) // stride
        right = frames * stride + kernel - left - example.shape[-1]
        self.output = torch.empty(
            (example.shape[0], weight.shape[0], 1, frames + 1),
            device=example.device,
            dtype=torch.float32,
        ).contiguous(memory_format=torch.channels_last)

        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        self.graph = graph
        self.input_tensor = graph.tensor_like(example, name="input")
        self.weight_tensor = graph.tensor_like(self.weight, name="weight")
        self.bias_tensor = graph.tensor_like(self.bias, name="bias")
        convolved = graph.conv_fprop(
            image=self.input_tensor,
            weight=self.weight_tensor,
            pre_padding=[0, left],
            post_padding=[0, right],
            stride=[1, stride],
            dilation=[1, 1],
            name="convolution",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        dimensions = list(self.output.shape)
        output_stride = list(self.output.stride())
        convolved.set_dim(dimensions).set_stride(output_stride).set_data_type(
            cudnn.data_type.FLOAT
        )
        result = graph.bias(
            input=convolved,
            bias=self.bias_tensor,
            name="bias_add",
            compute_data_type=cudnn.data_type.FLOAT,
        )
        result.set_dim(dimensions).set_stride(output_stride).set_data_type(
            cudnn.data_type.FLOAT
        ).set_output(True)
        self.output_tensor = result

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK]
        )
        plans: list[tuple[int, torch.Tensor]] = []
        plan_specs: list[tuple[int, int]] = []
        for index in range(graph.get_execution_plan_count()):
            try:
                graph.build_plan_at_index(index)
                workspace_size = graph.get_workspace_size()
                if workspace_size > 512 * 1024 * 1024:
                    continue
                if require_exact:
                    plans.append(
                        (
                            index,
                            torch.empty(
                                workspace_size,
                                device=example.device,
                                dtype=torch.uint8,
                            ),
                        )
                    )
                else:
                    plan_specs.append((index, workspace_size))
            except Exception:
                continue
        if require_exact and not plans:
            raise RuntimeError("cuDNN found no executable encoder plan")
        if not require_exact and not plan_specs:
            raise RuntimeError("cuDNN found no executable encoder plan")
        if reference is None:
            reference = F.conv1d(
                F.pad(example.squeeze(2).contiguous(), (left, right)),
                weight3,
                bias,
                stride=stride,
            )
        self.selection: dict[str, float | int | bool] = {}
        if require_exact:
            self.plan = self._autotune_exact(example, reference, plans)
        else:
            self.plan = self._autotune_quality_specs(example, reference, plan_specs)

    def _select(self, plan: tuple[int, torch.Tensor]) -> None:
        self.graph.build_plan_at_index(plan[0])
        self.plan = plan

    def _execute(self, values: torch.Tensor) -> torch.Tensor:
        cudnn.set_stream(
            handle=self.handle,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        self.graph.execute(
            {
                self.input_tensor: values,
                self.weight_tensor: self.weight,
                self.bias_tensor: self.bias,
                self.output_tensor: self.output,
            },
            self.plan[1],
            handle=self.handle,
        )
        return self.output

    def _autotune_exact(
        self,
        values: torch.Tensor,
        reference: torch.Tensor,
        plans: list[tuple[int, torch.Tensor]],
    ) -> tuple[int, torch.Tensor]:
        timings: list[tuple[float, tuple[int, torch.Tensor]]] = []
        repetitions = 7
        for plan in plans:
            self._select(plan)
            self._execute(values)
            self._execute(values)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                self._execute(values)
            end.record()
            end.synchronize()
            if torch.equal(self.output.squeeze(2), reference):
                timings.append((start.elapsed_time(end) / repetitions, plan))
        if not timings:
            raise RuntimeError("cuDNN found no bit-exact encoder plan")
        selected = min(timings, key=lambda item: item[0])[1]
        self._select(selected)
        return selected

    def _autotune_quality(
        self,
        values: torch.Tensor,
        reference: torch.Tensor,
        plans: list[tuple[int, torch.Tensor]],
    ) -> tuple[int, torch.Tensor]:
        """Experimental fallback: prefer numerical proximity, then latency.

        Production callers retain ``require_exact=True``.  This route exists so
        a layout-preserving chain can be evaluated by the frozen end-to-end
        code/audio gates when no individual frontend plan is bit-identical.
        """
        ranked: list[tuple[float, float, float, tuple[int, torch.Tensor]]] = []
        repetitions = 7
        for plan in plans:
            self._select(plan)
            self._execute(values)
            self._execute(values)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                self._execute(values)
            end.record()
            end.synchronize()
            delta = (self.output.squeeze(2) - reference).abs()
            ranked.append(
                (
                    float(delta.max().item()),
                    float(delta.mean().item()),
                    start.elapsed_time(end) / repetitions,
                    plan,
                )
            )
        if not ranked:
            raise RuntimeError("cuDNN found no quality-ranked encoder plan")
        maximum, mean, milliseconds, selected = min(
            ranked, key=lambda item: (item[0], item[1], item[2])
        )
        self.selection = {
            "plan_index": selected[0],
            "max_abs": maximum,
            "mean_abs": mean,
            "milliseconds": milliseconds,
            "exact": maximum == 0.0,
            "plan_count": len(ranked),
        }
        self._select(selected)
        return selected

    def _autotune_quality_specs(
        self,
        values: torch.Tensor,
        reference: torch.Tensor,
        plan_specs: list[tuple[int, int]],
    ) -> tuple[int, torch.Tensor]:
        """Quality-rank plans while retaining only one workspace at a time."""
        ranked: list[tuple[float, float, float, int, int]] = []
        repetitions = 7
        for index, workspace_size in plan_specs:
            self.graph.build_plan_at_index(index)
            workspace = torch.empty(
                workspace_size,
                device=values.device,
                dtype=torch.uint8,
            )
            self.plan = (index, workspace)
            self._execute(values)
            self._execute(values)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                self._execute(values)
            end.record()
            end.synchronize()

            output_flat = self.output.squeeze(2).reshape(-1)
            reference_flat = reference.reshape(-1)
            maximum = 0.0
            total = 0.0
            chunk = 8 * 1024 * 1024
            for offset in range(0, output_flat.numel(), chunk):
                delta = (
                    output_flat[offset : offset + chunk]
                    - reference_flat[offset : offset + chunk]
                ).abs()
                maximum = max(maximum, float(delta.max().item()))
                total += float(delta.sum(dtype=torch.float64).item())
            ranked.append(
                (
                    maximum,
                    total / output_flat.numel(),
                    start.elapsed_time(end) / repetitions,
                    index,
                    workspace_size,
                )
            )
            del workspace
        maximum, mean, milliseconds, index, workspace_size = min(
            ranked, key=lambda item: (item[0], item[1], item[2])
        )
        selected = (
            index,
            torch.empty(
                workspace_size,
                device=values.device,
                dtype=torch.uint8,
            ),
        )
        self.selection = {
            "plan_index": index,
            "max_abs": maximum,
            "mean_abs": mean,
            "milliseconds": milliseconds,
            "exact": maximum == 0.0,
            "plan_count": len(ranked),
            "workspace_bytes": workspace_size,
        }
        self._select(selected)
        return selected

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return self._execute(values)


class _StaticCudaGraph:
    """Replay a fixed-shape CUDA graph while keeping input copies inside the call."""

    def __init__(self, function: Any, example: torch.Tensor) -> None:
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                function(self.input)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = function(self.input)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        self.input.copy_(values)
        self.graph.replay()
        return self.output


class _FixedCudaGraph:
    """Replay a fixed-pointer CUDA graph fed by an upstream static output."""

    def __init__(self, function: Any) -> None:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = function()

    def __call__(self) -> Any:
        self.graph.replay()
        return self.output


class OptimizedLongMimi(CompiledSeanetsMimi):
    """Long-form Mimi with exact window skipping and a mixed-precision NHWC decoder tail."""

    attention_block = 64
    attention_prefix = 256
    fp16_decoder_layers = frozenset(
        {
            "decoder.layers.12.block.3.conv",
        }
    )

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self._compiled_encoder_suffix = torch.compile(
            self._encoder_suffix_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_encoder_prefix1 = torch.compile(
            self._encoder_prefix1_graph,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        self._compiled_encoder_after3 = torch.compile(
            self._encoder_after3_graph,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        self._compiled_encoder_e3_bridge = torch.compile(
            self._encoder_e3_bridge_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._encoder_cudnn_cache: dict[
            tuple[torch.device, int],
            _CudnnExactEncoderConv | None,
        ] = {}
        self._encoder_e3_cudnn_cache: dict[
            tuple[torch.device, int],
            _CudnnExactEncoderConv | None,
        ] = {}
        self._local_mask_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._transformer_graph_cache: dict[
            tuple[Any, ...],
            _StaticCudaGraph | None,
        ] = {}
        self._fixed_encoder_suffix_cache: dict[
            tuple[Any, ...],
            _FixedCudaGraph | None,
        ] = {}
        self._fixed_bottleneck_cache: dict[int, _FixedCudaGraph | None] = {}
        self._fixed_decoder_cache: dict[int, _FixedCudaGraph | None] = {}
        self._packed_qkv = PackedQkvTransformer(self)
        self._embed("quantizer.semantic_residual_vector_quantizer.layers.0.codebook")
        for index in range(7):
            self._embed(
                f"quantizer.acoustic_residual_vector_quantizer.layers.{index}.codebook"
            )
        self._compiled_bottleneck_downsample = torch.compile(
            self._bottleneck_downsample_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_bottleneck_reconstruct = torch.compile(
            self._bottleneck_reconstruct_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._decoder_weights4 = {
            key: value.unsqueeze(2).contiguous(memory_format=torch.channels_last)
            for key, value in self.state.items()
            if key.startswith("decoder.")
            and key.endswith(".conv.weight")
            and value.ndim == 3
        }
        self._decoder_half_weights4 = {
            key: value.to(torch.float16)
            for key, value in self._decoder_weights4.items()
            if key.removesuffix(".weight") in self.fp16_decoder_layers
        }
        self._decoder_half_biases = {
            key: value.to(torch.float16)
            for key, value in self.state.items()
            if key.removesuffix(".bias") in self.fp16_decoder_layers
        }
        self._cutlass_bias_dgrad = load_cutlass_bias_dgrad()
        self._fused_decoder9_wmma: FusedDecoder9Wmma | None = load_fused_decoder9_wmma()
        self._decoder12_final_wmma: Decoder12FinalWmma | None = (
            load_decoder12_final_wmma()
        )
        self._native_final_post: NativeFinalPost | None = load_native_final_post()
        self._native_final_post_outputs: dict[
            tuple[torch.device, int], torch.Tensor
        ] = {}
        self._compiled_nhwc_decoder = torch.compile(
            self._decoder_nhwc_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_nhwc_decoder_prefix = torch.compile(
            self._decoder_nhwc_prefix,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_nhwc_decoder_prefix2 = torch.compile(
            self._decoder_nhwc_prefix2,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        self._compiled_nhwc_decoder_after2 = torch.compile(
            self._decoder_nhwc_after2,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_nhwc_decoder_after5 = torch.compile(
            self._decoder_nhwc_after5,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_nhwc_decoder_after9 = torch.compile(
            self._decoder_nhwc_after9,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._compiled_nhwc_decoder_post12 = torch.compile(
            self._decoder_nhwc_post12,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        self._decoder_cudnn_cache: dict[
            tuple[torch.device, int],
            tuple[Any, ...] | None,
        ] = {}

    def _bottleneck_downsample_graph(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.downsample(embeddings.transpose(1, 2))

    def _bottleneck_reconstruct_graph(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed = self.quantizer_decode(codes)
        return self.upsample(reconstructed).transpose(1, 2)

    def _quality_safe_bottleneck(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = self._compiled_bottleneck_downsample(embeddings)
        # Keep cdist/argmin outside Inductor. Its alternative reduction order
        # crossed a nearest-code boundary on the frozen 100-second seed 1103.
        codes = self.quantizer_encode(compressed, 8)
        decoded = self._compiled_bottleneck_reconstruct(codes)
        return codes, decoded

    def _make_transformer_cuda_graph(
        self,
        hidden: torch.Tensor,
        prefix: str,
    ) -> _StaticCudaGraph | None:
        key = (prefix, hidden.device, hidden.dtype, tuple(hidden.shape))
        if key in self._transformer_graph_cache:
            return self._transformer_graph_cache[key]
        if hidden.device.type != "cuda":
            self._transformer_graph_cache[key] = None
            return None
        try:
            transformer = (
                self._packed_qkv
                if tuple(hidden.shape) == (1, 2_500, self.hidden_size)
                else self.transformer
            )
            result = _StaticCudaGraph(
                lambda values: transformer(values, prefix),
                hidden,
            )
        except Exception:
            result = None
        self._transformer_graph_cache[key] = result
        return result

    def _make_fixed_encoder_suffix(
        self,
        input_values: torch.Tensor,
        first_graph: _CudnnExactEncoderConv,
        e3_graph: _CudnnExactEncoderConv,
    ) -> _FixedCudaGraph | None:
        key = (input_values.device, input_values.dtype, tuple(input_values.shape))
        if key in self._fixed_encoder_suffix_cache:
            return self._fixed_encoder_suffix_cache[key]
        if tuple(input_values.shape) != (1, 1, 2_400_000):
            self._fixed_encoder_suffix_cache[key] = None
            return None

        def suffix() -> torch.Tensor:
            hidden = self._compiled_encoder_prefix1(first_graph.output.squeeze(2))
            source4 = self._compiled_encoder_e3_bridge(hidden)
            downsampled = e3_graph(source4).squeeze(2)
            return self._compiled_encoder_after3(downsampled)

        try:
            result = _FixedCudaGraph(suffix)
        except Exception:
            result = None
        self._fixed_encoder_suffix_cache[key] = result
        return result

    def _make_fixed_bottleneck(
        self,
        encoder_graph: _StaticCudaGraph,
    ) -> _FixedCudaGraph | None:
        key = id(encoder_graph)
        if key in self._fixed_bottleneck_cache:
            return self._fixed_bottleneck_cache[key]
        if tuple(encoder_graph.output.shape) != (1, 2_500, self.hidden_size):
            self._fixed_bottleneck_cache[key] = None
            return None
        try:
            result = _FixedCudaGraph(
                lambda: self._quality_safe_bottleneck(encoder_graph.output)
            )
        except Exception:
            result = None
        self._fixed_bottleneck_cache[key] = result
        return result

    def _make_fixed_decoder(
        self,
        decoder_graph: _StaticCudaGraph,
    ) -> _FixedCudaGraph | None:
        key = id(decoder_graph)
        if key in self._fixed_decoder_cache:
            return self._fixed_decoder_cache[key]
        if tuple(decoder_graph.output.shape) != (1, 2_500, self.hidden_size):
            self._fixed_decoder_cache[key] = None
            return None
        try:
            result = _FixedCudaGraph(
                lambda: self.decoder(decoder_graph.output.transpose(1, 2))
            )
        except Exception:
            result = None
        self._fixed_decoder_cache[key] = result
        return result

    def _encoder_suffix_graph(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self._residual(hidden, "encoder.layers.1")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.3.conv",
            stride=4,
        )
        hidden = self._residual(hidden, "encoder.layers.4")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.6.conv",
            stride=5,
        )
        hidden = self._residual(hidden, "encoder.layers.7")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.9.conv",
            stride=6,
        )
        hidden = self._residual(hidden, "encoder.layers.10")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.12.conv",
            stride=8,
        )
        return self._conv(F.elu(hidden), "encoder.layers.14.conv")

    def _encoder_prefix1_graph(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.elu(self._residual(hidden, "encoder.layers.1"))

    @staticmethod
    def _encoder_e3_bridge_graph(hidden: torch.Tensor) -> torch.Tensor:
        return hidden.unsqueeze(2).contiguous(memory_format=torch.channels_last)

    def _encoder_after3_graph(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self._residual(hidden, "encoder.layers.4")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.6.conv",
            stride=5,
        )
        hidden = self._residual(hidden, "encoder.layers.7")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.9.conv",
            stride=6,
        )
        hidden = self._residual(hidden, "encoder.layers.10")
        hidden = self._conv(
            F.elu(hidden),
            "encoder.layers.12.conv",
            stride=8,
        )
        return self._conv(F.elu(hidden), "encoder.layers.14.conv")

    def _make_encoder_cudnn_graph(
        self,
        input_values: torch.Tensor,
    ) -> _CudnnExactEncoderConv | None:
        key = (input_values.device, input_values.shape[-1])
        if key in self._encoder_cudnn_cache:
            return self._encoder_cudnn_cache[key]
        if cudnn is None or input_values.device.type != "cuda":
            self._encoder_cudnn_cache[key] = None
            return None
        try:
            example = input_values.unsqueeze(2).contiguous(
                memory_format=torch.channels_last
            )
            result = _CudnnExactEncoderConv(
                example,
                self.state["encoder.layers.0.conv.weight"],
                self.state["encoder.layers.0.conv.bias"],
            )
        except Exception:
            result = None
        self._encoder_cudnn_cache[key] = result
        return result

    def _make_encoder_e3_cudnn_graph(
        self,
        hidden: torch.Tensor,
    ) -> _CudnnExactEncoderConv | None:
        key = (hidden.device, hidden.shape[-1])
        if key in self._encoder_e3_cudnn_cache:
            return self._encoder_e3_cudnn_cache[key]
        if cudnn is None or hidden.device.type != "cuda":
            self._encoder_e3_cudnn_cache[key] = None
            return None
        try:
            example = self._compiled_encoder_e3_bridge(hidden)
            reference = self._conv(
                hidden,
                "encoder.layers.3.conv",
                stride=4,
            )
            result = _CudnnExactEncoderConv(
                example,
                self.state["encoder.layers.3.conv.weight"],
                self.state["encoder.layers.3.conv.bias"],
                stride=4,
                reference=reference,
            )
        except Exception:
            result = None
        self._encoder_e3_cudnn_cache[key] = result
        return result

    def encoder(self, input_values: torch.Tensor) -> torch.Tensor:
        graph = self._make_encoder_cudnn_graph(input_values)
        if graph is None:
            return super().encoder(input_values)
        values4 = input_values.unsqueeze(2).contiguous(
            memory_format=torch.channels_last
        )
        hidden = graph(values4).squeeze(2)
        fixed_key = (
            input_values.device,
            input_values.dtype,
            tuple(input_values.shape),
        )
        cached_fixed = self._fixed_encoder_suffix_cache.get(fixed_key)
        if cached_fixed is not None:
            return cached_fixed()
        prefix = self._compiled_encoder_prefix1(hidden)
        e3_graph = self._make_encoder_e3_cudnn_graph(prefix)
        if e3_graph is None:
            return self._compiled_encoder_suffix(hidden)
        fixed = self._make_fixed_encoder_suffix(input_values, graph, e3_graph)
        if fixed is not None:
            return fixed()
        source4 = self._compiled_encoder_e3_bridge(prefix)
        downsampled = e3_graph(source4).squeeze(2)
        return self._compiled_encoder_after3(downsampled)

    @staticmethod
    def _block_view(
        values: torch.Tensor,
        *,
        start: int,
        count: int,
        step: int,
        length: int,
    ) -> torch.Tensor:
        stride = values.stride()
        return torch.as_strided(
            values,
            size=(count, values.shape[1], length, values.shape[3]),
            stride=(step * stride[2], stride[1], stride[2], stride[3]),
            storage_offset=values.storage_offset() + start * stride[2],
        )

    def _local_mask(
        self,
        query_offset: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (query_offset, query_length, key_length, device)
        cached = self._local_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        query = torch.arange(
            query_offset,
            query_offset + query_length,
            device=device,
        )[:, None]
        key = torch.arange(key_length, device=device)[None, :]
        result = ((key <= query) & (key > query - self.sliding_window))[None, None]
        self._local_mask_cache[cache_key] = result
        return result

    def _early_mask(self, device: torch.device) -> torch.Tensor:
        cache_key = ("early", self.attention_block, device)
        cached = self._local_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        result = torch.cat(
            tuple(
                self._local_mask(
                    index * self.attention_block,
                    self.attention_block,
                    self.attention_prefix,
                    device,
                )
                for index in range(self.attention_prefix // self.attention_block)
            ),
            dim=0,
        )
        self._local_mask_cache[cache_key] = result
        return result

    def _window_attention_nlc(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        length = query.shape[2]
        scaling = self.head_dim**-0.5
        if length < self.attention_prefix:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=self._attention_mask(length, query.device, query.dtype),
                dropout_p=0.0,
                scale=scaling,
            )
            return (
                attended.transpose(1, 2)
                .contiguous()
                .view(
                    1,
                    length,
                    self.hidden_size,
                )
            )

        block = self.attention_block
        prefix = self.attention_prefix
        early_count = prefix // block
        early_query = self._block_view(
            query,
            start=0,
            count=early_count,
            step=block,
            length=block,
        )
        early_key = key[:, :, :prefix].expand(early_count, -1, -1, -1)
        early_value = value[:, :, :prefix].expand(early_count, -1, -1, -1)
        early = F.scaled_dot_product_attention(
            early_query,
            early_key,
            early_value,
            attn_mask=self._early_mask(query.device),
            dropout_p=0.0,
            scale=scaling,
        )

        full_blocks = length // block
        middle_count = full_blocks - early_count
        if middle_count:
            middle_query = self._block_view(
                query,
                start=prefix,
                count=middle_count,
                step=block,
                length=block,
            )
            middle_key = self._block_view(
                key,
                start=0,
                count=middle_count,
                step=block,
                length=prefix + block,
            )
            middle_value = self._block_view(
                value,
                start=0,
                count=middle_count,
                step=block,
                length=prefix + block,
            )
            middle = F.scaled_dot_product_attention(
                middle_query,
                middle_key,
                middle_value,
                attn_mask=self._local_mask(
                    prefix,
                    block,
                    prefix + block,
                    query.device,
                ),
                dropout_p=0.0,
                scale=scaling,
            )
        else:
            middle = early

        tail_start = full_blocks * block
        if tail_start < length:
            tail_length = length - tail_start
            tail = F.scaled_dot_product_attention(
                query[:, :, tail_start:],
                key[:, :, tail_start - prefix :],
                value[:, :, tail_start - prefix :],
                attn_mask=self._local_mask(
                    prefix,
                    tail_length,
                    prefix + tail_length,
                    query.device,
                ),
                dropout_p=0.0,
                scale=scaling,
            )
        else:
            tail = early

        output = torch.empty(
            (1, length, self.hidden_size),
            device=query.device,
            dtype=query.dtype,
        )
        elements = output.numel()
        pack_block = 256
        _attention_pack_nlc_kernel[(triton.cdiv(elements, pack_block),)](
            early,
            middle,
            tail,
            output,
            length,
            self.num_heads,
            self.head_dim,
            block,
            prefix,
            tail_start,
            BLOCK=pack_block,
            num_warps=4,
        )
        return output

    def transformer(self, hidden: torch.Tensor, prefix: str) -> torch.Tensor:
        length = hidden.shape[1]
        cosine, sine = self._position_embeddings(length, hidden.device, hidden.dtype)

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
            query = F.linear(
                normalized,
                self.state[f"{name}.self_attn.q_proj.weight"],
            )
            key = F.linear(
                normalized,
                self.state[f"{name}.self_attn.k_proj.weight"],
            )
            value = F.linear(
                normalized,
                self.state[f"{name}.self_attn.v_proj.weight"],
            )
            query, key = _triton_rope(query, key, cosine, sine)
            value = value.view(
                1,
                length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)
            attended = self._window_attention_nlc(query, key, value)
            attended = F.linear(
                attended,
                self.state[f"{name}.self_attn.o_proj.weight"],
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
            mlp = F.linear(normalized, self.state[f"{name}.mlp.fc1.weight"])
            mlp = F.gelu(mlp)
            mlp = F.linear(mlp, self.state[f"{name}.mlp.fc2.weight"])
            hidden = residual + mlp * self.state[f"{name}.mlp_layer_scale.scale"]
        return hidden

    def _decoder_conv2d(
        self,
        values: torch.Tensor,
        name: str,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> torch.Tensor:
        use_half = name in self.fp16_decoder_layers
        weight = (
            self._decoder_half_weights4[f"{name}.weight"]
            if use_half
            else self._decoder_weights4[f"{name}.weight"]
        )
        effective_kernel = (weight.shape[-1] - 1) * dilation + 1
        padding_left = effective_kernel - stride
        length = values.shape[-1]
        numerator = length - effective_kernel + padding_left
        frame_count = (numerator + stride - 1) // stride
        ideal_length = frame_count * stride + effective_kernel - padding_left
        padding_right = ideal_length - length
        values = F.pad(values, (padding_left, padding_right, 0, 0))
        if use_half:
            values = values.to(torch.float16)
        bias = self.state.get(f"{name}.bias")
        if use_half and bias is not None:
            bias = self._decoder_half_biases[f"{name}.bias"]
        output = F.conv2d(
            values,
            weight,
            bias,
            stride=(1, stride),
            dilation=(1, dilation),
        )
        return output.to(torch.float32) if use_half else output

    def _decoder_conv_transpose2d(
        self,
        values: torch.Tensor,
        name: str,
        *,
        stride: int,
    ) -> torch.Tensor:
        weight = self._decoder_weights4[f"{name}.weight"]
        values = F.conv_transpose2d(
            values,
            weight,
            self.state.get(f"{name}.bias"),
            stride=(1, stride),
        )
        padding_right = weight.shape[-1] - stride
        return values[..., :-padding_right] if padding_right else values

    def _decoder_residual2d(
        self,
        values: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        hidden = self._decoder_conv2d(F.elu(values), f"{name}.block.1.conv")
        hidden = self._decoder_conv2d(F.elu(hidden), f"{name}.block.3.conv")
        return values + hidden

    def _decoder_nhwc_prefix(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = embeddings.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        hidden = self._decoder_conv2d(hidden, "decoder.layers.0.conv")
        hidden = self._decoder_conv_transpose2d(
            F.elu(hidden),
            "decoder.layers.2.conv",
            stride=8,
        )
        hidden = self._decoder_residual2d(hidden, "decoder.layers.3")
        hidden = self._decoder_conv_transpose2d(
            F.elu(hidden),
            "decoder.layers.5.conv",
            stride=6,
        )
        hidden = self._decoder_residual2d(hidden, "decoder.layers.6")
        return F.elu(hidden)

    def _decoder_nhwc_prefix2(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = embeddings.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        hidden = self._decoder_conv2d(hidden, "decoder.layers.0.conv")
        return F.elu(hidden)

    def _decoder_nhwc_after2(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.elu(self._decoder_residual2d(hidden, "decoder.layers.3"))

    def _decoder_nhwc_after5(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.elu(self._decoder_residual2d(hidden, "decoder.layers.6"))

    def _decoder_nhwc_after9(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        return F.elu(residual + branch)

    def _decoder_nhwc_post12(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._decoder_conv2d(
            F.elu(residual + branch),
            "decoder.layers.14.conv",
        )
        return hidden.squeeze(2)

    def _make_decoder_cudnn_graphs(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[Any, ...] | None:
        key = (embeddings.device, embeddings.shape[-1])
        if key in self._decoder_cudnn_cache:
            return self._decoder_cudnn_cache[key]
        if cudnn is None or embeddings.device.type != "cuda":
            self._decoder_cudnn_cache[key] = None
            return None
        stage = "prefix2"
        try:
            activated2 = self._compiled_nhwc_decoder_prefix2(embeddings)
            stage = "deconv2"
            graph2 = _CudnnDeconv(
                activated2,
                self._decoder_weights4["decoder.layers.2.conv.weight"],
                self.state["decoder.layers.2.conv.bias"],
                8,
                require_exact=False,
                quality_first=True,
            )
            activated5 = self._compiled_nhwc_decoder_after2(graph2(activated2))
            stage = "deconv5"
            graph5 = _CudnnDeconv(
                activated5,
                self._decoder_weights4["decoder.layers.5.conv.weight"],
                self.state["decoder.layers.5.conv.bias"],
                6,
                require_exact=False,
                quality_first=True,
            )
            activated8 = self._compiled_nhwc_decoder_after5(graph5(activated5))
            stage = "deconv8"
            graph8 = _CudnnDeconv(
                activated8,
                self._decoder_weights4["decoder.layers.8.conv.weight"],
                self.state["decoder.layers.8.conv.bias"],
                5,
                quality_first=True,
            )
            residual9 = graph8(activated8)
            stage = "residual9_conv1"
            graph9b1 = _CudnnHalfInputConv(
                residual9,
                self._decoder_weights4["decoder.layers.9.block.1.conv.weight"],
                self.state["decoder.layers.9.block.1.conv.bias"],
            )
            branch9 = graph9b1(residual9)
            stage = "residual9_conv3"
            graph9b3 = _CudnnHalfInputConv(
                branch9,
                self._decoder_weights4["decoder.layers.9.block.3.conv.weight"],
                self.state["decoder.layers.9.block.3.conv.bias"],
            )
            branch9 = graph9b3(branch9)
            activated11 = self._compiled_nhwc_decoder_after9(
                residual9,
                branch9,
            )
            stage = "deconv11"
            graph11 = _CudnnDeconv(
                activated11,
                self._decoder_weights4["decoder.layers.11.conv.weight"],
                self.state["decoder.layers.11.conv.bias"],
                4,
                quality_first=True,
            )
            cutlass11 = None
            if self._cutlass_bias_dgrad is not None:
                try:
                    cutlass11 = _CutlassDecoderLayer11(
                        self._cutlass_bias_dgrad,
                        residual9,
                        branch9,
                        self._decoder_weights4["decoder.layers.11.conv.weight"],
                        self.state["decoder.layers.11.conv.bias"],
                    )
                except Exception:
                    cutlass11 = None
            if (
                cutlass11 is not None
                and self._fused_decoder9_wmma is not None
                and residual9.shape[-1] == 600_000
            ):
                try:
                    cutlass11.decoder9 = _FusedDecoder9(
                        self._fused_decoder9_wmma,
                        residual9,
                        self._decoder_weights4["decoder.layers.9.block.1.conv.weight"],
                        self.state["decoder.layers.9.block.1.conv.bias"],
                        self._decoder_weights4["decoder.layers.9.block.3.conv.weight"],
                        self.state["decoder.layers.9.block.3.conv.bias"],
                    )
                except Exception:
                    cutlass11.decoder9 = None
            residual12 = (
                cutlass11(residual9, branch9)
                if cutlass11 is not None
                else graph11(activated11)
            )
            if (
                cutlass11 is not None
                and self._decoder12_final_wmma is not None
                and residual12.shape[-1] == 2_400_000
            ):
                try:
                    cutlass11.decoder12_final = _FusedDecoder12Final(
                        self._decoder12_final_wmma,
                        residual12,
                        self._decoder_weights4["decoder.layers.12.block.1.conv.weight"],
                        self.state["decoder.layers.12.block.1.conv.bias"],
                        self._decoder_weights4["decoder.layers.12.block.3.conv.weight"],
                        self.state["decoder.layers.12.block.3.conv.bias"],
                        self.state["decoder.layers.14.conv.weight"],
                        self.state["decoder.layers.14.conv.bias"],
                    )
                except Exception:
                    cutlass11.decoder12_final = None
            stage = "residual12_conv1"
            graph12b1 = _CudnnHalfInputConv(
                residual12,
                self._decoder_weights4["decoder.layers.12.block.1.conv.weight"],
                self.state["decoder.layers.12.block.1.conv.bias"],
            )
            branch12 = graph12b1(residual12)
            stage = "residual12_conv3"
            graph12b3 = _CudnnHalfInputConv(
                branch12,
                self._decoder_weights4["decoder.layers.12.block.3.conv.weight"],
                self.state["decoder.layers.12.block.3.conv.bias"],
            )
            branch12 = graph12b3(branch12)
            reference_post = self._compiled_nhwc_decoder_post12(
                residual12,
                branch12,
            )
            try:
                graph_post = _CudnnFusedDecoderPost(
                    residual12,
                    branch12,
                    self._decoder_weights4["decoder.layers.14.conv.weight"],
                    self.state["decoder.layers.14.conv.bias"],
                    reference_post,
                )
            except Exception:
                graph_post = None
            result = (
                graph2,
                graph5,
                graph8,
                graph9b1,
                graph9b3,
                graph11,
                cutlass11,
                graph12b1,
                graph12b3,
                graph_post,
            )
        except Exception as error:
            # Preserve the fallback while retaining the backend failure for
            # diagnostics. Silent plan failures otherwise look like genuine
            # performance regressions in the compiled NHWC path.
            self._decoder_cudnn_error = f"{stage}: {error!r}"
            result = None
        self._decoder_cudnn_cache[key] = result
        return result

    def _decoder_nhwc_graph(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = embeddings.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        hidden = self._decoder_conv2d(hidden, "decoder.layers.0.conv")
        hidden = self._decoder_conv_transpose2d(
            F.elu(hidden),
            "decoder.layers.2.conv",
            stride=8,
        )
        hidden = self._decoder_residual2d(hidden, "decoder.layers.3")
        hidden = self._decoder_conv_transpose2d(
            F.elu(hidden),
            "decoder.layers.5.conv",
            stride=6,
        )
        hidden = self._decoder_residual2d(hidden, "decoder.layers.6")
        hidden = self._decoder_conv_transpose2d(
            F.elu(hidden),
            "decoder.layers.8.conv",
            stride=5,
        )
        hidden = self._decoder_residual2d(hidden, "decoder.layers.9")
        hidden = self._decoder_conv_transpose2d(
            F.elu(hidden),
            "decoder.layers.11.conv",
            stride=4,
        )
        hidden = self._decoder_residual2d(hidden, "decoder.layers.12")
        hidden = self._decoder_conv2d(F.elu(hidden), "decoder.layers.14.conv")
        return hidden.squeeze(2)

    def decoder(self, embeddings: torch.Tensor) -> torch.Tensor:
        graphs = self._make_decoder_cudnn_graphs(embeddings)
        if graphs is None:
            return self._compiled_nhwc_decoder(embeddings)
        (
            graph2,
            graph5,
            graph8,
            graph9b1,
            graph9b3,
            graph11,
            cutlass11,
            graph12b1,
            graph12b3,
            graph_post,
        ) = graphs
        activated2 = self._compiled_nhwc_decoder_prefix2(embeddings)
        activated5 = self._compiled_nhwc_decoder_after2(graph2(activated2))
        activated8 = self._compiled_nhwc_decoder_after5(graph5(activated5))
        residual9 = graph8(activated8)
        if cutlass11 is not None and cutlass11.decoder9 is not None:
            activated11 = cutlass11.decoder9(residual9)
            residual12 = cutlass11.from_half(activated11)
        else:
            branch9 = graph9b1(residual9)
            branch9 = graph9b3(branch9)
        if cutlass11 is None:
            activated11 = self._compiled_nhwc_decoder_after9(residual9, branch9)
            residual12 = graph11(activated11)
        elif cutlass11.decoder9 is None:
            residual12 = cutlass11(residual9, branch9)
        if cutlass11 is not None and cutlass11.decoder12_final is not None:
            return cutlass11.decoder12_final(residual12)
        branch12 = graph12b1(residual12)
        branch12 = graph12b3(branch12)
        if self._native_final_post is not None:
            output_key = (residual12.device, residual12.shape[-1])
            output = self._native_final_post_outputs.get(output_key)
            if output is None:
                output = torch.empty(
                    (1, 1, residual12.shape[-1]),
                    device=residual12.device,
                    dtype=torch.float32,
                )
                self._native_final_post_outputs[output_key] = output
            return self._native_final_post(
                residual12,
                branch12,
                self.state["decoder.layers.14.conv.weight"],
                self.state["decoder.layers.14.conv.bias"],
                output,
                reduction_order=14,
                exp_mode=2,
                config=0,
            )
        if graph_post is None:
            return self._compiled_nhwc_decoder_post12(residual12, branch12)
        return graph_post(residual12, branch12)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
        num_quantizers: int,
    ) -> SimpleNamespace:
        embeddings = self.encoder(input_values).transpose(1, 2)
        encoder_graph = self._make_transformer_cuda_graph(
            embeddings,
            "encoder_transformer",
        )
        embeddings = (
            encoder_graph(embeddings)
            if encoder_graph is not None
            else self.transformer(embeddings, "encoder_transformer")
        )
        if num_quantizers == 8:
            fixed_bottleneck = (
                self._make_fixed_bottleneck(encoder_graph)
                if encoder_graph is not None
                else None
            )
            codes, decoded = (
                fixed_bottleneck()
                if fixed_bottleneck is not None
                else self._quality_safe_bottleneck(embeddings)
            )
        else:
            compressed = self.downsample(embeddings.transpose(1, 2))
            codes = self.quantizer_encode(compressed, num_quantizers)
            decoded = self.upsample(self.quantizer_decode(codes)).transpose(1, 2)
        decoder_graph = self._make_transformer_cuda_graph(
            decoded,
            "decoder_transformer",
        )
        decoded = (
            decoder_graph(decoded)
            if decoder_graph is not None
            else self.transformer(decoded, "decoder_transformer")
        )
        fixed_decoder = (
            self._make_fixed_decoder(decoder_graph)
            if decoder_graph is not None and num_quantizers == 8
            else None
        )
        audio_values = (
            fixed_decoder()
            if fixed_decoder is not None
            else self.decoder(decoded.transpose(1, 2))
        )
        if padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]
        return SimpleNamespace(
            audio_codes=codes,
            audio_values=audio_values,
            encoder_past_key_values=None,
            decoder_past_key_values=None,
        )


class AdaptiveMimi:
    """Use eager PyTorch for short clips and the profiled long-form path otherwise."""

    long_audio_samples = 240_000

    def __init__(self, model: Any) -> None:
        self.model = model
        self.short_audio = CompiledSeanetsMimi(model)
        self.long_audio: OptimizedLongMimi | None = None

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
        num_quantizers: int,
    ) -> Any:
        if input_values.shape[-1] < self.long_audio_samples:
            return self.short_audio.forward(input_values, padding_mask, num_quantizers)
        if self.long_audio is None:
            self.long_audio = OptimizedLongMimi(self.model)
        return self.long_audio.forward(input_values, padding_mask, num_quantizers)
