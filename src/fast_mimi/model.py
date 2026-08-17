"""Exact float32 implementation of Kyutai's Mimi neural audio codec.

The checkpoint-compatible module tree lives in this file so its parameter
paths remain easy to compare with the Transformers reference. Tensor operation
order is intentional: changing a reduction, residual addition, or padding
representation can change float32 bits even when the formulas are equivalent.

The default model transforms a 24 kHz mono waveform through a causal SEANet
encoder, an eight-layer local-attention Transformer, a stride-two bottleneck,
and a split residual vector quantizer. Decoding mirrors that path through
upsampling, a second Transformer, and a causal SEANet decoder. One final token
frame represents 1,920 samples (80 ms); the checkpoint provides one semantic
and 31 acoustic codebooks.

This module is ordered by responsibility: output/cache types, causal
convolutions, Transformer context layers, residual vector quantization, and the
public ``MimiModel``. Runtime-only compiled graphs and native handles are never
checkpoint state and are cleared whenever weights or devices change.

Optional accelerated paths are narrow, profiler-derived specializations. They
check their complete runtime contract before dispatch and fail closed to the
portable PyTorch implementation when a shape, device, stream, dtype, version,
or toolchain is unsupported.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .config import (
    KYUTAI_MIMI_MODEL_ID,
    KYUTAI_MIMI_PARAMETER_COUNT,
    KYUTAI_MIMI_REVISION,
    KYUTAI_MIMI_WEIGHTS_SHA256,
    MimiConfig,
)

_CUDNN_BENCHMARK_LOCK = threading.Lock()
_DISABLED_ENV_VALUES = frozenset({"1", "true", "yes"})

# Shapes below are specializations measured for the pinned checkpoint. They are
# dispatch guards, not general model constraints.
_PROFILED_SM_CAPABILITY = (12, 0)
_PROFILED_ONE_SECOND_AUDIO_SHAPE = (1, 1, 24_000)
_PROFILED_LONG_AUDIO_SHAPE = (1, 1, 2_400_000)
_PROFILED_ONE_SECOND_DECODER_SHAPE = (1, 512, 26)
_PROFILED_LONG_DECODER_SHAPE = (1, 512, 2_500)
_PROFILED_LONG_TRANSFORMER_SHAPE = (1, 2_500, 512)
_PROFILED_LONG_ATTENTION_SHAPE = (1, 8, 2_500, 64)
_PROFILED_LONG_ATTENTION_STRIDE = (1_280_000, 64, 512, 1)
_PROFILED_RVQ_EMBEDDINGS_SHAPE = (1, 512, 13)
_PROFILED_RVQ_CODES_SHAPE = (1, 32, 13)
_DECLARED_MIMI_CONFIG = {
    "sampling_rate": 24_000,
    "frame_rate": 12.5,
    "audio_channels": 1,
    "codebook_size": 2_048,
}


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without duplicating the checkpoint in RAM."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_declared_mimi_config(config: MimiConfig) -> None:
    """Reject architecture drift for the published, identity-locked checkpoint."""
    for name, expected in _DECLARED_MIMI_CONFIG.items():
        actual = getattr(config, name)
        if actual != expected:
            raise RuntimeError(
                f"declared checkpoint config {name!r} changed: "
                f"{actual!r} != {expected!r}"
            )


def _optional_path_enabled(disable_variable: str) -> bool:
    """Read one fail-closed optimization opt-out flag.

    Args:
        disable_variable: Environment-variable name whose true-like value
            disables an optional accelerated path.

    Returns:
        ``False`` for ``1``, ``true``, or ``yes`` (case-insensitive), otherwise
        ``True``.
    """
    return os.environ.get(disable_variable, "").lower() not in _DISABLED_ENV_VALUES


def _can_prime_long_cudnn_benchmark() -> bool:
    """Check the complete contract for long-shape cuDNN plan priming.

    Returns:
        ``True`` only for the verified PyTorch, CUDA, cuDNN, and global backend
        settings. TF32, deterministic mode, or an existing benchmark mode makes
        the specialization ineligible.
    """
    return (
        torch.__version__.split("+", 1)[0] == "2.13.0"
        and torch.version.cuda == "13.0"
        and torch.backends.cudnn.is_available()
        and torch.backends.cudnn.enabled
        and torch.backends.cudnn.version() == 92000
        and not torch.backends.cudnn.benchmark
        and not torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.allow_tf32
    )


def _run_with_long_cudnn_benchmark(
    function: Callable[[Tensor], Tensor], hidden_states: Tensor
) -> Tensor:
    """Run one function with bounded cuDNN plan search under a lock.

    Args:
        function: Compiled single-input encoder or decoder callable.
        hidden_states: Fixed-shape float32 tensor passed to ``function``.

    Returns:
        Tensor produced by ``function`` after cuDNN selects a plan.

    Raises:
        Exception: Propagates any error from ``function`` after restoring the
            caller's process-global cuDNN settings.
    """
    with _CUDNN_BENCHMARK_LOCK:
        previous_benchmark = torch.backends.cudnn.benchmark
        previous_limit = torch.backends.cudnn.benchmark_limit
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 10
            return function(hidden_states)
        finally:
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cudnn.benchmark_limit = previous_limit


# Output and streaming-state types.


class _OutputMixin:
    """Provide the useful tuple and mapping behavior of ``ModelOutput``.

    Dataclass subclasses declare the actual fields. ``None`` fields are omitted
    from positional iteration while named access continues to expose them.
    """

    def _present_values(self) -> tuple[Any, ...]:
        """Collect non-empty dataclass fields in declaration order.

        Returns:
            Tuple used by positional lookup, iteration, and length reporting.
        """
        return tuple(
            getattr(self, item.name)
            for item in fields(self)
            if getattr(self, item.name) is not None
        )

    def __getitem__(self, key: int | slice | str) -> Any:
        """Look up an output field by position, slice, or name.

        Args:
            key: Integer or slice for the compact tuple view, or a string for
                direct attribute lookup.

        Returns:
            Selected field value or tuple slice.

        Raises:
            AttributeError: If a string key is not a declared attribute.
            IndexError: If a positional key is outside the compact tuple.
        """
        if isinstance(key, str):
            return getattr(self, key)
        return self._present_values()[key]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over fields present in the compact tuple view.

        Returns:
            Iterator over non-``None`` values in declaration order.
        """
        return iter(self._present_values())

    def __len__(self) -> int:
        """Count fields present in the compact tuple view.

        Returns:
            Number of non-``None`` dataclass fields.
        """
        return len(self._present_values())

    def get(self, key: str, default: Any = None) -> Any:
        """Read an output attribute with mapping-style fallback behavior.

        Args:
            key: Attribute name to retrieve.
            default: Value returned when ``key`` is not present.

        Returns:
            Attribute value or ``default``.
        """
        return getattr(self, key, default)


@dataclass
class MimiOutput(_OutputMixin):
    """Hold a complete encode/decode result.

    Attributes:
        audio_codes: Int64 token tensor shaped ``[batch, codebooks, frames]``.
        audio_values: Reconstructed float waveform shaped
            ``[batch, channels, samples]``.
        encoder_past_key_values: Optional encoder Transformer cache.
        decoder_past_key_values: Optional decoder Transformer cache.
    """

    audio_codes: Tensor | None = None
    audio_values: Tensor | None = None
    encoder_past_key_values: MimiKVCache | None = None
    decoder_past_key_values: MimiKVCache | None = None


@dataclass
class MimiEncoderOutput(_OutputMixin):
    """Hold encoded tokens and optional streaming state.

    Attributes:
        audio_codes: Int64 token tensor shaped ``[batch, codebooks, frames]``.
        encoder_past_key_values: Transformer state for the next encoder chunk.
        padding_cache: Causal convolution context for the next encoder chunk.
    """

    audio_codes: Tensor | None = None
    encoder_past_key_values: MimiKVCache | None = None
    padding_cache: MimiConv1dPaddingCache | None = None


@dataclass
class MimiDecoderOutput(_OutputMixin):
    """Hold a decoded waveform and optional continuation state.

    Attributes:
        audio_values: Reconstructed float waveform shaped
            ``[batch, channels, samples]``.
        decoder_past_key_values: Transformer state for the next decode call.
    """

    audio_values: Tensor | None = None
    decoder_past_key_values: MimiKVCache | None = None


class _MimiCudaGraphRunner:
    """Own fixed-address buffers for guarded CUDA Graph replay.

    Replay is restricted to the stream used for capture. Inputs are copied into
    private static buffers, and outputs are cloned so later replays cannot
    mutate tensors already returned to a caller.
    """

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        input_values: Tensor,
        padding_mask: Tensor,
        output: MimiOutput,
        stream: torch.cuda.Stream,
    ) -> None:
        """Keep graph-owned objects alive and record the capture stream.

        Args:
            graph: Captured one-second Q32 round-trip CUDA Graph.
            input_values: Static waveform buffer owned by the capture.
            padding_mask: Static boolean mask buffer owned by the capture.
            output: Graph-owned result tensors populated by each replay.
            stream: CUDA stream on which capture and valid replay occur.
        """
        self.graph = graph
        self.input_values = input_values
        self.padding_mask = padding_mask
        self.output = output
        self.stream_id = stream.cuda_stream
        self.lock = threading.Lock()

    def matches_current_stream(self) -> bool:
        """Check whether the caller is on the captured CUDA stream.

        Returns:
            ``True`` when current and capture stream handles are identical.
        """
        current = torch.cuda.current_stream(self.input_values.device)
        return current.cuda_stream == self.stream_id

    def replay(self, input_values: Tensor, padding_mask: Tensor) -> MimiOutput:
        """Replay the graph with new values and return caller-owned outputs.

        Args:
            input_values: Contiguous one-second waveform matching the captured
                shape, dtype, and device.
            padding_mask: Contiguous boolean mask matching ``input_values``.

        Returns:
            ``MimiOutput`` whose token and waveform tensors are independent
            clones of capture-owned storage.
        """
        with self.lock:
            self.input_values.copy_(input_values)
            self.padding_mask.copy_(padding_mask)
            self.graph.replay()
            return MimiOutput(
                self.output.audio_codes.clone(),
                self.output.audio_values.clone(),
                None,
                None,
            )


@dataclass
class _TransformerOutput(_OutputMixin):
    """Hold internal Transformer outputs.

    Attributes:
        last_hidden_state: Final hidden tensor shaped ``[batch, frames, hidden]``.
        past_key_values: Optional updated sliding key/value cache.
        hidden_states: Optional tuple containing input and per-layer states.
        attentions: Optional tuple of eager attention-probability tensors.
    """

    last_hidden_state: Tensor | None = None
    past_key_values: MimiKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class MimiKVCache:
    """Track cumulative positions while bounding physical key/value storage.

    Each layer stores at most ``sliding_window - 1`` previous frames. The
    cumulative length remains unbounded so RoPE and mask offsets use the same
    absolute positions as the reference implementation.
    """

    def __init__(self, num_layers: int = 0, sliding_window: int | None = None) -> None:
        """Create empty cache slots.

        Args:
            num_layers: Number of layer slots to allocate initially. Additional
                indices can be created lazily.
            sliding_window: Visible attention window, or ``None`` to retain all
                accumulated key/value frames.

        Raises:
            ValueError: If ``sliding_window`` is not positive.
        """
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        self.sliding_window = sliding_window
        self.key_cache: list[Tensor | None] = [None] * num_layers
        self.value_cache: list[Tensor | None] = [None] * num_layers
        self._cumulative_lengths: list[int] = [0] * num_layers

    def _ensure_layer(self, layer_idx: int) -> None:
        """Grow all parallel storage lists through one layer index.

        Args:
            layer_idx: Zero-based Transformer layer index that must exist.
        """
        missing = layer_idx + 1 - len(self.key_cache)
        if missing > 0:
            self.key_cache.extend([None] * missing)
            self.value_cache.extend([None] * missing)
            self._cumulative_lengths.extend([0] * missing)

    def configure_sliding_window(self, sliding_window: int) -> None:
        """Set a cache window and crop previously accumulated storage.

        Args:
            sliding_window: Positive visible-frame count required by the model.

        Raises:
            ValueError: If the window is non-positive or conflicts with an
                existing cache configuration.
        """
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if self.sliding_window is not None:
            if self.sliding_window != sliding_window:
                raise ValueError(
                    "cache sliding_window does not match the model configuration"
                )
            return
        self.sliding_window = sliding_window
        retained = sliding_window - 1
        for layer_idx, key in enumerate(self.key_cache):
            value = self.value_cache[layer_idx]
            if key is None:
                continue
            if self._cumulative_lengths[layer_idx] == 0:
                self._cumulative_lengths[layer_idx] = key.shape[-2]
            if retained == 0:
                self.key_cache[layer_idx] = key[..., :0, :]
                self.value_cache[layer_idx] = value[..., :0, :]
            else:
                self.key_cache[layer_idx] = key[..., -retained:, :]
                self.value_cache[layer_idx] = value[..., -retained:, :]

    def update(
        self, key: Tensor, value: Tensor, layer_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Append one layer's current key/value states.

        Args:
            key: Current key states shaped ``[batch, heads, frames, head_dim]``.
            value: Current value states with the same shape and device as
                ``key``.
            layer_idx: Zero-based layer whose state is being updated.

        Returns:
            Full key and value tensors visible to the current attention call,
            before old frames are cropped from persistent storage.
        """
        self._ensure_layer(layer_idx)
        old_key = self.key_cache[layer_idx]
        old_value = self.value_cache[layer_idx]
        new_length = key.shape[-2]
        if old_key is not None:
            key = torch.cat((old_key, key), dim=-2)
            value = torch.cat((old_value, value), dim=-2)
        self._cumulative_lengths[layer_idx] += new_length
        if self.sliding_window is None:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        else:
            # Retain window - 1 old frames. The current query is concatenated
            # on the next update, yielding exactly one complete visible window.
            retained = self.sliding_window - 1
            if retained == 0:
                self.key_cache[layer_idx] = key[..., :0, :]
                self.value_cache[layer_idx] = value[..., :0, :]
            else:
                self.key_cache[layer_idx] = key[..., -retained:, :]
                self.value_cache[layer_idx] = value[..., -retained:, :]
        return key, value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Read a layer's cumulative sequence length.

        Args:
            layer_idx: Zero-based layer index. Missing layers report zero.

        Returns:
            Total frames ever appended to the layer, including cropped frames.
        """
        if layer_idx >= len(self._cumulative_lengths):
            return 0
        return self._cumulative_lengths[layer_idx]

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        """Compute key-mask dimensions for an upcoming cache update.

        Args:
            query_length: Number of new frames in the current query.
            layer_idx: Layer whose cumulative position determines the offset.

        Returns:
            Pair ``(key_length, key_offset)`` describing visible storage and
            its first absolute sequence position.
        """
        cumulative_length = self.get_seq_length(layer_idx)
        if self.sliding_window is None:
            return cumulative_length + query_length, 0
        key_offset = max(cumulative_length - self.sliding_window + 1, 0)
        if cumulative_length >= self.sliding_window:
            key_length = self.sliding_window - 1 + query_length
        else:
            key_length = cumulative_length + query_length
        return key_length, key_offset

    def to(self, device: torch.device | str) -> MimiKVCache:
        """Move all cached tensors to another device in place.

        Args:
            device: PyTorch device object or device string.

        Returns:
            This cache instance after its tensors are moved.
        """
        self.key_cache = [
            item.to(device) if item is not None else None for item in self.key_cache
        ]
        self.value_cache = [
            item.to(device) if item is not None else None for item in self.value_cache
        ]
        return self


class MimiConv1dPaddingCache:
    """Retain per-layer left context for chunked causal convolution.

    Padding lengths, modes, and channel counts are fixed when the cache is
    created so each streaming update can reconstruct the offline boundary rule.
    """

    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[int],
        per_layer_padding_mode: list[str],
        per_layer_in_channels: list[int],
    ) -> None:
        """Validate layer metadata and allocate empty context slots.

        Args:
            num_layers: Number of cached causal convolutions.
            per_layer_padding: Required left-context length for every layer.
            per_layer_padding_mode: Initial context mode for every layer,
                currently ``"constant"`` or ``"replicate"``.
            per_layer_in_channels: Input channel count for every layer.

        Raises:
            ValueError: If metadata lists do not all contain ``num_layers``
                entries.
        """
        lengths = {
            len(per_layer_padding),
            len(per_layer_padding_mode),
            len(per_layer_in_channels),
        }
        if lengths != {num_layers}:
            raise ValueError(
                "all per-layer cache settings must contain num_layers entries"
            )
        self.per_layer_padding = per_layer_padding
        self.per_layer_padding_mode = per_layer_padding_mode
        self.per_layer_in_channels = per_layer_in_channels
        self.padding_cache: list[Tensor | None] = [None] * num_layers

    def _initial_state(self, hidden_states: Tensor, layer_idx: int) -> Tensor:
        """Construct reference-equivalent context for a stream's first chunk.

        Args:
            hidden_states: Current chunk shaped ``[batch, channels, samples]``.
            layer_idx: Convolution layer whose padding rule is required.

        Returns:
            Initial left-context tensor on the input dtype and device.

        Raises:
            NotImplementedError: If the configured padding mode has no verified
                streaming implementation.
        """
        batch = hidden_states.shape[0]
        padding = self.per_layer_padding[layer_idx]
        channels = self.per_layer_in_channels[layer_idx]
        mode = self.per_layer_padding_mode[layer_idx]
        if mode == "constant":
            return hidden_states.new_zeros((batch, channels, padding))
        if mode == "replicate":
            return hidden_states[..., :1].expand(batch, channels, padding).clone()
        raise NotImplementedError(
            f"padding mode {mode!r} is not supported for streaming"
        )

    def update(self, hidden_states: Tensor, layer_idx: int) -> Tensor:
        """Advance one layer's streaming context.

        Args:
            hidden_states: Current pre-convolution chunk shaped
                ``[batch, channels, samples]``.
            layer_idx: Convolution layer whose context is updated.

        Returns:
            Prior context to prepend to the current chunk. The cache retains the
            newest required samples for the next call.
        """
        padding = self.per_layer_padding[layer_idx]
        channels = self.per_layer_in_channels[layer_idx]
        current = self.padding_cache[layer_idx]
        if current is None:
            current = self._initial_state(hidden_states, layer_idx)

        if padding > 0:
            shortfall = max(0, padding - hidden_states.shape[-1])
            if shortfall:
                next_state = torch.cat(
                    (current[:, :, -shortfall:], hidden_states), dim=-1
                )
            else:
                next_state = hidden_states[:, :, -padding:]
        else:
            next_state = hidden_states.new_empty((hidden_states.shape[0], channels, 0))
        self.padding_cache[layer_idx] = next_state
        return current


# Causal convolutional codec.


class MimiConv1d(nn.Module):
    """Apply one-dimensional convolution with Mimi's asymmetric padding.

    Offline execution materializes the reference padding. Streaming execution
    instead prepends the corresponding layer's cached left context. A guarded
    long-form method can express verified constant padding inside cuDNN.
    """

    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: str | None = None,
        bias: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        """Construct one checkpoint-compatible convolution.

        Args:
            config: Model configuration supplying causal and padding defaults.
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Width of the convolution kernel.
            stride: Sampling stride along the temporal axis.
            dilation: Spacing between kernel elements.
            groups: Number of independent channel groups.
            pad_mode: Optional override for the configuration padding mode.
            bias: Whether the underlying convolution contains a bias tensor.
            layer_idx: Index into a streaming padding cache, or ``None`` when
                the layer is not used by streaming encoding.
        """
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.layer_idx = layer_idx
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        effective_kernel = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        stride_value = self.conv.stride[0]
        self.register_buffer(
            "stride", torch.tensor(stride_value, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "kernel_size",
            torch.tensor(effective_kernel, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padding_total",
            torch.tensor(effective_kernel - stride_value, dtype=torch.int64),
            persistent=False,
        )
        total = effective_kernel - stride_value
        self.padding_right = total // 2
        self.padding_left = total - self.padding_right

    def _get_extra_padding_for_conv1d(self, hidden_states: Tensor) -> int:
        """Compute right padding needed to align input length to the stride.

        Args:
            hidden_states: Tensor whose final dimension is the input length.

        Returns:
            Number of right-side samples needed for stride alignment.
        """
        length = hidden_states.shape[-1]
        return (-length) % self.conv.stride[0]

    @staticmethod
    def _pad1d(
        hidden_states: Tensor,
        paddings: tuple[int | Tensor, int | Tensor],
        mode: str,
        value: float = 0.0,
    ) -> Tensor:
        """Apply temporal padding, including safe short-signal reflection.

        Args:
            hidden_states: Tensor shaped ``[batch, channels, samples]``.
            paddings: Pair ``(left, right)`` of integer padding widths.
            mode: PyTorch padding mode such as ``constant`` or ``reflect``.
            value: Fill value used by constant padding.

        Returns:
            Padded tensor with the requested logical boundary values.
        """
        left, right = (int(item) for item in paddings)
        if mode != "reflect":
            return F.pad(hidden_states, (left, right), mode, value)
        extra = max(0, max(left, right) - hidden_states.shape[-1] + 1)
        if extra:
            hidden_states = F.pad(hidden_states, (0, extra))
        padded = F.pad(hidden_states, (left, right), mode, value)
        return padded[..., : padded.shape[-1] - extra] if extra else padded

    def _get_output_length(self, input_length: Tensor) -> Tensor:
        """Apply reference convolution length arithmetic to sample counts.

        Args:
            input_length: Scalar or batched tensor of unpadded input lengths.

        Returns:
            Int64 tensor containing the corresponding output lengths.
        """
        frames = (
            input_length - self.kernel_size + self.padding_total
        ) / self.stride + 1
        frames = torch.ceil(frames).to(torch.int64) - 1
        ideal_length = frames * self.stride + self.kernel_size - self.padding_total
        extra = ideal_length - input_length
        if self.causal:
            left, right = self.padding_total, extra
        else:
            left, right = self.padding_left, self.padding_right + extra
        length = input_length + left + right
        return (
            length
            + 2 * self.conv.padding[0]
            - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            - 1
        ) // self.conv.stride[0] + 1

    def forward(
        self, hidden_states: Tensor, padding_cache: MimiConv1dPaddingCache | None = None
    ) -> Tensor:
        """Pad and convolve offline input or consume streaming context.

        Args:
            hidden_states: Input tensor shaped ``[batch, channels, samples]``.
            padding_cache: Optional per-layer causal context. Supplying it
                selects streaming behavior and updates the cache in place.

        Returns:
            Convolved tensor with this layer's output channel count.

        Raises:
            ValueError: If streaming context is supplied to a non-causal layer.
        """
        extra = self._get_extra_padding_for_conv1d(hidden_states)
        if not self.causal and padding_cache is not None:
            raise ValueError("streaming padding cache requires causal convolutions")
        if self.causal and padding_cache is not None:
            cached = padding_cache.update(hidden_states, self.layer_idx)
            hidden_states = torch.cat((cached, hidden_states), dim=-1)
        elif self.causal:
            hidden_states = self._pad1d(
                hidden_states,
                (self.padding_left + self.padding_right, extra),
                self.pad_mode,
            )
        else:
            hidden_states = self._pad1d(
                hidden_states,
                (self.padding_left, self.padding_right + extra),
                self.pad_mode,
            )
        return self.conv(hidden_states)

    def _forward_with_builtin_causal_padding(self, hidden_states: Tensor) -> Tensor:
        """Fold eligible offline zero padding into the convolution operator.

        Args:
            hidden_states: Long-form tensor shaped
                ``[batch, in_channels, samples]``.

        Returns:
            Tensor bitwise equal to ordinary explicit causal padding, cropped to
            the exact reference output length.
        """
        if (
            not self.causal
            or self.pad_mode != "constant"
            or self.padding_left + self.padding_right == 0
        ):
            return self(hidden_states)

        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        effective_kernel = (self.conv.kernel_size[0] - 1) * dilation + 1
        total_padding = self.padding_left + self.padding_right
        extra = (-hidden_states.shape[-1]) % stride
        output_length = (
            hidden_states.shape[-1] + total_padding + extra - effective_kernel
        ) // stride + 1
        output = F.conv1d(
            hidden_states,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=total_padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return output[..., :output_length]


class MimiConvTranspose1d(nn.Module):
    """Upsample features with Mimi's exact causal trimming rule.

    The underlying transposed convolution creates boundary samples that the
    reference removes asymmetrically. Trimming is performed after convolution
    without changing checkpoint parameter paths.
    """

    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Construct one checkpoint-compatible transposed convolution.

        Args:
            config: Model configuration supplying causal trimming behavior.
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Width of the transposed-convolution kernel.
            stride: Temporal upsampling stride.
            groups: Number of independent channel groups.
            bias: Whether the underlying operation contains a bias tensor.

        Raises:
            ValueError: If asymmetric right trimming is requested for a
                non-causal layer.
        """
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )
        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError(
                "trim_right_ratio != 1 only applies to causal convolutions"
            )
        total = self.conv.kernel_size[0] - self.conv.stride[0]
        self.padding_right = (
            math.ceil(total * self.trim_right_ratio) if self.causal else total // 2
        )
        self.padding_left = total - self.padding_right

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Upsample and remove the reference boundary region.

        Args:
            hidden_states: Tensor shaped ``[batch, in_channels, frames]``.

        Returns:
            Trimmed tensor shaped ``[batch, out_channels, samples]``.
        """
        hidden_states = self.conv(hidden_states)
        end = hidden_states.shape[-1] - self.padding_right
        return hidden_states[..., self.padding_left : end]


class MimiResnetBlock(nn.Module):
    """Apply a two-convolution SEANet residual block.

    The main branch uses ELU, a dilated convolution, ELU, and a pointwise
    convolution. The shortcut is either identity or a checkpoint-compatible
    learned convolution.
    """

    def __init__(
        self, config: MimiConfig, dim: int, dilations: tuple[int, int] | list[int]
    ) -> None:
        """Construct the residual and shortcut branches.

        Args:
            config: Model configuration supplying kernels, compression, and
                shortcut behavior.
            dim: Input and output channel width of the block.
            dilations: Two dilation values corresponding to the residual and
                pointwise convolution kernels.

        Raises:
            ValueError: If ``dilations`` does not contain one entry per kernel.
        """
        super().__init__()
        hidden = dim // config.compress
        kernels = (config.residual_kernel_size, 1)
        if len(kernels) != len(dilations):
            raise ValueError("one dilation is required for every residual kernel")
        block: list[nn.Module] = []
        for index, (kernel, dilation) in enumerate(zip(kernels, dilations)):
            input_dim = dim if index == 0 else hidden
            output_dim = dim if index == len(kernels) - 1 else hidden
            block.extend(
                (
                    nn.ELU(),
                    MimiConv1d(
                        config, input_dim, output_dim, kernel, dilation=dilation
                    ),
                )
            )
        self.block = nn.ModuleList(block)
        self.shortcut: nn.Module = (
            MimiConv1d(config, dim, dim, 1)
            if config.use_conv_shortcut
            else nn.Identity()
        )

    def forward(
        self, hidden_states: Tensor, padding_cache: MimiConv1dPaddingCache | None = None
    ) -> Tensor:
        """Apply residual and shortcut branches in checkpoint operation order.

        Args:
            hidden_states: Input tensor shaped ``[batch, dim, samples]``.
            padding_cache: Optional shared streaming cache consumed by each
                causal convolution in the block.

        Returns:
            Elementwise sum of shortcut and residual branch outputs.
        """
        residual = hidden_states
        for layer in self.block:
            hidden_states = (
                layer(hidden_states, padding_cache)
                if isinstance(layer, MimiConv1d)
                else layer(hidden_states)
            )
        if isinstance(self.shortcut, MimiConv1d):
            residual = self.shortcut(residual, padding_cache)
        else:
            residual = self.shortcut(residual)
        return residual + hidden_states

    def _forward_with_builtin_causal_padding(self, hidden_states: Tensor) -> Tensor:
        """Run the verified long-form convolution-native padding variant.

        Args:
            hidden_states: Long-form tensor shaped ``[batch, dim, samples]``.

        Returns:
            Block output bitwise equal to the explicit-padding implementation.
        """
        residual = hidden_states
        for layer in self.block:
            hidden_states = (
                layer._forward_with_builtin_causal_padding(hidden_states)
                if isinstance(layer, MimiConv1d)
                else layer(hidden_states)
            )
        residual = (
            self.shortcut._forward_with_builtin_causal_padding(residual)
            if isinstance(self.shortcut, MimiConv1d)
            else self.shortcut(residual)
        )
        return residual + hidden_states


class MimiEncoder(nn.Module):
    """Convert waveforms to 25 Hz embeddings with causal SEANet layers.

    Fixed one-second and 100-second float32 SM120 inputs may use separately
    compiled graphs. Every other device, shape, dtype, mode, or failure uses the
    same eager layer sequence used by streaming execution.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Construct the analysis stack and optional dispatch state.

        Args:
            config: Complete Mimi configuration defining channel widths,
                residual blocks, strides, kernels, and optimization guards.
        """
        super().__init__()
        model: list[nn.Module] = [
            MimiConv1d(
                config, config.audio_channels, config.num_filters, config.kernel_size
            )
        ]
        conv_names = ["layers.0"]
        scale = 1
        for ratio in reversed(config.upsampling_ratios):
            channels = scale * config.num_filters
            for residual_index in range(config.num_residual_layers):
                conv_names.extend(
                    (f"layers.{len(model)}.block.1", f"layers.{len(model)}.block.3")
                )
                model.append(
                    MimiResnetBlock(
                        config,
                        channels,
                        (config.dilation_growth_rate**residual_index, 1),
                    )
                )
            model.append(nn.ELU())
            conv_names.append(f"layers.{len(model)}")
            model.append(
                MimiConv1d(config, channels, channels * 2, ratio * 2, stride=ratio)
            )
            scale *= 2
        model.append(nn.ELU())
        conv_names.append(f"layers.{len(model)}")
        model.append(
            MimiConv1d(
                config,
                scale * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
            )
        )
        self.layers = nn.ModuleList(model)
        self._mimiconv1d_layer_names = conv_names
        for index, name in enumerate(conv_names):
            self.get_submodule(name).layer_idx = index
        self._compiled_forward = None
        self._compiled_forward_failed = False
        self._compiled_forward_long = None
        self._compiled_forward_long_failed = False
        self._compiled_forward_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_COMPILED_ENCODER"
        )
        self._cudnn_benchmark_prime_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_CUDNN_BENCHMARK_PRIME"
        )
        self._long_builtin_padding_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_LONG_BUILTIN_PADDING"
        )

    def _forward_eager(
        self, hidden_states: Tensor, padding_cache: MimiConv1dPaddingCache | None = None
    ) -> Tensor:
        """Execute the portable encoder layer sequence.

        Args:
            hidden_states: Waveform or intermediate tensor entering the stack.
            padding_cache: Optional streaming context shared by causal layers.

        Returns:
            Encoded tensor shaped ``[batch, hidden_size, encodec_frames]``.
        """
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states

    def _forward_offline(self, hidden_states: Tensor) -> Tensor:
        """Expose a cache-free target for fixed-shape compilation.

        Args:
            hidden_states: Offline waveform tensor.

        Returns:
            Same tensor result as ``_forward_eager`` without a cache.
        """
        return self._forward_eager(hidden_states)

    def _forward_long_with_builtin_padding(self, hidden_states: Tensor) -> Tensor:
        """Run long-form encoding with convolution-native causal padding.

        Args:
            hidden_states: Fixed 100-second waveform tensor.

        Returns:
            Contiguous encoder output bitwise equal to explicit padding.
        """
        for layer in self.layers:
            hidden_states = (
                layer._forward_with_builtin_causal_padding(hidden_states)
                if isinstance(layer, (MimiConv1d, MimiResnetBlock))
                else layer(hidden_states)
            )
        return hidden_states.contiguous()

    def _can_use_compiled_forward(
        self,
        hidden_states: Tensor,
        padding_cache: MimiConv1dPaddingCache | None,
    ) -> bool:
        """Check whether a profiled fixed-shape encoder graph is eligible.

        Args:
            hidden_states: Candidate waveform tensor.
            padding_cache: Optional streaming state, which always disables
                compiled offline dispatch.

        Returns:
            ``True`` only when mode, shape, layout, dtype, device, architecture,
            compiler availability, and failure state match the verified path.
        """
        shape = tuple(hidden_states.shape)
        failed = (
            self._compiled_forward_long_failed
            if shape == _PROFILED_LONG_AUDIO_SHAPE
            else self._compiled_forward_failed
        )
        return (
            self._compiled_forward_enabled
            and not failed
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and padding_cache is None
            and hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.float32
            and shape in {_PROFILED_ONE_SECOND_AUDIO_SHAPE, _PROFILED_LONG_AUDIO_SHAPE}
            and hidden_states.is_contiguous()
            and torch.cuda.get_device_capability(hidden_states.device)
            == _PROFILED_SM_CAPABILITY
            and hasattr(torch, "compile")
        )

    def _clear_runtime_caches(self) -> None:
        """Drop compiled functions and reset fail-closed state.

        Returns:
            ``None`` after one-second and long-form runtime state is cleared.
        """
        self._compiled_forward = None
        self._compiled_forward_failed = False
        self._compiled_forward_long = None
        self._compiled_forward_long_failed = False

    def _apply(self, fn, recurse: bool = True):
        """Invalidate compiled state before applying a tensor transformation.

        Args:
            fn: Function applied by ``nn.Module._apply`` to parameters and
                buffers, commonly for a device or dtype move.
            recurse: Whether the transformation descends into child modules.

        Returns:
            This module after PyTorch applies ``fn``.
        """
        self._clear_runtime_caches()
        return super()._apply(fn, recurse)

    def forward(
        self, hidden_states: Tensor, padding_cache: MimiConv1dPaddingCache | None = None
    ) -> Tensor:
        """Encode audio through a compiled graph or portable fallback.

        Args:
            hidden_states: Waveform tensor shaped ``[batch, channels, samples]``.
            padding_cache: Optional streaming convolution context.

        Returns:
            Encoder embeddings shaped ``[batch, hidden_size, frames]``.
        """
        if self._can_use_compiled_forward(hidden_states, padding_cache):
            is_long = tuple(hidden_states.shape) == _PROFILED_LONG_AUDIO_SHAPE
            compiled_name = "_compiled_forward_long" if is_long else "_compiled_forward"
            failed_name = (
                "_compiled_forward_long_failed"
                if is_long
                else "_compiled_forward_failed"
            )
            try:
                compiled_forward = getattr(self, compiled_name)
                if compiled_forward is None:
                    if torch.cuda.is_current_stream_capturing():
                        return self._forward_eager(hidden_states, padding_cache)
                    compile_target = (
                        self._forward_long_with_builtin_padding
                        if is_long
                        and self._long_builtin_padding_enabled
                        and _can_prime_long_cudnn_benchmark()
                        else self._forward_offline
                    )
                    compiled_forward = torch.compile(
                        compile_target,
                        backend="inductor",
                        fullgraph=True,
                        dynamic=False,
                        mode="default",
                    )
                    setattr(self, compiled_name, compiled_forward)
                    if (
                        is_long
                        and self._cudnn_benchmark_prime_enabled
                        and _can_prime_long_cudnn_benchmark()
                    ):
                        return _run_with_long_cudnn_benchmark(
                            compiled_forward, hidden_states
                        )
                return compiled_forward(hidden_states)
            except Exception:  # noqa: BLE001 - optional compilation fails closed.
                setattr(self, compiled_name, None)
                setattr(self, failed_name, True)
        return self._forward_eager(hidden_states, padding_cache)


class MimiDecoder(nn.Module):
    """Convert context embeddings back to a 24 kHz waveform.

    Fixed one-second and 100-second float32 SM120 inputs may use separate
    compiled synthesis graphs. All unsupported conditions and compilation
    failures execute the checkpoint-ordered eager stack.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Construct the synthesis stack and optional dispatch state.

        Args:
            config: Complete Mimi configuration defining widths, residual
                blocks, transposed-convolution strides, and optimization guards.
        """
        super().__init__()
        scale = 2 ** len(config.upsampling_ratios)
        model: list[nn.Module] = [
            MimiConv1d(
                config,
                config.hidden_size,
                scale * config.num_filters,
                config.kernel_size,
            )
        ]
        for ratio in config.upsampling_ratios:
            channels = scale * config.num_filters
            model.extend(
                (
                    nn.ELU(),
                    MimiConvTranspose1d(
                        config, channels, channels // 2, ratio * 2, stride=ratio
                    ),
                )
            )
            for residual_index in range(config.num_residual_layers):
                model.append(
                    MimiResnetBlock(
                        config,
                        channels // 2,
                        (config.dilation_growth_rate**residual_index, 1),
                    )
                )
            scale //= 2
        model.extend(
            (
                nn.ELU(),
                MimiConv1d(
                    config,
                    config.num_filters,
                    config.audio_channels,
                    config.last_kernel_size,
                ),
            )
        )
        self.layers = nn.ModuleList(model)
        self._compiled_forward = None
        self._compiled_forward_failed = False
        self._compiled_forward_long = None
        self._compiled_forward_long_failed = False
        self._compiled_forward_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_COMPILED_DECODER"
        )
        self._cudnn_benchmark_prime_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_CUDNN_BENCHMARK_PRIME"
        )
        self._long_builtin_padding_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_LONG_BUILTIN_PADDING"
        )

    def _forward_eager(self, hidden_states: Tensor) -> Tensor:
        """Execute the portable synthesis layer sequence.

        Args:
            hidden_states: Context embeddings shaped
                ``[batch, hidden_size, frames]``.

        Returns:
            Reconstructed waveform tensor.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def _forward_long_with_builtin_padding(self, hidden_states: Tensor) -> Tensor:
        """Run long-form decoding with convolution-native causal padding.

        Args:
            hidden_states: Fixed long-form context tensor shaped
                ``[1, hidden_size, 2500]``.

        Returns:
            Canonically strided waveform bitwise equal to explicit padding.
        """
        for layer in self.layers:
            hidden_states = (
                layer._forward_with_builtin_causal_padding(hidden_states)
                if isinstance(layer, (MimiConv1d, MimiResnetBlock))
                else layer(hidden_states)
            )
        length = hidden_states.shape[-1]
        channels = hidden_states.shape[-2]
        # Prefix slicing preserves values but inherits the wider cuDNN backing
        # stride. The verified view restores the canonical contiguous strides
        # expected by the reference decoder output and waveform parity checks.
        return hidden_states.as_strided(
            hidden_states.shape,
            (channels * length, length, 1),
        )

    def _can_use_compiled_forward(self, hidden_states: Tensor) -> bool:
        """Check whether a profiled fixed-shape decoder graph is eligible.

        Args:
            hidden_states: Candidate decoder input tensor.

        Returns:
            ``True`` only when mode, shape, layout, dtype, device, architecture,
            compiler availability, and failure state match the verified path.
        """
        shape = tuple(hidden_states.shape)
        failed = (
            self._compiled_forward_long_failed
            if shape == _PROFILED_LONG_DECODER_SHAPE
            else self._compiled_forward_failed
        )
        return (
            self._compiled_forward_enabled
            and not failed
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.float32
            and shape
            in {_PROFILED_ONE_SECOND_DECODER_SHAPE, _PROFILED_LONG_DECODER_SHAPE}
            and hidden_states.is_contiguous()
            and torch.cuda.get_device_capability(hidden_states.device)
            == _PROFILED_SM_CAPABILITY
            and hasattr(torch, "compile")
        )

    def _clear_runtime_caches(self) -> None:
        """Drop compiled functions and reset fail-closed state.

        Returns:
            ``None`` after one-second and long-form runtime state is cleared.
        """
        self._compiled_forward = None
        self._compiled_forward_failed = False
        self._compiled_forward_long = None
        self._compiled_forward_long_failed = False

    def _apply(self, fn, recurse: bool = True):
        """Invalidate compiled state before applying a tensor transformation.

        Args:
            fn: Function applied by ``nn.Module._apply`` to parameters and
                buffers, commonly for a device or dtype move.
            recurse: Whether the transformation descends into child modules.

        Returns:
            This module after PyTorch applies ``fn``.
        """
        self._clear_runtime_caches()
        return super()._apply(fn, recurse)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Decode embeddings through a compiled graph or portable fallback.

        Args:
            hidden_states: Tensor shaped ``[batch, hidden_size, frames]``.

        Returns:
            Reconstructed waveform shaped ``[batch, channels, samples]``.
        """
        if self._can_use_compiled_forward(hidden_states):
            is_long = tuple(hidden_states.shape) == _PROFILED_LONG_DECODER_SHAPE
            compiled_name = "_compiled_forward_long" if is_long else "_compiled_forward"
            failed_name = (
                "_compiled_forward_long_failed"
                if is_long
                else "_compiled_forward_failed"
            )
            try:
                compiled_forward = getattr(self, compiled_name)
                if compiled_forward is None:
                    if torch.cuda.is_current_stream_capturing():
                        return self._forward_eager(hidden_states)
                    compile_target = (
                        self._forward_long_with_builtin_padding
                        if is_long
                        and self._long_builtin_padding_enabled
                        and _can_prime_long_cudnn_benchmark()
                        else self._forward_eager
                    )
                    compiled_forward = torch.compile(
                        compile_target,
                        backend="inductor",
                        fullgraph=True,
                        dynamic=False,
                        mode="default",
                    )
                    setattr(self, compiled_name, compiled_forward)
                    if (
                        is_long
                        and self._cudnn_benchmark_prime_enabled
                        and _can_prime_long_cudnn_benchmark()
                    ):
                        return _run_with_long_cudnn_benchmark(
                            compiled_forward, hidden_states
                        )
                return compiled_forward(hidden_states)
            except Exception:  # noqa: BLE001 - optional compilation fails closed.
                setattr(self, compiled_name, None)
                setattr(self, failed_name, True)
        return self._forward_eager(hidden_states)


# Transformer context model.


class MimiLayerScale(nn.Module):
    """Apply learned per-channel scaling to a residual branch.

    The scale vector is a persistent checkpoint parameter shared across every
    temporal position and batch item.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Initialize the scale vector from the model configuration.

        Args:
            config: Configuration supplying hidden width and initial scale.
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_initial_scale)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Scale every hidden channel.

        Args:
            hidden_states: Tensor whose final dimension is ``hidden_size``.

        Returns:
            Elementwise product with the learned scale vector.
        """
        return self.scale * hidden_states


class MimiRotaryEmbedding(nn.Module):
    """Generate checkpoint-compatible rotary position embeddings.

    Inverse frequencies are a non-persistent derived buffer. Trigonometric
    tables are always computed with autocast disabled before conversion back to
    the hidden-state dtype.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Precompute inverse frequencies for one attention head.

        Args:
            config: Configuration supplying ``rope_theta`` and ``head_dim``.
        """
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.int64).float()
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, hidden_states: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Build cosine and sine tables for absolute sequence positions.

        Args:
            hidden_states: Tensor used to select output device and dtype.
            position_ids: Absolute positions shaped ``[batch, frames]``.

        Returns:
            Pair of cosine and sine tensors shaped
            ``[batch, frames, head_dim]``.
        """
        inv_freq = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(hidden_states.device)
        )
        positions = position_ids[:, None, :].float()
        device_type = (
            hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (inv_freq.float() @ positions.float()).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine, sine = embedding.cos(), embedding.sin()
        return cosine.to(hidden_states.dtype), sine.to(hidden_states.dtype)


def _rotate_half(hidden_states: Tensor) -> Tensor:
    """Rotate paired feature halves for RoPE.

    Args:
        hidden_states: Tensor whose final dimension contains two equal halves.

    Returns:
        Concatenation of the negated second half and original first half.
    """
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to query and key projections.

    Args:
        query: Query tensor shaped ``[batch, heads, frames, head_dim]``.
        key: Key tensor with the same frame and head dimensions.
        cosine: Cosine table shaped ``[batch, frames, head_dim]``.
        sine: Sine table shaped ``[batch, frames, head_dim]``.

    Returns:
        Rotated query and key tensors in that order.
    """
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return query * cosine + _rotate_half(query) * sine, key * cosine + _rotate_half(
        key
    ) * sine


def _repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    """Expand grouped key/value heads to the query-head count.

    Args:
        hidden_states: Tensor shaped ``[batch, kv_heads, frames, head_dim]``.
        repeats: Number of query-head groups sharing each key/value head.

    Returns:
        Tensor shaped ``[batch, kv_heads * repeats, frames, head_dim]``.
    """
    if repeats == 1:
        return hidden_states
    batch, heads, length, dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(batch, heads, repeats, length, dim)
    return hidden_states.reshape(batch, heads * repeats, length, dim)


def _native_window_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    window: int,
) -> Tensor:
    """Run PyTorch's verified CUTLASS causal-window attention operator.

    Args:
        query: Float32 query tensor shaped ``[batch, heads, frames, head_dim]``.
        key: Float32 key tensor with the same shape and verified stride.
        value: Float32 value tensor with the same shape and verified stride.
        scale: Scalar applied to query-key scores.
        window: Number of causal frames visible to each query.

    Returns:
        Attention result shaped like ``query``.
    """
    output = torch.ops.aten._efficient_attention_forward(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        None,
        None,
        None,
        None,
        None,
        0.0,
        1,
        False,
        scale=scale,
        window_size=window,
    )[0]
    return output.transpose(1, 2)


def _attention_mask(
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    query_offset: int,
    key_offset: int,
    sliding_window: int,
    padding_mask: Tensor | None,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    """Build Mimi's causal sliding-window mask at absolute positions.

    Args:
        batch_size: Number of sequences represented by the mask.
        query_length: Number of current query frames.
        key_length: Number of visible current and cached key frames.
        query_offset: Absolute position of the first query frame.
        key_offset: Absolute position of the first physically retained key.
        sliding_window: Maximum causal window width in frames.
        padding_mask: Optional key-validity mask, or a prebuilt four-dimensional
            attention mask returned unchanged.
        backend: ``"sdpa"`` for a boolean mask or ``"eager"`` for an additive
            float mask.
        dtype: Floating dtype used by an eager additive mask.
        device: Device on which mask tensors are allocated.

    Returns:
        Boolean SDPA mask, additive eager mask, supplied four-dimensional mask,
        or ``None`` when SDPA can express the required causality directly.
    """
    if padding_mask is not None and padding_mask.ndim == 4:
        return padding_mask

    can_skip = (
        backend == "sdpa"
        and key_length < sliding_window
        and (query_length == 1 or key_length == query_length or query_offset == 0)
        and (padding_mask is None or bool(padding_mask.all()))
    )
    if can_skip:
        return None

    query_positions = torch.arange(query_length, device=device)[:, None] + query_offset
    key_positions = torch.arange(key_length, device=device)[None, :] + key_offset
    allowed = (key_positions <= query_positions) & (
        key_positions > query_positions - sliding_window
    )
    allowed = allowed[None, None].expand(batch_size, 1, query_length, key_length)
    if padding_mask is not None:
        missing = key_offset + key_length - padding_mask.shape[-1]
        if missing > 0:
            padding_mask = F.pad(padding_mask, (0, missing))
        key_indices = torch.arange(key_length, device=device) + key_offset
        key_padding = padding_mask[:, key_indices].bool()[:, None, None, :]
        allowed = allowed & key_padding
    if backend == "sdpa":
        return allowed
    minimum = torch.finfo(dtype).min
    return torch.where(allowed, torch.tensor(0.0, dtype=dtype, device=device), minimum)


class MimiAttention(nn.Module):
    """Apply sliding-window causal attention with eager or SDPA execution.

    Query, key, value, and output projections preserve checkpoint names and
    operation order. The fixed long-form SDPA workload may dispatch to a native
    CUTLASS window operator after every numerical and layout guard passes.
    """

    def __init__(self, config: MimiConfig, layer_idx: int, backend: str) -> None:
        """Construct projections and cache the attention contract.

        Args:
            config: Configuration supplying head counts, dimensions, dropout,
                window size, scaling inputs, and projection bias behavior.
            layer_idx: Zero-based index used to update the matching KV cache.
            backend: ``"sdpa"`` for PyTorch scaled-dot-product attention or
                ``"eager"`` for explicit score, softmax, and value operations.

        Raises:
            ValueError: If ``backend`` is not ``"sdpa"`` or ``"eager"``.
        """
        super().__init__()
        if backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be 'sdpa' or 'eager'")
        self.layer_idx = layer_idx
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / math.sqrt(config.head_dim)
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window
        self._native_window_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_NATIVE_WINDOW_ATTENTION"
        )
        self._native_window_failed = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def _can_use_native_window_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        canonical_window_mask: bool,
        dropout: float,
    ) -> bool:
        """Check the native SM120 local-attention dispatch contract.

        Args:
            query: Projected and rotated query tensor.
            key: Projected, rotated, and head-expanded key tensor.
            value: Projected and head-expanded value tensor.
            canonical_window_mask: Whether the caller requires the fixed
                unpadded 2,500-frame causal window.
            dropout: Effective attention dropout probability for this call.

        Returns:
            ``True`` only when shape, stride, dtype, mode, software versions,
            architecture, TF32 settings, and prior failure state all match the
            verified CUTLASS path.
        """
        return (
            self._native_window_enabled
            and not self._native_window_failed
            and canonical_window_mask
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and query.device.type == "cuda"
            and query.dtype == torch.float32
            and key.dtype == torch.float32
            and value.dtype == torch.float32
            and tuple(query.shape) == _PROFILED_LONG_ATTENTION_SHAPE
            and tuple(key.shape) == _PROFILED_LONG_ATTENTION_SHAPE
            and tuple(value.shape) == _PROFILED_LONG_ATTENTION_SHAPE
            and tuple(query.stride()) == _PROFILED_LONG_ATTENTION_STRIDE
            and tuple(key.stride()) == _PROFILED_LONG_ATTENTION_STRIDE
            and tuple(value.stride()) == _PROFILED_LONG_ATTENTION_STRIDE
            and self.sliding_window == 250
            and self.scaling == 0.125
            and dropout == 0.0
            and torch.__version__.split("+", 1)[0] == "2.13.0"
            and torch.version.cuda == "13.0"
            and torch.cuda.get_device_capability(query.device)
            == _PROFILED_SM_CAPABILITY
            and not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32
            and hasattr(torch.ops.aten, "_efficient_attention_forward")
        )

    def _clear_runtime_caches(self) -> None:
        """Clear fail-closed native-attention state.

        Returns:
            ``None`` after a later eligible call is allowed to retry dispatch.
        """
        self._native_window_failed = False

    def _apply(self, fn, recurse: bool = True):
        """Reset backend state before applying a tensor transformation.

        Args:
            fn: Function applied by ``nn.Module._apply`` to parameters and
                buffers.
            recurse: Whether the transformation descends into child modules.

        Returns:
            This module after PyTorch applies ``fn``.
        """
        self._clear_runtime_caches()
        return super()._apply(fn, recurse)

    def _sdpa_fallback(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        canonical_window_mask: bool,
        dropout: float,
    ) -> Tensor:
        """Run public SDPA and reconstruct an omitted mask when necessary.

        Args:
            query: Query tensor shaped ``[batch, heads, query_frames, head_dim]``.
            key: Key tensor shaped ``[batch, heads, key_frames, head_dim]``.
            value: Value tensor matching ``key``.
            attention_mask: Boolean or additive attention mask, or ``None``.
            canonical_window_mask: Whether ``None`` represents the omitted
                fixed long-form mask rather than unrestricted causality.
            dropout: Effective attention dropout probability.

        Returns:
            SDPA result shaped like ``query``.
        """
        if attention_mask is None and canonical_window_mask:
            attention_mask = _attention_mask(
                batch_size=query.shape[0],
                query_length=query.shape[-2],
                key_length=key.shape[-2],
                query_offset=0,
                key_offset=0,
                sliding_window=self.sliding_window,
                padding_mask=None,
                backend="sdpa",
                dtype=query.dtype,
                device=query.device,
            )
        is_causal = query.shape[-2] > 1 and attention_mask is None
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=self.scaling,
            is_causal=is_causal,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        past_key_values: MimiKVCache | None,
        position_embeddings: tuple[Tensor, Tensor],
        output_attentions: bool = False,
        canonical_window_mask: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Project Q/K/V and apply one local-attention layer.

        Args:
            hidden_states: Input tensor shaped
                ``[batch, query_frames, hidden_size]``.
            attention_mask: Optional boolean or additive attention mask.
            past_key_values: Optional sliding cache updated at ``layer_idx``.
            position_embeddings: Pair of cosine and sine RoPE tables.
            output_attentions: Whether eager execution returns attention
                probabilities. SDPA intentionally returns ``None``.
            canonical_window_mask: Whether this is the fixed unpadded
                2,500-frame workload eligible for native window attention.

        Returns:
            Pair containing projected attention output and optional eager
            attention probabilities.
        """
        batch, length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        query, key = _apply_rope(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        dropout = self.attention_dropout if self.training else 0.0
        weights: Tensor | None = None
        if self.backend == "sdpa":
            if self._can_use_native_window_attention(
                query, key, value, canonical_window_mask, dropout
            ):
                try:
                    output = _native_window_attention(
                        query,
                        key,
                        value,
                        scale=self.scaling,
                        window=self.sliding_window,
                    )
                except Exception:  # noqa: BLE001 - optional backend fails closed.
                    self._native_window_failed = True
                    output = self._sdpa_fallback(
                        query,
                        key,
                        value,
                        attention_mask,
                        canonical_window_mask,
                        dropout,
                    )
            else:
                output = self._sdpa_fallback(
                    query,
                    key,
                    value,
                    attention_mask,
                    canonical_window_mask,
                    dropout,
                )
            output = output.transpose(1, 2).contiguous()
        else:
            weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                weights = weights + attention_mask
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=self.training)
            output = torch.matmul(weights, value).transpose(1, 2).contiguous()
            if not output_attentions:
                weights = None
        return self.o_proj(output.reshape(batch, length, -1).contiguous()), weights


class MimiMLP(nn.Module):
    """Apply Mimi's two-layer GELU feed-forward network.

    Both linear layers are bias-free checkpoint parameters. Their order and
    PyTorch GELU implementation are preserved because alternative compiled GELU
    approximations changed float32 output bits during profiling.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Construct the feed-forward projections.

        Args:
            config: Configuration supplying hidden width, intermediate width,
                and activation name.

        Raises:
            ValueError: If the configured activation is not ``"gelu"``.
        """
        super().__init__()
        if config.hidden_act != "gelu":
            raise ValueError(
                "the independent runtime currently supports Mimi's GELU activation"
            )
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply input projection, GELU, and output projection in order.

        Args:
            hidden_states: Tensor shaped ``[batch, frames, hidden_size]``.

        Returns:
            Tensor with the same shape as ``hidden_states``.
        """
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class MimiTransformerLayer(nn.Module):
    """Combine pre-norm attention and MLP residual branches.

    Each branch is multiplied by its own learned layer-scale vector before the
    residual addition. The additions remain separate to preserve checkpoint
    float32 operation order.
    """

    def __init__(self, config: MimiConfig, layer_idx: int, backend: str) -> None:
        """Construct one context layer.

        Args:
            config: Transformer and normalization configuration.
            layer_idx: Zero-based index used by this layer's KV cache.
            backend: Attention backend, either ``"sdpa"`` or ``"eager"``.
        """
        super().__init__()
        self.self_attn = MimiAttention(config, layer_idx, backend)
        self.mlp = MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        past_key_values: MimiKVCache | None,
        position_embeddings: tuple[Tensor, Tensor],
        output_attentions: bool,
        canonical_window_mask: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply attention and MLP residual branches.

        Args:
            hidden_states: Input shaped ``[batch, frames, hidden_size]``.
            attention_mask: Optional local causal or padding mask.
            past_key_values: Optional shared sliding cache updated by attention.
            position_embeddings: Pair of cosine and sine RoPE tables.
            output_attentions: Whether eager attention probabilities are kept.
            canonical_window_mask: Whether the fixed long native-window
                specialization is semantically valid.

        Returns:
            Updated hidden states and optional eager attention probabilities.
        """
        attention_output, weights = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            past_key_values,
            position_embeddings,
            output_attentions,
            canonical_window_mask,
        )
        hidden_states = hidden_states + self.self_attn_layer_scale(attention_output)
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.mlp_layer_scale(mlp_output), weights


class MimiTransformerModel(nn.Module):
    """Run a stack of causal local-attention blocks with RoPE.

    The model owns mask construction, absolute cache positions, optional hidden
    and attention diagnostics, and the guarded omission of the dense long-form
    mask consumed directly by native-window attention.
    """

    def __init__(self, config: MimiConfig, backend: str) -> None:
        """Construct all Transformer layers and shared rotary embeddings.

        Args:
            config: Configuration defining layer count, dimensions, window,
                caching defaults, and output behavior.
            backend: Attention implementation passed to every layer; accepted
                values are ``"sdpa"`` and ``"eager"``.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            MimiTransformerLayer(config, layer_idx, backend)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.rotary_emb = MimiRotaryEmbedding(config)
        self.config = config
        self.backend = backend
        self._omit_native_window_mask_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_NATIVE_WINDOW_MASK_OMISSION"
        )

    def _can_omit_native_window_mask(
        self, hidden_states: Tensor, canonical_window_mask: bool
    ) -> bool:
        """Check whether dense long-form mask allocation can be omitted.

        Args:
            hidden_states: Candidate Transformer input tensor.
            canonical_window_mask: Whether input semantics require exactly the
                fixed unpadded causal 250-frame window.

        Returns:
            ``True`` only when every attention layer can consume the native
            window and rebuild the exact SDPA mask if a native call fails.
        """
        return (
            self._omit_native_window_mask_enabled
            and canonical_window_mask
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.float32
            and tuple(hidden_states.shape) == _PROFILED_LONG_TRANSFORMER_SHAPE
            and torch.__version__.split("+", 1)[0] == "2.13.0"
            and torch.version.cuda == "13.0"
            and torch.cuda.get_device_capability(hidden_states.device)
            == _PROFILED_SM_CAPABILITY
            and not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32
            and all(
                layer.self_attn._native_window_enabled
                and not layer.self_attn._native_window_failed
                for layer in self.layers
            )
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: MimiKVCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool | None = None,
    ) -> _TransformerOutput | tuple[Any, ...]:
        """Run all context layers while maintaining absolute cache positions.

        Args:
            hidden_states: Input shaped ``[batch, frames, hidden_size]``.
            attention_mask: Optional two-dimensional padding mask or prebuilt
                four-dimensional attention mask.
            position_ids: Optional absolute positions shaped ``[batch, frames]``;
                omitted positions are derived from cache length.
            past_key_values: Optional sliding KV cache shared across layers.
            use_cache: Whether to create and return a cache when none is passed;
                ``None`` uses the configuration default.
            output_attentions: Whether eager attention maps are collected.
            output_hidden_states: Whether input and per-layer hidden tensors are
                collected.
            return_dict: Whether to return ``_TransformerOutput``; ``None`` uses
                the configuration default and ``False`` returns a tuple.

        Returns:
            Named or tuple output containing final hidden state and any enabled
            cache or diagnostics.
        """
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.return_dict if return_dict is None else return_dict
        canonical_window_mask = (
            not use_cache
            and past_key_values is None
            and attention_mask is None
            and self.backend == "sdpa"
            and tuple(hidden_states.shape) == _PROFILED_LONG_TRANSFORMER_SHAPE
        )
        # PT-047 omits allocation only when every layer can consume the same
        # canonical window directly. Any failure reconstructs this exact mask.
        if use_cache and past_key_values is None:
            past_key_values = MimiKVCache(
                len(self.layers), sliding_window=self.config.sliding_window
            )
        if past_key_values is not None:
            past_key_values.configure_sliding_window(self.config.sliding_window)
        past_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + hidden_states.shape[1],
                device=hidden_states.device,
            ).unsqueeze(0)
        key_length, key_offset = (
            past_key_values.get_mask_sizes(hidden_states.shape[1])
            if past_key_values is not None
            else (hidden_states.shape[1], 0)
        )
        mask = None
        if not self._can_omit_native_window_mask(hidden_states, canonical_window_mask):
            mask = _attention_mask(
                batch_size=hidden_states.shape[0],
                query_length=hidden_states.shape[1],
                key_length=key_length,
                query_offset=past_length,
                key_offset=key_offset,
                sliding_window=self.config.sliding_window,
                padding_mask=attention_mask,
                backend=self.backend,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden: list[Tensor] | None = [] if output_hidden_states else None
        all_attentions: list[Tensor] | None = [] if output_attentions else None
        for layer in self.layers:
            if all_hidden is not None:
                all_hidden.append(hidden_states)
            hidden_states, weights = layer(
                hidden_states,
                mask,
                past_key_values,
                position_embeddings,
                output_attentions,
                canonical_window_mask,
            )
            if all_attentions is not None and weights is not None:
                all_attentions.append(weights)
        if all_hidden is not None:
            all_hidden.append(hidden_states)
        values = (
            hidden_states,
            past_key_values,
            tuple(all_hidden) if all_hidden is not None else None,
            tuple(all_attentions) if all_attentions is not None else None,
        )
        if not return_dict:
            return tuple(item for item in values if item is not None)
        return _TransformerOutput(*values)


# Residual vector quantization.


class MimiEuclideanCodebook(nn.Module):
    """Search one checkpoint-backed Euclidean codebook.

    Persistent ``embed_sum`` and ``cluster_usage`` buffers preserve checkpoint
    names. Their normalized quotient is derived lazily and invalidated on every
    device or dtype transformation.
    """

    def __init__(self, config: MimiConfig, epsilon: float = 1e-5) -> None:
        """Allocate checkpoint-compatible codebook buffers.

        Args:
            config: Configuration supplying codebook entry count and dimension.
            epsilon: Lower bound applied to cluster usage during normalization.
        """
        super().__init__()
        self.codebook_size = config.codebook_size
        self.epsilon = epsilon
        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer(
            "embed_sum", torch.zeros(config.codebook_size, config.codebook_dim)
        )
        self._embed: Tensor | None = None

    @property
    def embed(self) -> Tensor:
        """Materialize normalized codebook vectors on first access.

        Returns:
            Tensor shaped ``[codebook_size, codebook_dim]`` derived from
            checkpoint buffers and cached until a model transformation.
        """
        if self._embed is None:
            self._embed = (
                self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
        return self._embed

    def quantize(self, hidden_states: Tensor) -> Tensor:
        """Select the nearest codebook entry for every feature vector.

        Args:
            hidden_states: Two-dimensional tensor shaped
                ``[vectors, codebook_dim]``.

        Returns:
            Int64 nearest-entry indices. ``argmin`` preserves lowest-index tie
            behavior from the reference.
        """
        distances = torch.cdist(
            hidden_states[None].float(), self.embed[None].float(), p=2
        )[0]
        return distances.argmin(dim=-1)

    def encode(self, hidden_states: Tensor) -> Tensor:
        """Quantize an arbitrary leading shape of feature vectors.

        Args:
            hidden_states: Tensor whose final dimension is ``codebook_dim``.

        Returns:
            Int64 index tensor with the input's final dimension removed.
        """
        shape = hidden_states.shape
        return self.quantize(hidden_states.reshape(-1, shape[-1])).view(*shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        """Gather codebook vectors for discrete indices.

        Args:
            indices: Int64 tensor containing codebook entry indices.

        Returns:
            Float tensor with ``codebook_dim`` appended as the final dimension.
        """
        return F.embedding(indices, self.embed)

    def _apply(self, fn, recurse: bool = True):
        """Invalidate derived vectors before applying a tensor transformation.

        Args:
            fn: Function applied by ``nn.Module._apply`` to persistent buffers.
            recurse: Whether the transformation descends into child modules.

        Returns:
            This module after PyTorch applies ``fn``.
        """
        self._embed = None
        return super()._apply(fn, recurse)


class MimiVectorQuantization(nn.Module):
    """Adapt one Euclidean codebook to channel-first codec tensors.

    Encoding permutes ``[batch, channels, frames]`` into a final-feature layout;
    decoding reverses that permutation without changing codebook arithmetic.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Construct one vector-quantization layer.

        Args:
            config: Configuration used to size the Euclidean codebook.
        """
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, hidden_states: Tensor) -> Tensor:
        """Encode channel-first features with one codebook.

        Args:
            hidden_states: Tensor shaped ``[batch, codebook_dim, frames]``.

        Returns:
            Int64 indices shaped ``[batch, frames]``.
        """
        return self.codebook.encode(hidden_states.permute(0, 2, 1))

    def decode(self, indices: Tensor) -> Tensor:
        """Decode one layer of codebook indices.

        Args:
            indices: Int64 tensor shaped ``[batch, frames]``.

        Returns:
            Float tensor shaped ``[batch, codebook_dim, frames]``.
        """
        return self.codebook.decode(indices).permute(0, 2, 1)


class MimiResidualVectorQuantizer(nn.Module):
    """Encode or decode an ordered residual vector-quantizer stack.

    Every encode step subtracts its selected vector before the next search.
    Decode starts from a float32 scalar zero and adds layers left to right; this
    exact accumulation order is required for bitwise waveform parity.
    """

    def __init__(self, config: MimiConfig, num_quantizers: int | None = None) -> None:
        """Construct a residual stack and optional channel projections.

        Args:
            config: Configuration supplying dimensions and maximum layer count.
            num_quantizers: Number of codebook layers in this branch, or
                ``None`` to use the configuration maximum.
        """
        super().__init__()
        self.num_quantizers = (
            config.num_quantizers if num_quantizers is None else num_quantizers
        )
        self.layers = nn.ModuleList(
            MimiVectorQuantization(config) for _ in range(self.num_quantizers)
        )
        self.input_proj: nn.Conv1d | None = None
        self.output_proj: nn.Conv1d | None = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = nn.Conv1d(
                config.hidden_size,
                config.vector_quantization_hidden_dimension,
                1,
                bias=False,
            )
            self.output_proj = nn.Conv1d(
                config.vector_quantization_hidden_dimension,
                config.hidden_size,
                1,
                bias=False,
            )

    def encode(self, embeddings: Tensor, num_quantizers: int | None = None) -> Tensor:
        """Quantize embeddings through an ordered residual stack.

        Args:
            embeddings: Tensor shaped ``[batch, hidden_size, frames]``.
            num_quantizers: Number of leading branch layers to use, or ``None``
                to use every layer in this branch.

        Returns:
            Int64 codes shaped ``[quantizers, batch, frames]``.
        """
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)
        count = self.num_quantizers if num_quantizers is None else num_quantizers
        residual = embeddings
        indices = []
        for layer in self.layers[:count]:
            layer_indices = layer.encode(residual)
            residual = residual - layer.decode(layer_indices)
            indices.append(layer_indices)
        return torch.stack(indices)

    def decode(self, codes: Tensor) -> Tensor:
        """Decode and add branch codebooks in checkpoint order.

        Args:
            codes: Int64 tensor shaped ``[batch, quantizers, frames]``.

        Returns:
            Float embeddings shaped ``[batch, hidden_size, frames]``.
        """
        quantized = codes.new_zeros((), dtype=torch.float32)
        for index, indices in enumerate(codes.transpose(0, 1)):
            quantized = quantized + self.layers[index].decode(indices)
        return (
            self.output_proj(quantized) if self.output_proj is not None else quantized
        )


class MimiSplitResidualVectorQuantizer(nn.Module):
    """Combine one semantic RVQ branch with the acoustic branch.

    The split mirrors checkpoint module paths. Fixed batch-one Q32 encoding may
    use an Inductor graph, while fixed Q32 decoding may use the optional ordered
    CUDA gather/add backend; both paths fail closed to these eager methods.
    """

    def __init__(self, config: MimiConfig) -> None:
        """Construct both RVQ branches and optional backend state.

        Args:
            config: Configuration defining total, semantic, and acoustic
                codebook counts plus projection dimensions.
        """
        super().__init__()
        self.max_num_quantizers = config.num_quantizers
        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = (
            config.num_quantizers - config.num_semantic_quantizers
        )
        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_semantic_quantizers
        )
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_acoustic_quantizers
        )
        self._compiled_encode = None
        self._compiled_encode_failed = False
        self._compiled_encode_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_COMPILED_RVQ"
        )
        self._cuda_decode_backend = None
        self._cuda_decode_failed = False
        self._cuda_decode_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_CUDA_RVQ_DECODE"
        )

    def _encode_eager(self, embeddings: Tensor, count: int) -> Tensor:
        """Run the checkpoint-ordered split RVQ encoder.

        Args:
            embeddings: Tensor shaped ``[batch, hidden_size, frames]``.
            count: Total number of semantic-plus-acoustic codebooks to use.

        Returns:
            Int64 codes shaped ``[count, batch, frames]``.
        """
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)
        if count > self.num_semantic_quantizers:
            acoustic = self.acoustic_residual_vector_quantizer.encode(
                embeddings, count - self.num_semantic_quantizers
            )
            codes = torch.cat((codes, acoustic), dim=0)
        return codes

    def _can_use_compiled_encode(self, embeddings: Tensor, count: int) -> bool:
        """Check the fixed batch-one Q32 Inductor encode contract.

        Args:
            embeddings: Candidate pre-quantizer embedding tensor.
            count: Requested total codebook count.

        Returns:
            ``True`` only for the verified shape, dtype, layout, inference mode,
            device architecture, branch split, compiler, and failure state.
        """
        return (
            self._compiled_encode_enabled
            and not self._compiled_encode_failed
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and embeddings.device.type == "cuda"
            and embeddings.dtype == torch.float32
            and embeddings.shape == _PROFILED_RVQ_EMBEDDINGS_SHAPE
            and embeddings.is_contiguous()
            and count == 32
            and self.max_num_quantizers == 32
            and self.num_semantic_quantizers == 1
            and torch.cuda.get_device_capability(embeddings.device)
            == _PROFILED_SM_CAPABILITY
            and hasattr(torch, "compile")
        )

    def _materialize_codebooks(self) -> None:
        """Populate every derived codebook embedding cache.

        Returns:
            ``None`` after semantic and acoustic vectors are materialized.
        """
        for quantizer in (
            self.semantic_residual_vector_quantizer,
            self.acoustic_residual_vector_quantizer,
        ):
            for layer in quantizer.layers:
                _ = layer.codebook.embed

    def _clear_runtime_caches(self) -> None:
        """Drop compiled/native backends and reset fail-closed state.

        Returns:
            ``None`` after all optional quantizer runtime state is cleared.
        """
        self._compiled_encode = None
        self._compiled_encode_failed = False
        self._cuda_decode_backend = None
        self._cuda_decode_failed = False

    def _apply(self, fn, recurse: bool = True):
        """Clear runtime state before applying a tensor transformation.

        Args:
            fn: Function applied by ``nn.Module._apply`` to parameters and
                buffers.
            recurse: Whether the transformation descends into child modules.

        Returns:
            This module after PyTorch applies ``fn``.
        """
        self._clear_runtime_caches()
        return super()._apply(fn, recurse)

    def encode(self, embeddings: Tensor, num_quantizers: int | None = None) -> Tensor:
        """Encode embeddings through the split RVQ.

        Args:
            embeddings: Tensor shaped ``[batch, hidden_size, frames]``.
            num_quantizers: Total codebooks to use, or ``None`` for the
                checkpoint maximum.

        Returns:
            Int64 codes shaped ``[quantizers, batch, frames]``.

        Raises:
            ValueError: If the requested count is outside the semantic minimum
                and checkpoint maximum.
        """
        count = self.max_num_quantizers if num_quantizers is None else num_quantizers
        if count > self.max_num_quantizers:
            raise ValueError(
                f"num_quantizers must be <= {self.max_num_quantizers}, got {count}"
            )
        if count < self.num_semantic_quantizers:
            raise ValueError(
                f"num_quantizers must be >= {self.num_semantic_quantizers}, got {count}"
            )
        if self._can_use_compiled_encode(embeddings, count):
            try:
                if self._compiled_encode is None:
                    self._materialize_codebooks()
                    self._compiled_encode = torch.compile(
                        self._encode_eager,
                        backend="inductor",
                        fullgraph=True,
                        dynamic=False,
                        mode="default",
                    )
                return self._compiled_encode(embeddings, count)
            except Exception:  # noqa: BLE001 - optional compilation must fail closed.
                self._compiled_encode = None
                self._compiled_encode_failed = True
        return self._encode_eager(embeddings, count)

    def _decode_eager(self, codes: Tensor) -> Tensor:
        """Decode semantic and acoustic branches with ordered PyTorch ops.

        Args:
            codes: Int64 tensor shaped ``[batch, codebooks, frames]``.

        Returns:
            Float embeddings shaped ``[batch, hidden_size, frames]``.
        """
        quantized = self.semantic_residual_vector_quantizer.decode(
            codes[:, : self.num_semantic_quantizers]
        )
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized = quantized + self.acoustic_residual_vector_quantizer.decode(
                codes[:, self.num_semantic_quantizers :]
            )
        return quantized

    def _can_use_cuda_decode(self, codes: Tensor) -> bool:
        """Check the exact fixed Q32 CUDA gather/add contract.

        Args:
            codes: Candidate discrete-code tensor.

        Returns:
            ``True`` only for the verified shape, dtype, layout, inference mode,
            device architecture, branch split, and prior failure state.
        """
        return (
            self._cuda_decode_enabled
            and not self._cuda_decode_failed
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and codes.device.type == "cuda"
            and codes.dtype == torch.int64
            and codes.shape == _PROFILED_RVQ_CODES_SHAPE
            and codes.is_contiguous()
            and self.max_num_quantizers == 32
            and self.num_semantic_quantizers == 1
            and torch.cuda.get_device_capability(codes.device)
            == _PROFILED_SM_CAPABILITY
        )

    def _build_cuda_decode_backend(self):
        """Build the optional native decode wrapper.

        Returns:
            ``CudaRVQDecodeBackend`` bound to all ordered codebook tensors.

        Raises:
            Exception: Propagates compiler, library, or codebook validation
                errors so the caller can disable native dispatch and fall back.
        """
        from ._cuda_rvq import CudaRVQDecodeBackend

        self._materialize_codebooks()
        layers = [*self.semantic_residual_vector_quantizer.layers]
        layers.extend(self.acoustic_residual_vector_quantizer.layers)
        return CudaRVQDecodeBackend([layer.codebook.embed for layer in layers])

    def decode(self, codes: Tensor) -> Tensor:
        """Decode split RVQ codes through native or eager execution.

        Args:
            codes: Int64 tensor shaped ``[batch, codebooks, frames]``.

        Returns:
            Float embeddings shaped ``[batch, hidden_size, frames]``. Native
            errors are contained and transparently rerouted to eager PyTorch.
        """
        if self._can_use_cuda_decode(codes):
            try:
                if self._cuda_decode_backend is None:
                    if torch.cuda.is_current_stream_capturing():
                        return self._decode_eager(codes)
                    self._cuda_decode_backend = self._build_cuda_decode_backend()
                semantic, acoustic = self._cuda_decode_backend(codes)
                semantic_rvq = self.semantic_residual_vector_quantizer
                acoustic_rvq = self.acoustic_residual_vector_quantizer
                quantized = semantic_rvq.output_proj(semantic)
                return quantized + acoustic_rvq.output_proj(acoustic)
            except Exception:  # noqa: BLE001 - optional CUDA must fail closed.
                self._cuda_decode_backend = None
                self._cuda_decode_failed = True
        return self._decode_eager(codes)


# Public codec model.


class MimiModel(nn.Module):
    """Expose the checkpoint-compatible Mimi codec runtime.

    The model combines causal convolutional analysis/synthesis stacks, encoder
    and decoder Transformers, the split residual vector quantizer, streaming
    caches, strict checkpoint I/O, and guarded optional accelerators.
    """

    def __init__(
        self, config: MimiConfig | None = None, *, attention_backend: str = "sdpa"
    ) -> None:
        """Construct a Mimi model with uninitialized checkpoint parameters.

        Args:
            config: Architecture configuration, or ``None`` to use published
                ``kyutai/mimi`` defaults.
            attention_backend: ``"sdpa"`` for PyTorch scaled-dot-product
                attention or ``"eager"`` for explicit attention arithmetic.

        Raises:
            ValueError: If the attention backend is unsupported or codebook
                size is not a power of two.
        """
        super().__init__()
        self.config = MimiConfig() if config is None else config
        self.attention_backend = attention_backend
        self.encoder = MimiEncoder(self.config)
        self.encoder_transformer = MimiTransformerModel(self.config, attention_backend)
        self.downsample: MimiConv1d | None = None
        self.upsample: MimiConvTranspose1d | None = None
        if self.config.frame_rate != self.config.encodec_frame_rate:
            rate_ratio = int(self.config.encodec_frame_rate / self.config.frame_rate)
            self.downsample = MimiConv1d(
                self.config,
                self.config.hidden_size,
                self.config.hidden_size,
                2 * rate_ratio,
                stride=2,
                bias=False,
                pad_mode="replicate",
                layer_idx=len(self.encoder._mimiconv1d_layer_names),
            )
            self.upsample = MimiConvTranspose1d(
                self.config,
                self.config.hidden_size,
                self.config.hidden_size,
                2 * rate_ratio,
                stride=2,
                bias=False,
                groups=self.config.upsample_groups,
            )
        self.decoder_transformer = MimiTransformerModel(self.config, attention_backend)
        self.decoder = MimiDecoder(self.config)
        self.quantizer = MimiSplitResidualVectorQuantizer(self.config)
        self._cuda_graph_runner: _MimiCudaGraphRunner | None = None
        self._cuda_graph_failed = False
        self._cuda_graph_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_CUDA_GRAPH"
        )
        self._optimized_long_runtime: Any | None = None
        self._optimized_long_runtime_failed = False
        self._optimized_long_runtime_error: str | None = None
        self._optimized_long_runtime_enabled = _optional_path_enabled(
            "FAST_MIMI_DISABLE_OPTIMIZED_LONG"
        )
        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("codebook_size must be a power of two")
        self.apply(self._initialize_weights)
        self.register_load_state_dict_post_hook(self._reset_runtime_caches_after_load)

    def _clear_runtime_caches(self) -> None:
        """Clear every device- or weight-dependent acceleration cache.

        Returns:
            ``None`` after CUDA Graphs, compiled functions, native handles,
            codebook backend state, and attention failure flags are reset.
        """
        self._cuda_graph_runner = None
        self._cuda_graph_failed = False
        self._optimized_long_runtime = None
        self._optimized_long_runtime_failed = False
        self._optimized_long_runtime_error = None
        self.encoder._clear_runtime_caches()
        self.decoder._clear_runtime_caches()
        self.quantizer._clear_runtime_caches()
        for transformer in (self.encoder_transformer, self.decoder_transformer):
            for layer in transformer.layers:
                layer.self_attn._clear_runtime_caches()

    def _reset_runtime_caches_after_load(self, module, incompatible_keys) -> None:
        """Reset derived state after PyTorch loads checkpoint tensors.

        Args:
            module: Model instance supplied by the load-state post-hook API.
            incompatible_keys: PyTorch report of missing and unexpected keys;
                strict callers validate it before this hook completes.

        Returns:
            ``None`` after runtime caches and derived codebook vectors are
            invalidated.
        """
        del module, incompatible_keys
        self._clear_runtime_caches()
        for item in self.quantizer.modules():
            if isinstance(item, MimiEuclideanCodebook):
                item._embed = None

    def _apply(self, fn, recurse: bool = True):
        """Clear runtime state before applying a tensor transformation.

        Args:
            fn: Function applied by ``nn.Module._apply`` to every parameter and
                buffer, commonly a device or dtype conversion.
            recurse: Whether the transformation descends into child modules.

        Returns:
            This model after PyTorch applies ``fn``.
        """
        self._clear_runtime_caches()
        return super()._apply(fn, recurse)

    def _initialize_weights(self, module: nn.Module) -> None:
        """Initialize one module before pretrained state is loaded.

        Args:
            module: Linear, normalization, convolution, layer-scale, or other
                child visited by ``nn.Module.apply``.

        Returns:
            ``None``. Recognized parameter types are initialized in place;
            unrecognized modules require no direct initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                bound = math.sqrt(
                    module.groups / (module.in_channels * module.kernel_size[0])
                )
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, MimiLayerScale):
            nn.init.constant_(module.scale, self.config.layer_scale_initial_scale)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = KYUTAI_MIMI_MODEL_ID,
        *,
        revision: str = KYUTAI_MIMI_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        attention_backend: str = "sdpa",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MimiModel:
        """Load an exact Mimi config and safetensors checkpoint.

        ``model_id`` may be a local directory containing ``config.json`` and
        ``model.safetensors`` or a Hugging Face Hub repository. State loading is
        strict so missing, extra, or shape-incompatible checkpoint entries fail.

        Args:
            model_id: Local checkpoint directory or Hugging Face repository ID.
            revision: Immutable Hub branch, tag, or commit used for both config
                and weights when ``model_id`` is remote.
            cache_dir: Optional Hugging Face download-cache directory.
            local_files_only: If true, fail instead of accessing the network
                for an uncached remote checkpoint.
            attention_backend: ``"sdpa"`` or ``"eager"`` attention execution.
            device: Optional destination device applied after strict loading.
            dtype: Optional destination dtype. Exact published parity requires
                the default float32 dtype.

        Returns:
            Evaluation-mode ``MimiModel`` containing the exact checkpoint state.

        Raises:
            OSError: If local or Hub checkpoint files cannot be retrieved.
            ValueError: If configuration, backend, or checkpoint structure is
                invalid.
            RuntimeError: If strict state loading finds missing, unexpected, or
                incompatible tensors.
        """
        declared_checkpoint = str(model_id) == KYUTAI_MIMI_MODEL_ID
        if declared_checkpoint and revision != KYUTAI_MIMI_REVISION:
            raise ValueError(
                "declared kyutai/mimi revision changed: "
                f"{revision!r} != {KYUTAI_MIMI_REVISION!r}"
            )
        local_path = Path(model_id)
        if local_path.exists():
            config_path = local_path / "config.json"
            weights_path = local_path / "model.safetensors"
        else:
            config_path = Path(
                hf_hub_download(
                    str(model_id),
                    "config.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
            weights_path = Path(
                hf_hub_download(
                    str(model_id),
                    "model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        config = MimiConfig.from_json_file(config_path)
        if declared_checkpoint:
            _assert_declared_mimi_config(config)
            actual_sha256 = _file_sha256(weights_path)
            if actual_sha256 != KYUTAI_MIMI_WEIGHTS_SHA256:
                raise RuntimeError(
                    "declared kyutai/mimi checkpoint digest changed: "
                    f"{actual_sha256} != {KYUTAI_MIMI_WEIGHTS_SHA256}"
                )
        model = cls(config, attention_backend=attention_backend)
        model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        if declared_checkpoint:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count != KYUTAI_MIMI_PARAMETER_COUNT:
                raise RuntimeError(
                    "declared kyutai/mimi parameter count changed: "
                    f"{parameter_count} != {KYUTAI_MIMI_PARAMETER_COUNT}"
                )
        if device is not None or dtype is not None:
            model.to(device=device, dtype=dtype)
        return model.eval()

    def save_pretrained(self, directory: str | Path) -> None:
        """Save configuration and checkpoint-compatible safetensors state.

        Args:
            directory: Destination directory, created with parents if needed.

        Returns:
            ``None`` after ``config.json`` and ``model.safetensors`` are written.

        Raises:
            OSError: If the destination cannot be created or written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
        }
        save_file(state, directory / "model.safetensors", metadata={"format": "pt"})

    def _encode_frame(
        self,
        input_values: Tensor,
        num_quantizers: int,
        past_key_values: MimiKVCache | None,
        padding_cache: MimiConv1dPaddingCache | None,
        use_streaming: bool,
        return_dict: bool,
    ) -> tuple[Tensor, MimiKVCache | None, MimiConv1dPaddingCache | None]:
        """Encode one offline signal or streaming chunk.

        Args:
            input_values: Float waveform shaped ``[batch, channels, samples]``.
            num_quantizers: Number of existing checkpoint codebooks to emit.
            past_key_values: Optional encoder Transformer continuation cache.
            padding_cache: Optional causal convolution continuation cache.
            use_streaming: Whether the Transformer creates and returns cache
                state for later chunks.
            return_dict: Whether the internal Transformer result uses named
                output fields instead of tuple positions.

        Returns:
            Codes shaped ``[batch, codebooks, frames]`` together with updated
            Transformer and convolution caches.
        """
        embeddings = self.encoder(input_values, padding_cache)
        transformer_output = self.encoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            use_cache=use_streaming,
            return_dict=return_dict,
        )
        past_key_values = (
            transformer_output.past_key_values
            if return_dict
            else (transformer_output[1] if len(transformer_output) > 1 else None)
        )
        embeddings = transformer_output[0].transpose(1, 2)
        if self.downsample is not None:
            embeddings = self.downsample(embeddings, padding_cache)
        codes = self.quantizer.encode(embeddings, num_quantizers).transpose(0, 1)
        return codes, past_key_values, padding_cache

    def get_encoded_length(self, input_length: Tensor) -> Tensor:
        """Map waveform sample counts to encoded Mimi frame counts.

        Args:
            input_length: Scalar or batched tensor of valid waveform lengths.

        Returns:
            Int64 tensor of final code-frame counts after every encoder stride
            and the optional bottleneck downsample.
        """
        output_length = input_length
        for name in self.encoder._mimiconv1d_layer_names:
            output_length = self.encoder.get_submodule(name)._get_output_length(
                output_length
            )
        if self.downsample is not None:
            output_length = self.downsample._get_output_length(output_length)
        return output_length

    def get_audio_codes_mask(
        self, padding_mask: Tensor, padding_side: str = "right"
    ) -> Tensor:
        """Project a sample-validity mask onto the code-frame axis.

        Args:
            padding_mask: Boolean or integer sample mask shaped
                ``[batch, samples]`` or ``[batch, channels, samples]``.
            padding_side: ``"right"`` keeps valid frames at the start; any
                other value mirrors the mask for left-padded input behavior.

        Returns:
            Boolean mask shaped ``[batch, maximum_encoded_frames]``.
        """
        lengths = self.get_encoded_length(padding_mask.sum(dim=-1))
        mask = torch.arange(lengths.max(), device=lengths.device).expand(
            len(lengths), -1
        )
        mask = mask < lengths.unsqueeze(1)
        return mask if padding_side == "right" else mask.flip(-1)

    def _new_padding_cache(self) -> MimiConv1dPaddingCache:
        """Build convolution-cache metadata from the instantiated encoder.

        Returns:
            Empty ``MimiConv1dPaddingCache`` covering every streaming encoder
            convolution and the optional bottleneck downsample.
        """
        paddings: list[int] = []
        modes: list[str] = []
        channels: list[int] = []
        for name in self.encoder._mimiconv1d_layer_names:
            layer = self.encoder.get_submodule(name)
            paddings.append(layer.padding_left + layer.padding_right)
            modes.append(layer.pad_mode)
            channels.append(layer.in_channels)
        if self.downsample is not None:
            paddings.append(
                self.downsample.padding_left + self.downsample.padding_right
            )
            modes.append(self.downsample.pad_mode)
            channels.append(self.downsample.in_channels)
        return MimiConv1dPaddingCache(len(paddings), paddings, modes, channels)

    def encode(
        self,
        input_values: Tensor,
        padding_mask: Tensor | None = None,
        num_quantizers: int | None = None,
        encoder_past_key_values: MimiKVCache | None = None,
        padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
        return_dict: bool | None = None,
    ) -> MimiEncoderOutput | tuple[Any, ...]:
        """Encode ``[batch, channels, samples]`` float waveforms into tokens.

        Set ``use_streaming=True`` and pass the returned Transformer and
        convolution caches into the next call to continue a chunked stream.
        ``num_quantizers`` selects how many existing checkpoint codebooks are
        used; it does not change their values or precision.

        Args:
            input_values: Float waveform shaped
                ``[batch, channels, samples]``.
            padding_mask: Optional sample-validity mask. ``None`` treats every
                input sample as valid.
            num_quantizers: Number of leading semantic/acoustic codebooks to
                use, or ``None`` for the checkpoint maximum.
            encoder_past_key_values: Transformer cache returned by the previous
                streaming chunk.
            padding_cache: Causal convolution cache returned by the previous
                streaming chunk.
            use_streaming: Whether streaming caches are created and updated;
                ``None`` uses the configuration default.
            return_dict: Whether to return ``MimiEncoderOutput``; ``None`` uses
                the configuration default and ``False`` returns a tuple.

        Returns:
            Named or tuple output containing codes and optional updated caches.

        Raises:
            ValueError: If codebook count, tensor rank, or audio channel count
                is outside the supported model contract.
        """
        return_dict = self.config.return_dict if return_dict is None else return_dict
        use_streaming = (
            self.config.use_streaming if use_streaming is None else use_streaming
        )
        count = self.config.num_quantizers if num_quantizers is None else num_quantizers
        if count > self.config.num_quantizers:
            raise ValueError(
                f"num_quantizers must be <= {self.config.num_quantizers}, got {count}"
            )
        if input_values.ndim != 3:
            raise ValueError("input_values must have shape [batch, channels, samples]")
        if not 1 <= input_values.shape[1] <= 2:
            raise ValueError("input audio must have one or two channels")
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values, dtype=torch.bool)
        if use_streaming and padding_cache is None:
            padding_cache = self._new_padding_cache()
        codes, encoder_past_key_values, padding_cache = self._encode_frame(
            input_values,
            count,
            encoder_past_key_values,
            padding_cache,
            use_streaming,
            return_dict,
        )
        if not return_dict:
            return codes, encoder_past_key_values, padding_cache
        return MimiEncoderOutput(codes, encoder_past_key_values, padding_cache)

    def _decode_frame(
        self,
        codes: Tensor,
        past_key_values: MimiKVCache | None,
        return_dict: bool,
    ) -> tuple[Tensor, MimiKVCache | None]:
        """Decode one token tensor and update decoder context state.

        Args:
            codes: Int64 codes shaped ``[batch, codebooks, frames]``.
            past_key_values: Optional decoder Transformer continuation cache.
            return_dict: Whether the internal Transformer result uses named
                output fields instead of tuple positions.

        Returns:
            Reconstructed waveform and updated decoder Transformer cache.
        """
        embeddings = self.quantizer.decode(codes)
        if self.upsample is not None:
            embeddings = self.upsample(embeddings)
        transformer_output = self.decoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        past_key_values = (
            transformer_output.past_key_values
            if return_dict
            else (transformer_output[1] if len(transformer_output) > 1 else None)
        )
        audio = self.decoder(transformer_output[0].transpose(1, 2))
        return audio, past_key_values

    def decode(
        self,
        audio_codes: Tensor,
        padding_mask: Tensor | None = None,
        decoder_past_key_values: MimiKVCache | None = None,
        return_dict: bool | None = None,
    ) -> MimiDecoderOutput | tuple[Any, ...]:
        """Decode discrete Mimi tokens into a float waveform.

        Args:
            audio_codes: Int64 tensor shaped ``[batch, codebooks, frames]``.
            padding_mask: Optional sample-level mask whose length crops decoder
                overrun at the original waveform boundary.
            decoder_past_key_values: Optional Transformer cache from an earlier
                decode call.
            return_dict: Whether to return ``MimiDecoderOutput``; ``None`` uses
                the configuration default and ``False`` returns a tuple.

        Returns:
            Named or tuple output containing waveform and optional updated
            decoder cache.
        """
        return_dict = self.config.return_dict if return_dict is None else return_dict
        audio, decoder_past_key_values = self._decode_frame(
            audio_codes, decoder_past_key_values, return_dict
        )
        if padding_mask is not None and padding_mask.shape[-1] < audio.shape[-1]:
            audio = audio[..., : padding_mask.shape[-1]]
        if not return_dict:
            return audio, decoder_past_key_values
        return MimiDecoderOutput(audio, decoder_past_key_values)

    def _forward_eager(
        self,
        input_values: Tensor,
        padding_mask: Tensor | None = None,
        num_quantizers: int | None = None,
        audio_codes: Tensor | None = None,
        encoder_past_key_values: MimiKVCache | None = None,
        decoder_past_key_values: MimiKVCache | None = None,
        return_dict: bool | None = None,
    ) -> MimiOutput | tuple[Any, ...]:
        """Run the portable round-trip composition without CUDA Graph replay.

        Args:
            input_values: Float waveform shaped
                ``[batch, channels, samples]``. It also supplies default mask
                shape when ``audio_codes`` is already provided.
            padding_mask: Optional sample-validity and final-cropping mask.
            num_quantizers: Codebook count used when encoding is required.
            audio_codes: Optional precomputed tokens; when supplied, encoding is
                skipped.
            encoder_past_key_values: Optional encoder Transformer cache.
            decoder_past_key_values: Optional decoder Transformer cache.
            return_dict: Whether named output dataclasses are returned; ``None``
                uses the configuration default.

        Returns:
            ``MimiOutput`` or tuple containing tokens, reconstructed waveform,
            and optional encoder/decoder caches.
        """
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values, dtype=torch.bool)
        if audio_codes is None:
            encoded = self.encode(
                input_values,
                padding_mask,
                num_quantizers,
                encoder_past_key_values,
                return_dict=return_dict,
            )
            audio_codes = encoded[0]
            encoder_past_key_values = (
                encoded.encoder_past_key_values
                if return_dict
                else (encoded[1] if len(encoded) > 1 else None)
            )
        decoded = self.decode(
            audio_codes, padding_mask, decoder_past_key_values, return_dict
        )
        audio_values = decoded[0]
        decoder_past_key_values = (
            decoded.decoder_past_key_values
            if return_dict
            else (decoded[1] if len(decoded) > 1 else None)
        )
        if not return_dict:
            return (
                audio_codes,
                audio_values,
                encoder_past_key_values,
                decoder_past_key_values,
            )
        return MimiOutput(
            audio_codes, audio_values, encoder_past_key_values, decoder_past_key_values
        )

    def _can_use_optimized_long(
        self,
        input_values: Tensor,
        padding_mask: Tensor,
        num_quantizers: int,
        audio_codes: Tensor | None,
        encoder_past_key_values: MimiKVCache | None,
        decoder_past_key_values: MimiKVCache | None,
        return_dict: bool,
    ) -> bool:
        """Check the complete contract measured for the accepted SM120 path."""
        return (
            self._optimized_long_runtime_enabled
            and not self._optimized_long_runtime_failed
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and input_values.device.type == "cuda"
            and input_values.dtype == torch.float32
            and tuple(input_values.shape) == _PROFILED_LONG_AUDIO_SHAPE
            and input_values.is_contiguous()
            and not input_values.requires_grad
            and padding_mask.device == input_values.device
            and padding_mask.dtype == torch.bool
            and tuple(padding_mask.shape) == _PROFILED_LONG_AUDIO_SHAPE
            and padding_mask.is_contiguous()
            and num_quantizers == 8
            and audio_codes is None
            and encoder_past_key_values is None
            and decoder_past_key_values is None
            and return_dict
            and self.attention_backend == "sdpa"
            and torch.__version__.split("+", 1)[0] == "2.13.0"
            and torch.version.cuda == "13.0"
            and torch.cuda.get_device_capability(input_values.device)
            == _PROFILED_SM_CAPABILITY
            and not torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cudnn.allow_tf32
            and not torch.cuda.is_current_stream_capturing()
        )

    def _run_optimized_long(
        self,
        input_values: Tensor,
        padding_mask: Tensor,
        num_quantizers: int,
    ) -> MimiOutput | None:
        """Run the accepted runtime or permanently fail closed to pure PyTorch."""
        try:
            if self._optimized_long_runtime is None:
                from ._optimized_runtime import OptimizedLongMimi

                self._optimized_long_runtime = OptimizedLongMimi(self)
            output = self._optimized_long_runtime.forward(
                input_values,
                padding_mask,
                num_quantizers,
            )
            return MimiOutput(
                output.audio_codes.clone(),
                output.audio_values.clone(),
                None,
                None,
            )
        except Exception as error:  # noqa: BLE001 - optional backend fails closed.
            self._optimized_long_runtime = None
            self._optimized_long_runtime_failed = True
            self._optimized_long_runtime_error = repr(error)
            return None

    def _can_use_cuda_graph(
        self,
        input_values: Tensor,
        padding_mask: Tensor,
        num_quantizers: int,
        audio_codes: Tensor | None,
        encoder_past_key_values: MimiKVCache | None,
        decoder_past_key_values: MimiKVCache | None,
        return_dict: bool,
    ) -> bool:
        """Check the fixed one-second Q32 CUDA Graph replay contract.

        Args:
            input_values: Candidate one-second waveform tensor.
            padding_mask: Candidate boolean sample-validity mask.
            num_quantizers: Resolved codebook count for this call.
            audio_codes: Optional precomputed codes, which disable round-trip
                graph replay when present.
            encoder_past_key_values: Optional encoder cache, which disables the
                stateless graph path.
            decoder_past_key_values: Optional decoder cache, which disables the
                stateless graph path.
            return_dict: Resolved output mode; graph replay returns named output.

        Returns:
            ``True`` only when all mode, shape, layout, dtype, device,
            architecture, stream, codebook, cache, and backend guards pass.
        """
        return (
            self._cuda_graph_enabled
            and not self._cuda_graph_failed
            and not self.training
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and input_values.device.type == "cuda"
            and input_values.dtype == torch.float32
            and input_values.shape == _PROFILED_ONE_SECOND_AUDIO_SHAPE
            and input_values.is_contiguous()
            and not input_values.requires_grad
            and padding_mask.device == input_values.device
            and padding_mask.dtype == torch.bool
            and padding_mask.shape == input_values.shape
            and padding_mask.is_contiguous()
            and num_quantizers == 32
            and audio_codes is None
            and encoder_past_key_values is None
            and decoder_past_key_values is None
            and return_dict
            and self.attention_backend == "sdpa"
            and self.quantizer._compiled_encode_enabled
            and torch.cuda.get_device_capability(input_values.device)
            == _PROFILED_SM_CAPABILITY
            and not torch.cuda.is_current_stream_capturing()
        )

    def _build_cuda_graph(
        self, input_values: Tensor, padding_mask: Tensor
    ) -> _MimiCudaGraphRunner:
        """Warm and capture the fixed one-second Q32 round trip.

        Args:
            input_values: Eligible contiguous float32 CUDA waveform with shape
                ``[1, 1, 24000]``.
            padding_mask: Eligible contiguous boolean CUDA mask with the same
                shape and device.

        Returns:
            Runner owning captured graph, static inputs, outputs, and stream.

        Raises:
            RuntimeError: If the exact compiled RVQ path did not materialize.
            Exception: Propagates any warmup or capture failure so the caller can
                disable graph dispatch and continue eagerly.
        """
        current_stream = torch.cuda.current_stream(input_values.device)
        self._forward_eager(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=32,
            return_dict=True,
        )
        torch.cuda.synchronize(input_values.device)
        if self.quantizer._compiled_encode is None:
            raise RuntimeError("the exact compiled RVQ path is unavailable")

        static_input = input_values.clone()
        static_mask = padding_mask.clone()
        warm_stream = torch.cuda.Stream(device=input_values.device)
        warm_stream.wait_stream(current_stream)
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                self._forward_eager(
                    static_input,
                    padding_mask=static_mask,
                    num_quantizers=32,
                    return_dict=True,
                )
        current_stream.wait_stream(warm_stream)
        torch.cuda.synchronize(input_values.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward_eager(
                static_input,
                padding_mask=static_mask,
                num_quantizers=32,
                return_dict=True,
            )
        return _MimiCudaGraphRunner(
            graph, static_input, static_mask, output, current_stream
        )

    def forward(
        self,
        input_values: Tensor,
        padding_mask: Tensor | None = None,
        num_quantizers: int | None = None,
        audio_codes: Tensor | None = None,
        encoder_past_key_values: MimiKVCache | None = None,
        decoder_past_key_values: MimiKVCache | None = None,
        return_dict: bool | None = None,
    ) -> MimiOutput | tuple[Any, ...]:
        """Run an encode/decode round trip or decode supplied tokens.

        Args:
            input_values: Float waveform shaped
                ``[batch, channels, samples]``. It remains required when
                ``audio_codes`` is supplied because it defines default mask and
                output-length behavior.
            padding_mask: Optional sample-validity and output-cropping mask.
            num_quantizers: Number of checkpoint codebooks used for encoding,
                or ``None`` for the configuration maximum.
            audio_codes: Optional precomputed tokens that skip encoding.
            encoder_past_key_values: Optional encoder Transformer cache.
            decoder_past_key_values: Optional decoder Transformer cache.
            return_dict: Whether to return ``MimiOutput``; ``None`` uses the
                configuration default and ``False`` returns a tuple.

        Returns:
            Named or tuple result containing codes, reconstructed waveform, and
            optional continuation caches. Eligible one-second Q32 calls may use
            CUDA Graph replay with cloned caller-owned outputs.
        """
        resolved_return_dict = (
            self.config.return_dict if return_dict is None else return_dict
        )
        resolved_quantizers = (
            self.config.num_quantizers if num_quantizers is None else num_quantizers
        )
        effective_mask = (
            torch.ones_like(input_values, dtype=torch.bool)
            if padding_mask is None
            else padding_mask
        )
        if self._can_use_optimized_long(
            input_values,
            effective_mask,
            resolved_quantizers,
            audio_codes,
            encoder_past_key_values,
            decoder_past_key_values,
            resolved_return_dict,
        ):
            optimized_output = self._run_optimized_long(
                input_values,
                effective_mask,
                resolved_quantizers,
            )
            if optimized_output is not None:
                return optimized_output
        if self._can_use_cuda_graph(
            input_values,
            effective_mask,
            resolved_quantizers,
            audio_codes,
            encoder_past_key_values,
            decoder_past_key_values,
            resolved_return_dict,
        ):
            runner = self._cuda_graph_runner
            if runner is not None and runner.matches_current_stream():
                return runner.replay(input_values, effective_mask)
            if runner is None:
                try:
                    runner = self._build_cuda_graph(input_values, effective_mask)
                    self._cuda_graph_runner = runner
                    return runner.replay(input_values, effective_mask)
                except Exception:  # noqa: BLE001 - optional graph must fail closed.
                    self._cuda_graph_runner = None
                    self._cuda_graph_failed = True
        return self._forward_eager(
            input_values,
            effective_mask,
            resolved_quantizers,
            audio_codes,
            encoder_past_key_values,
            decoder_past_key_values,
            resolved_return_dict,
        )
