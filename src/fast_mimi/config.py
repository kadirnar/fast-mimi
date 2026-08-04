"""Checkpoint-compatible Mimi configuration without Transformers.

The defaults describe the published ``kyutai/mimi`` inference architecture.
Serialization accepts the additional metadata found in Transformers configs
but emits only fields owned by this independent runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

KYUTAI_MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"


@dataclass
class MimiConfig:
    """Describe the published Mimi architecture and runtime defaults.

    Args:
        sampling_rate: Expected waveform sampling rate in hertz.
        audio_channels: Number of waveform channels consumed by the model.
        hidden_size: Channel width of codec embeddings and Transformer states.
        num_filters: Base channel width of the SEANet convolution stacks.
        num_residual_layers: Residual blocks inserted at each SEANet scale.
        upsampling_ratios: Decoder strides; the encoder uses them in reverse.
        kernel_size: Kernel width of the first encoder and decoder convolutions.
        last_kernel_size: Kernel width of the final encoder and decoder layers.
        residual_kernel_size: Non-pointwise kernel width inside residual blocks.
        dilation_growth_rate: Exponential dilation base for residual layers.
        use_causal_conv: Whether convolutions use left-context causal padding.
        pad_mode: PyTorch padding mode used by ordinary convolution execution.
        compress: Divisor that selects the residual block's hidden width.
        trim_right_ratio: Fraction of transposed-convolution trimming applied on
            the right side for causal decoding.
        codebook_size: Number of discrete entries in every RVQ codebook.
        codebook_dim: Feature dimension stored by each codebook entry.
        num_quantizers: Maximum number of checkpoint RVQ codebooks.
        use_conv_shortcut: Whether residual blocks use learned 1x1 shortcuts.
        vector_quantization_hidden_dimension: Channel width presented to RVQ.
        num_semantic_quantizers: Leading codebooks assigned to semantics.
        upsample_groups: Group count of the bottleneck transposed convolution.
        num_hidden_layers: Layer count in each Transformer stack.
        intermediate_size: Hidden width of Transformer feed-forward networks.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads before grouped expansion.
        head_dim: Per-head feature width, or ``None`` to derive it from the
            hidden size and query-head count.
        hidden_act: Transformer feed-forward activation name; only ``"gelu"``
            is checkpoint-compatible in this runtime.
        max_position_embeddings: Reference configuration's position capacity.
        initializer_range: Standard deviation used for new linear parameters.
        norm_eps: Epsilon used by Transformer layer normalization.
        use_cache: Default Transformer key/value cache behavior.
        use_streaming: Default encoder convolution-cache behavior.
        rope_theta: Base period used to construct rotary embeddings.
        sliding_window: Maximum visible local-attention frame count.
        attention_dropout: Attention probability dropout used during training.
        layer_scale_initial_scale: Initial residual layer-scale coefficient.
        attention_bias: Whether Q/K/V/output projections contain bias terms.
        frame_rate_override: Final codec frame rate, or ``None`` to derive it
            from the sampling rate and convolution strides.
        return_dict: Whether model methods return named output dataclasses by
            default instead of tuples.
    """

    # Waveform and causal SEANet configuration.
    sampling_rate: int = 24_000
    audio_channels: int = 1
    hidden_size: int = 512
    num_filters: int = 64
    num_residual_layers: int = 1
    upsampling_ratios: tuple[int, ...] = (8, 6, 5, 4)
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "constant"
    compress: int = 2
    trim_right_ratio: float = 1.0

    # Split residual vector quantizer configuration.
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_quantizers: int = 32
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256
    num_semantic_quantizers: int = 1
    upsample_groups: int = 512

    # Encoder and decoder Transformer configuration.
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int | None = 64
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8000
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_cache: bool = False
    use_streaming: bool = False
    rope_theta: float = 10_000.0
    sliding_window: int = 250
    attention_dropout: float = 0.0
    layer_scale_initial_scale: float = 0.01
    attention_bias: bool = False

    # Public output and final frame-rate defaults.
    frame_rate_override: float | None = 12.5
    return_dict: bool = True

    def __post_init__(self) -> None:
        """Normalize collection types and validate dependent dimensions.

        Returns:
            ``None`` after derived values and invariant checks are complete.

        Raises:
            ValueError: If attention heads, quantizer splits, or codebook
                dimensions are incompatible.
        """
        self.upsampling_ratios = tuple(self.upsampling_ratios)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.num_semantic_quantizers >= self.num_quantizers:
            raise ValueError(
                "num_semantic_quantizers must be smaller than num_quantizers"
            )
        if self.codebook_dim != self.vector_quantization_hidden_dimension:
            raise ValueError(
                "codebook_dim must match vector_quantization_hidden_dimension"
            )

    @property
    def encodec_frame_rate(self) -> int:
        """Compute the SEANet frame rate before the stride-two bottleneck.

        Returns:
            Integer analysis frames per second after the encoder strides.
        """
        return math.ceil(self.sampling_rate / math.prod(self.upsampling_ratios))

    @property
    def frame_size(self) -> int:
        """Compute waveform samples represented by one final code frame.

        Returns:
            Product of all encoder strides including the bottleneck stride.
        """
        return math.prod(self.upsampling_ratios) * 2

    @property
    def frame_rate(self) -> float:
        """Resolve the final codec frame rate.

        Returns:
            Explicit frame-rate override when present, otherwise sampling rate
            divided by the derived frame size.
        """
        if self.frame_rate_override is not None:
            return self.frame_rate_override
        return self.sampling_rate / self.frame_size

    @property
    def num_codebooks(self) -> int:
        """Expose the Transformers-compatible quantizer-count alias.

        Returns:
            Value of ``num_quantizers``.
        """
        return self.num_quantizers

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MimiConfig:
        """Build a config while ignoring unrelated Transformers metadata.

        Args:
            values: Mapping loaded from a checkpoint configuration. The public
                ``frame_rate`` key is mapped to ``frame_rate_override``.

        Returns:
            Validated ``MimiConfig`` containing all recognized values.

        Raises:
            ValueError: If recognized values violate configuration invariants.
        """
        known = {field.name for field in fields(cls)}
        mapped = dict(values)
        if "frame_rate" in mapped and "frame_rate_override" not in mapped:
            mapped["frame_rate_override"] = mapped.pop("frame_rate")
        return cls(**{key: value for key, value in mapped.items() if key in known})

    @classmethod
    def from_json_file(cls, path: str | Path) -> MimiConfig:
        """Load configuration values from a local JSON file.

        Args:
            path: Path to a UTF-8 JSON configuration file.

        Returns:
            Parsed and validated ``MimiConfig`` instance.

        Raises:
            OSError: If the file cannot be opened.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If parsed values violate configuration invariants.
        """
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "kyutai/mimi",
        *,
        revision: str = KYUTAI_MIMI_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> MimiConfig:
        """Load ``config.json`` from a Hugging Face Hub revision.

        Args:
            model_id: Hub repository containing the Mimi configuration.
            revision: Immutable branch, tag, or commit used for the download.
            cache_dir: Optional Hugging Face cache directory override.
            local_files_only: If true, fail instead of accessing the network
                when the requested revision is absent from the local cache.

        Returns:
            Parsed and validated ``MimiConfig`` instance.

        Raises:
            OSError: If the pinned file cannot be retrieved or read.
            ValueError: If parsed values violate configuration invariants.
        """
        path = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls.from_json_file(path)

    def to_dict(self) -> dict[str, Any]:
        """Create the checkpoint-compatible serialization mapping.

        Returns:
            Plain dictionary with a list-valued ``upsampling_ratios`` field and
            public ``frame_rate`` key.
        """
        result = asdict(self)
        result["upsampling_ratios"] = list(self.upsampling_ratios)
        result["frame_rate"] = result.pop("frame_rate_override")
        return result

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write a checkpoint-compatible ``config.json`` file.

        Args:
            directory: Destination directory, created with parents if needed.

        Returns:
            Path to the written JSON file.

        Raises:
            OSError: If the destination cannot be created or written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path
