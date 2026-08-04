from __future__ import annotations

import torch

from fast_mimi import MimiConfig, MimiModel


def tiny_config() -> MimiConfig:
    return MimiConfig(
        sampling_rate=64,
        hidden_size=16,
        num_filters=2,
        upsampling_ratios=(2, 2),
        codebook_size=16,
        codebook_dim=8,
        num_quantizers=4,
        vector_quantization_hidden_dimension=8,
        num_semantic_quantizers=1,
        upsample_groups=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=8,
        frame_rate_override=8,
    )


def paired_models(attention_backend: str = "eager"):
    from transformers import MimiConfig as TransformersMimiConfig
    from transformers import MimiModel as TransformersMimiModel

    config = tiny_config()
    reference_config = TransformersMimiConfig(**config.to_dict())
    reference_config._attn_implementation = attention_backend
    reference = TransformersMimiModel(reference_config).eval()
    independent = MimiModel(config, attention_backend=attention_backend).eval()

    generator = torch.Generator().manual_seed(1234)
    state = reference.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            if name.endswith(".initialized"):
                value.fill_(1)
            elif name.endswith(".cluster_usage"):
                value.copy_(torch.rand(value.shape, generator=generator) + 0.5)
            elif name.endswith("layernorm.weight"):
                value.copy_(torch.rand(value.shape, generator=generator) * 0.2 + 0.9)
            else:
                value.copy_(torch.randn(value.shape, generator=generator) * 0.05)
    reference.load_state_dict(state, strict=True)
    independent.load_state_dict(state, strict=True)
    return independent, reference
