from __future__ import annotations

import os

import pytest
import torch
from conftest import paired_models, tiny_config

from fast_mimi import KYUTAI_MIMI_REVISION, MimiKVCache, MimiModel


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_module_tree_and_forward_are_exact(backend: str) -> None:
    independent, reference = paired_models(backend)
    independent_state = independent.state_dict()
    reference_state = reference.state_dict()
    assert list(independent_state) == list(reference_state)
    assert all(
        independent_state[key].shape == reference_state[key].shape
        for key in independent_state
    )

    generator = torch.Generator().manual_seed(7)
    audio = torch.randn((2, 1, 65), generator=generator) * 0.05
    with torch.inference_mode():
        independent_output = independent(audio)
        reference_output = reference(audio)
    assert torch.equal(independent_output.audio_codes, reference_output.audio_codes)
    assert torch.equal(independent_output.audio_values, reference_output.audio_values)


def test_sliding_window_path_is_exact() -> None:
    independent, reference = paired_models("sdpa")
    audio = torch.linspace(-0.1, 0.1, 72).view(1, 1, -1)
    with torch.inference_mode():
        independent_codes = independent.encode(audio).audio_codes
        reference_codes = reference.encode(audio).audio_codes
    assert independent_codes.shape[-1] > independent.config.sliding_window
    assert torch.equal(independent_codes, reference_codes)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_convolution_padding_boundaries_are_exact(backend: str) -> None:
    independent, reference = paired_models(backend)
    for length in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65):
        audio = torch.linspace(-0.1, 0.1, length).view(1, 1, -1)
        with torch.inference_mode():
            independent_output = independent(audio)
            reference_output = reference(audio)
        assert torch.equal(independent_output.audio_codes, reference_output.audio_codes)
        assert torch.equal(
            independent_output.audio_values, reference_output.audio_values
        )


def test_streaming_encoder_tokens_and_cache_lengths_are_exact() -> None:
    independent, reference = paired_models("sdpa")
    audio = torch.linspace(-0.1, 0.1, 40).view(1, 1, -1)
    independent_kv = independent_padding = None
    reference_kv = reference_padding = None

    for chunk in audio.split(8, dim=-1):
        with torch.inference_mode():
            independent_output = independent.encode(
                chunk,
                encoder_past_key_values=independent_kv,
                padding_cache=independent_padding,
                use_streaming=True,
            )
            reference_output = reference.encode(
                chunk,
                encoder_past_key_values=reference_kv,
                padding_cache=reference_padding,
                use_streaming=True,
            )
        assert torch.equal(independent_output.audio_codes, reference_output.audio_codes)
        independent_kv = independent_output.encoder_past_key_values
        independent_padding = independent_output.padding_cache
        reference_kv = reference_output.encoder_past_key_values
        reference_padding = reference_output.padding_cache
        assert independent_kv.get_seq_length() == reference_kv.get_seq_length()


def test_cached_decoder_frames_are_exact() -> None:
    from transformers import DynamicCache

    independent, reference = paired_models("sdpa")
    independent_cache = MimiKVCache(independent.config.num_hidden_layers)
    reference_cache = DynamicCache(config=reference.config)
    generator = torch.Generator().manual_seed(99)

    for _ in range(10):
        codes = torch.randint(
            0, independent.config.codebook_size, (1, 4, 1), generator=generator
        )
        with torch.inference_mode():
            independent_output = independent.decode(
                codes, decoder_past_key_values=independent_cache
            )
            reference_output = reference.decode(
                codes, decoder_past_key_values=reference_cache
            )
        assert torch.equal(
            independent_output.audio_values, reference_output.audio_values
        )
        assert independent_cache.get_seq_length() == reference_cache.get_seq_length()
        assert all(
            key is None or key.shape[-2] <= independent.config.sliding_window - 1
            for key in independent_cache.key_cache
        )


def test_quantizer_decode_zero_and_addition_order_are_exact() -> None:
    independent, reference = paired_models("sdpa")
    codes = torch.tensor([[[0, 1, 2], [3, 2, 1], [1, 0, 3], [2, 3, 0]]])
    with torch.inference_mode():
        independent_output = independent.quantizer.decode(codes)
        reference_output = reference.quantizer.decode(codes)
    assert independent_output.dtype == torch.float32
    assert torch.equal(independent_output, reference_output)


def test_runtime_acceleration_fallback_and_cache_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_MIMI_DISABLE_COMPILED_RVQ", "1")
    monkeypatch.setenv("FAST_MIMI_DISABLE_COMPILED_ENCODER", "1")
    monkeypatch.setenv("FAST_MIMI_DISABLE_COMPILED_DECODER", "true")
    monkeypatch.setenv("FAST_MIMI_DISABLE_CUDNN_BENCHMARK_PRIME", "yes")
    monkeypatch.setenv("FAST_MIMI_DISABLE_LONG_BUILTIN_PADDING", "1")
    monkeypatch.setenv("FAST_MIMI_DISABLE_NATIVE_WINDOW_ATTENTION", "true")
    monkeypatch.setenv("FAST_MIMI_DISABLE_NATIVE_WINDOW_MASK_OMISSION", "1")
    monkeypatch.setenv("FAST_MIMI_DISABLE_CUDA_GRAPH", "true")
    monkeypatch.setenv("FAST_MIMI_DISABLE_CUDA_RVQ_DECODE", "yes")
    model = MimiModel(tiny_config()).eval()
    assert not model.quantizer._compiled_encode_enabled
    assert not model.encoder._compiled_forward_enabled
    assert not model.decoder._compiled_forward_enabled
    assert not model.encoder._cudnn_benchmark_prime_enabled
    assert not model.decoder._cudnn_benchmark_prime_enabled
    assert not model.encoder._long_builtin_padding_enabled
    assert not model.decoder._long_builtin_padding_enabled
    assert all(
        not layer.self_attn._native_window_enabled
        for transformer in (model.encoder_transformer, model.decoder_transformer)
        for layer in transformer.layers
    )
    assert not model.encoder_transformer._omit_native_window_mask_enabled
    assert not model.decoder_transformer._omit_native_window_mask_enabled
    assert not model._cuda_graph_enabled
    assert not model.quantizer._cuda_decode_enabled

    model._cuda_graph_runner = object()
    model._cuda_graph_failed = True
    model.quantizer._compiled_encode = lambda embeddings, count: embeddings
    model.quantizer._compiled_encode_failed = True
    model.encoder._compiled_forward = lambda hidden_states: hidden_states
    model.encoder._compiled_forward_failed = True
    model.encoder._compiled_forward_long = lambda hidden_states: hidden_states
    model.encoder._compiled_forward_long_failed = True
    model.decoder._compiled_forward = lambda hidden_states: hidden_states
    model.decoder._compiled_forward_failed = True
    model.decoder._compiled_forward_long = lambda hidden_states: hidden_states
    model.decoder._compiled_forward_long_failed = True
    model.quantizer._cuda_decode_backend = object()
    model.quantizer._cuda_decode_failed = True
    for transformer in (model.encoder_transformer, model.decoder_transformer):
        for layer in transformer.layers:
            layer.self_attn._native_window_failed = True
    _ = model.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed
    model.load_state_dict(model.state_dict(), strict=True)
    assert model._cuda_graph_runner is None
    assert not model._cuda_graph_failed
    assert model.quantizer._compiled_encode is None
    assert not model.quantizer._compiled_encode_failed
    assert model.encoder._compiled_forward is None
    assert not model.encoder._compiled_forward_failed
    assert model.encoder._compiled_forward_long is None
    assert not model.encoder._compiled_forward_long_failed
    assert model.decoder._compiled_forward is None
    assert not model.decoder._compiled_forward_failed
    assert model.decoder._compiled_forward_long is None
    assert not model.decoder._compiled_forward_long_failed
    assert model.quantizer._cuda_decode_backend is None
    assert not model.quantizer._cuda_decode_failed
    assert all(
        not layer.self_attn._native_window_failed
        for transformer in (model.encoder_transformer, model.decoder_transformer)
        for layer in transformer.layers
    )
    assert (
        model.quantizer.semantic_residual_vector_quantizer.layers[0].codebook._embed
        is None
    )


def test_long_cudnn_benchmark_prime_restores_global_flags() -> None:
    from fast_mimi.model import _run_with_long_cudnn_benchmark

    previous_benchmark = torch.backends.cudnn.benchmark
    previous_limit = torch.backends.cudnn.benchmark_limit
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark_limit = 7
    hidden_states = torch.ones(1)
    observed = None
    try:

        def function(value: torch.Tensor) -> torch.Tensor:
            nonlocal observed
            observed = (
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.benchmark_limit,
            )
            return value + 1

        output = _run_with_long_cudnn_benchmark(function, hidden_states)
        assert torch.equal(output, hidden_states + 1)
        assert observed == (True, 10)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.benchmark_limit == 7
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.benchmark_limit = previous_limit


def test_decoder_builtin_causal_padding_is_exact() -> None:
    model = MimiModel(tiny_config()).eval()
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.zero_()
    generator = torch.Generator().manual_seed(20260804)
    hidden_states = torch.randn((1, model.config.hidden_size, 9), generator=generator)
    with torch.inference_mode():
        reference = model.decoder._forward_eager(hidden_states)
        output = model.decoder._forward_long_with_builtin_padding(hidden_states)
    assert torch.equal(reference, output)
    assert reference.shape == output.shape
    assert reference.stride() == output.stride()


def test_encoder_builtin_causal_padding_is_exact() -> None:
    model = MimiModel(tiny_config()).eval()
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.zero_()
    hidden_states = torch.linspace(-0.1, 0.1, 37).view(1, 1, -1)
    with torch.inference_mode():
        reference = model.encoder._forward_offline(hidden_states)
        output = model.encoder._forward_long_with_builtin_padding(hidden_states)
    assert torch.equal(reference, output)
    assert reference.shape == output.shape
    assert reference.stride() == output.stride()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MIMI_INTEGRATION") != "1", reason="set RUN_MIMI_INTEGRATION=1"
)
def test_published_checkpoint_is_bitwise_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import MimiModel as TransformersMimiModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    independent = MimiModel.from_pretrained(
        revision=KYUTAI_MIMI_REVISION,
        attention_backend="sdpa",
        device=device,
    )
    reference = (
        TransformersMimiModel.from_pretrained(
            "kyutai/mimi",
            revision=KYUTAI_MIMI_REVISION,
            attn_implementation="sdpa",
        )
        .eval()
        .to(device)
    )
    audio = torch.linspace(-0.1, 0.1, 6_001, device=device).view(1, 1, -1)
    with torch.inference_mode():
        independent_output = independent(audio)
        reference_output = reference(audio)
    assert torch.equal(independent_output.audio_codes, reference_output.audio_codes)
    assert torch.equal(independent_output.audio_values, reference_output.audio_values)

    monkeypatch.setattr(independent, "_cuda_graph_enabled", False)
    monkeypatch.setattr(independent.quantizer, "_compiled_encode_enabled", False)
    monkeypatch.setattr(independent.encoder, "_compiled_forward_enabled", False)
    monkeypatch.setattr(independent.decoder, "_compiled_forward_enabled", False)
    compiled_audio = torch.linspace(-0.1, 0.1, 24_000, device=device).view(1, 1, -1)
    with torch.inference_mode():
        eager_output = independent(compiled_audio)
        oracle_output = reference(compiled_audio)
    if os.environ.get("FAST_MIMI_REQUIRE_CUDA_RVQ") == "1":
        assert independent.quantizer._cuda_decode_backend is not None
        assert not independent.quantizer._cuda_decode_failed
    monkeypatch.setattr(independent.quantizer, "_compiled_encode_enabled", True)
    monkeypatch.setattr(independent.encoder, "_compiled_forward_enabled", True)
    monkeypatch.setattr(independent.decoder, "_compiled_forward_enabled", True)
    with torch.inference_mode():
        compiled_output = independent(compiled_audio)
    assert independent.quantizer._compiled_encode is not None
    assert independent.encoder._compiled_forward is not None
    assert independent.decoder._compiled_forward is not None
    assert torch.equal(eager_output.audio_codes, compiled_output.audio_codes)
    assert torch.equal(eager_output.audio_values, compiled_output.audio_values)
    assert torch.equal(oracle_output.audio_codes, compiled_output.audio_codes)
    assert torch.equal(oracle_output.audio_values, compiled_output.audio_values)

    monkeypatch.setattr(independent, "_cuda_graph_enabled", True)
    monkeypatch.setattr(independent, "_cuda_graph_failed", False)
    with torch.inference_mode():
        graph_output = independent(compiled_audio)
    assert independent._cuda_graph_runner is not None
    assert torch.equal(compiled_output.audio_codes, graph_output.audio_codes)
    assert torch.equal(compiled_output.audio_values, graph_output.audio_values)
    retained_codes = graph_output.audio_codes.clone()
    retained_waveform = graph_output.audio_values.clone()
    zero_audio = torch.zeros_like(compiled_audio)
    with torch.inference_mode():
        zero_graph_output = independent(zero_audio)
        zero_oracle_output = reference(zero_audio)
    assert torch.equal(zero_oracle_output.audio_codes, zero_graph_output.audio_codes)
    assert torch.equal(zero_oracle_output.audio_values, zero_graph_output.audio_values)
    assert torch.equal(graph_output.audio_codes, retained_codes)
    assert torch.equal(graph_output.audio_values, retained_waveform)

    other_stream = torch.cuda.Stream(device=device)
    with torch.inference_mode(), torch.cuda.stream(other_stream):
        other_stream_output = independent(compiled_audio)
    other_stream.synchronize()
    assert torch.equal(compiled_output.audio_codes, other_stream_output.audio_codes)
    assert torch.equal(compiled_output.audio_values, other_stream_output.audio_values)

    if os.environ.get("FAST_MIMI_REQUIRE_CUDA_RVQ") == "1":
        monkeypatch.setenv("FAST_MIMI_CUDA_NVCC", "/definitely/missing/nvcc")
        independent.quantizer._cuda_decode_backend = None
        independent.quantizer._cuda_decode_failed = False
        with torch.inference_mode():
            fallback_output = independent.decode(compiled_output.audio_codes)
            fallback_oracle = reference.decode(compiled_output.audio_codes)
        assert independent.quantizer._cuda_decode_failed
        assert torch.equal(fallback_oracle.audio_values, fallback_output.audio_values)
