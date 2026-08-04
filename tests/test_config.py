from __future__ import annotations

import json

import pytest
import torch

from fast_mimi import KYUTAI_MIMI_REVISION, MimiConfig, MimiModel


def test_published_defaults() -> None:
    config = MimiConfig()
    assert KYUTAI_MIMI_REVISION == "89091b3e466eb6a9d11e537bf26b144f194978f7"
    assert config.sampling_rate == 24_000
    assert config.frame_rate == 12.5
    assert config.encodec_frame_rate == 25
    assert config.frame_size == 1_920
    assert config.num_codebooks == 32
    assert config.codebook_size == 2_048


def test_json_round_trip_maps_frame_rate(tmp_path) -> None:
    config = MimiConfig()
    path = config.save_pretrained(tmp_path)
    loaded = MimiConfig.from_json_file(path)
    assert loaded == config
    assert json.loads(path.read_text())["frame_rate"] == 12.5


def test_encoded_lengths_match_frame_boundaries() -> None:
    model = MimiModel(MimiConfig())
    lengths = torch.tensor([1, 1_919, 1_920, 1_921, 24_000, 48_001])
    assert model.get_encoded_length(lengths).tolist() == [1, 1, 1, 2, 13, 26]


def test_invalid_quantizer_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_semantic_quantizers"):
        MimiConfig(num_quantizers=1, num_semantic_quantizers=1)
