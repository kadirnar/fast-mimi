from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

import fast_mimi
from fast_mimi import (
    KYUTAI_MIMI_MODEL_ID,
    KYUTAI_MIMI_PARAMETER_COUNT,
    KYUTAI_MIMI_PARAMETER_FINGERPRINT,
    KYUTAI_MIMI_REVISION,
    KYUTAI_MIMI_WEIGHTS_SHA256,
    MimiConfig,
    MimiModel,
)


def test_published_defaults() -> None:
    config = MimiConfig()
    assert KYUTAI_MIMI_MODEL_ID == "kyutai/mimi"
    assert KYUTAI_MIMI_REVISION == "89091b3e466eb6a9d11e537bf26b144f194978f7"
    assert KYUTAI_MIMI_PARAMETER_COUNT == 79_308_609
    assert len(KYUTAI_MIMI_PARAMETER_FINGERPRINT) == 64
    assert len(KYUTAI_MIMI_WEIGHTS_SHA256) == 64
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


def test_declared_checkpoint_revision_is_locked() -> None:
    with pytest.raises(ValueError, match="revision changed"):
        MimiModel.from_pretrained(
            KYUTAI_MIMI_MODEL_ID,
            revision="main",
            local_files_only=True,
        )


def test_production_runtime_has_no_transformers_import() -> None:
    package = Path(fast_mimi.__file__).parent
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert all(name != "transformers" for name in imports), source_path
