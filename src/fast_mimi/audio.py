"""Small audio I/O helpers used by the command-line tools.

Audio dependencies remain optional so importing the model runtime does not
require NumPy, SciPy, or SoundFile.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor


def load_audio(path: str | Path, target_rate: int = 24_000) -> Tensor:
    """Load and normalize an audio file for Mimi inference.

    Args:
        path: Input file understood by SoundFile.
        target_rate: Requested output sampling rate in hertz. Input at another
            rate is resampled with a rational polyphase filter.

    Returns:
        Contiguous mono float32 tensor with shape ``[1, 1, samples]``.

    Raises:
        ImportError: If the optional audio dependencies are not installed.
        RuntimeError: If SoundFile cannot read the requested input.
    """
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as error:
        raise ImportError(
            "audio I/O requires `pip install fast-mimi[audio]`"
        ) from error

    samples, sampling_rate = sf.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sampling_rate != target_rate:
        divisor = math.gcd(sampling_rate, target_rate)
        samples = resample_poly(
            samples, target_rate // divisor, sampling_rate // divisor
        ).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(samples)).view(1, 1, -1)


def save_audio(path: str | Path, audio: Tensor, sampling_rate: int = 24_000) -> None:
    """Write a tensor as floating-point audio through SoundFile.

    Args:
        path: Destination path whose suffix selects the SoundFile format.
        audio: Waveform tensor. A three-dimensional tensor contributes its
            first batch item; a two-dimensional tensor is interpreted as
            ``[channels, samples]`` and transposed for SoundFile.
        sampling_rate: Output sampling rate in hertz.

    Returns:
        ``None`` after the file is written.

    Raises:
        ImportError: If SoundFile is not installed.
        RuntimeError: If SoundFile cannot encode or write the destination.
    """
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError(
            "audio I/O requires `pip install fast-mimi[audio]`"
        ) from error

    values = audio.detach().float().cpu()
    if values.ndim == 3:
        values = values[0]
    if values.ndim == 2:
        values = values.transpose(0, 1)
    sf.write(path, values.numpy(), sampling_rate, subtype="FLOAT")
