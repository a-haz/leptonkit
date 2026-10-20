"""Y16 frame helpers.

With radiometry and TLinear enabled, a Lepton 3.5 sends every pixel as an
unsigned 16-bit value in centikelvin, so degrees C = raw / 100 - 273.15.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WIDTH = 160
HEIGHT = 120
KELVIN_OFFSET = 273.15


def raw_to_celsius(raw) -> np.ndarray:
    """Centikelvin counts to degrees C (float64)."""
    return np.asarray(raw, dtype=np.float64) / 100.0 - KELVIN_OFFSET


def celsius_to_raw(celsius) -> np.ndarray:
    """Degrees C to centikelvin counts (uint16, rounded and clipped)."""
    raw = np.round((np.asarray(celsius, dtype=np.float64) + KELVIN_OFFSET) * 100.0)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def looks_radiometric(raw) -> bool:
    """True when the median looks like centikelvin for an everyday scene (250 K to 400 K).

    A Lepton with TLinear off sends relative counts (around 8000) instead."""
    med = float(np.median(raw))
    return 25000.0 < med < 40000.0


def to_u16(frame, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Turn a frame from OpenCV's V4L2 backend into a (height, width) uint16 array.

    Depending on the OpenCV build, a Y16 frame arrives as uint16, as (H, W, 2)
    uint8 or as (H, 2W) uint8. Bytes are little-endian."""
    a = np.asarray(frame)
    if a.dtype != np.uint16:
        a = np.ascontiguousarray(a, dtype=np.uint8)
        if a.size == width * height * 2 and a.shape[0] != height:
            a = a.reshape(height, width * 2)
        a = a.view("<u2").reshape(a.shape[0], -1)
    return np.ascontiguousarray(a[:height, :width], dtype=np.uint16)


def read_y16(path, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Read a headerless little-endian Y16 file (one frame) as (height, width) uint16."""
    raw = np.fromfile(Path(path), dtype="<u2")
    if raw.size != width * height:
        raise ValueError(f"{path}: {raw.size} values, expected {width}x{height}")
    return raw.reshape(height, width).astype(np.uint16)


def write_y16(path, frame) -> None:
    """Write one frame as headerless little-endian uint16, row by row."""
    np.asarray(frame, dtype="<u2").tofile(Path(path))


def stretch_percentile(img, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Linear stretch between two percentiles to uint8 (a quick preview, not the Lepton AGC)."""
    f = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(f, [lo_pct, hi_pct])
    return np.clip((f - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
