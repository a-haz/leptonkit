import numpy as np
import pytest

from leptonkit import y16


def synthetic_frame(seed=0):
    """A 160x120 centikelvin frame: 21 C room, a 34 C person-sized blob, a little noise."""
    rng = np.random.default_rng(seed)
    c = np.full((y16.HEIGHT, y16.WIDTH), 21.0) + rng.normal(0, 0.05, (y16.HEIGHT, y16.WIDTH))
    c[30:100, 70:95] = 34.0
    return y16.celsius_to_raw(c)


def test_raw_to_celsius_known_values():
    assert y16.raw_to_celsius(27315) == pytest.approx(0.0)
    assert y16.raw_to_celsius(29315) == pytest.approx(20.0)
    assert y16.raw_to_celsius(31015) == pytest.approx(37.0)
    assert y16.raw_to_celsius(0) == pytest.approx(-273.15)


def test_conversion_matches_formula_on_a_frame():
    raw = synthetic_frame()
    c = y16.raw_to_celsius(raw)
    assert c.dtype == np.float64
    np.testing.assert_allclose(c, raw.astype(np.float64) / 100.0 - 273.15)
    assert np.median(c) == pytest.approx(21.0, abs=0.05)
    assert c[60, 80] == pytest.approx(34.0, abs=0.01)


def test_celsius_round_trip_within_one_count():
    c = np.linspace(-20.0, 120.0, 1001)
    back = y16.raw_to_celsius(y16.celsius_to_raw(c))
    assert np.max(np.abs(back - c)) <= 0.005 + 1e-9


def test_celsius_to_raw_clips():
    assert y16.celsius_to_raw(-300.0) == 0
    assert y16.celsius_to_raw(1000.0) == 65535


def test_looks_radiometric():
    assert y16.looks_radiometric(synthetic_frame())
    assert not y16.looks_radiometric(np.full((120, 160), 8000, np.uint16))


def test_to_u16_accepts_opencv_layouts():
    raw = synthetic_frame()
    as_bytes = raw.astype("<u2").view(np.uint8)
    np.testing.assert_array_equal(y16.to_u16(raw), raw)
    np.testing.assert_array_equal(y16.to_u16(as_bytes.reshape(120, 160, 2)), raw)
    np.testing.assert_array_equal(y16.to_u16(as_bytes.reshape(120, 320)), raw)
    np.testing.assert_array_equal(y16.to_u16(as_bytes.reshape(1, -1)), raw)


def test_y16_file_round_trip(tmp_path):
    raw = synthetic_frame()
    p = tmp_path / "raw_1.y16"
    y16.write_y16(p, raw)
    assert p.stat().st_size == 160 * 120 * 2
    assert p.read_bytes()[:2] == int(raw[0, 0]).to_bytes(2, "little")
    np.testing.assert_array_equal(y16.read_y16(p), raw)
    with pytest.raises(ValueError):
        y16.read_y16(p, width=80, height=60)


def test_stretch_percentile_range():
    out = y16.stretch_percentile(synthetic_frame())
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
