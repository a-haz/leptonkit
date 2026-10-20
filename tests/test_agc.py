import numpy as np
import pytest

from leptonkit import agc, y16


def scene(seed=0, person_c=34.0):
    """160x120 centikelvin frame: 21 C background, a warm person, a hot spot."""
    rng = np.random.default_rng(seed)
    c = np.full((120, 160), 21.0) + rng.normal(0, 0.05, (120, 160))
    c[30:100, 70:95] = person_c
    c[5:10, 5:10] = 60.0
    return y16.celsius_to_raw(c)


def test_output_is_uint8_same_shape_full_range():
    out = agc.heq(scene())
    assert out.dtype == np.uint8
    assert out.shape == (120, 160)
    assert out.min() <= 5
    assert out.max() >= 250


def test_monotonic_in_input():
    raw = scene()
    out = agc.heq(raw).astype(int)
    order = np.argsort(raw.ravel(), kind="stable")
    assert np.all(np.diff(out.ravel()[order]) >= 0)


def test_warm_person_brighter_than_background():
    out = agc.heq(scene())
    assert out[60, 80] > out[110, 10] + 50


def test_flat_frame_gives_zeros():
    out = agc.heq(np.full((120, 160), 29315, np.uint16))
    assert out.dtype == np.uint8
    assert not out.any()


def test_empty_frame():
    out, lut = agc.heq(np.zeros((0, 0), np.uint16), return_lut=True)
    assert out.size == 0 and lut is None


def test_linear_percent_one_is_linear_stretch():
    raw = scene()
    out = agc.heq(raw, linear_percent=1.0)
    lin = np.clip(agc.linear_minmax(raw) + 0.5, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(out, lin)


def test_linear_minmax_percentiles():
    ramp = np.arange(10000, dtype=np.float32).reshape(100, 100)
    out = agc.linear_minmax(ramp)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 255.0
    lo, hi = np.percentile(ramp, (1, 99))
    assert out[50, 0] == pytest.approx((5000 - lo) / (hi - lo) * 255, rel=1e-5)


def test_max_gain_limits_contrast():
    rng = np.random.default_rng(3)
    raw = (29315 + rng.normal(0, 2, (120, 160))).astype(np.float32)
    raw[0, 0] = 40000.0  # one outlier sets the full range
    free = agc.linear_minmax(raw)
    capped = agc.linear_minmax(raw, max_gain=2.0)
    assert np.std(capped) < np.std(free)


def test_plateau_spreads_less_contrast_into_background():
    # 90 % of pixels at a near-constant background: plain HEQ gives the
    # background most of the gray range, a low plateau keeps it compact.
    rng = np.random.default_rng(4)
    raw = (29315 + rng.normal(0, 3, (120, 160))).astype(np.uint16)
    raw[:, 144:] = np.linspace(30000, 31000, 16).astype(np.uint16)
    bg = (slice(None), slice(0, 144))
    plain = agc.heq(raw, linear_percent=0.0, clip_limit_high=1.0)
    plateau = agc.heq(raw, linear_percent=0.0, clip_limit_high=0.002)
    assert np.ptp(plateau[bg]) < np.ptp(plain[bg])


def test_dampening_blends_lookup_tables():
    a, b = scene(1), scene(2, person_c=36.0)
    _, lut_a = agc.heq(a, return_lut=True)
    _, lut_b = agc.heq(b, return_lut=True)
    _, lut_ab = agc.heq(b, prev_lut=lut_a, return_lut=True)
    np.testing.assert_allclose(lut_ab, 0.25 * lut_a + 0.75 * lut_b, rtol=1e-5, atol=1e-3)
    _, lut_off = agc.heq(b, prev_lut=lut_a, dampening=0, return_lut=True)
    np.testing.assert_allclose(lut_off, lut_b)


def test_agc_class_keeps_state_and_resets():
    a, b = scene(1), scene(2, person_c=36.0)
    stream = agc.Agc()
    first = stream(a)
    np.testing.assert_array_equal(first, agc.heq(a))
    second = stream(b)
    _, lut_a = agc.heq(a, return_lut=True)
    np.testing.assert_array_equal(second, agc.heq(b, prev_lut=lut_a))
    stream.reset()
    np.testing.assert_array_equal(stream(b), agc.heq(b))
