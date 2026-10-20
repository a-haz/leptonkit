"""Lepton-style automatic gain control (AGC) from 16-bit frames to 8-bit images.

`heq` is a host-side version of the plateau histogram-equalisation AGC that
the Lepton runs on the camera, following the AGC chapter of the FLIR Lepton
Software Interface Description Document (IDD). The parameter names follow
the IDD registers: clip limit high, clip limit low, linear percent and
dampening factor. It is a reconstruction from the documentation, not a
bit-exact copy of the camera's output.

Steps:
  1. Histogram of the frame over n_bins between two robust percentiles.
  2. Clip every bin at clip_limit_high * N pixels (the plateau).
  3. Add clip_limit_low * N pixels, spread evenly over the non-empty bins.
  4. Cumulative histogram -> lookup table to 0..255.
  5. Optional temporal dampening of the lookup table across frames:
       lut = (D / 256) * previous + ((256 - D) / 256) * current
  6. Blend with a linear min-max stretch:
       out = linear_percent * linear + (1 - linear_percent) * heq
"""
from __future__ import annotations

import numpy as np

# Defaults, from the IDD register defaults for Lepton 3.x.
PERCENTILE = (1.0, 99.0)
# Lepton raw data is 14-bit; 4096 bins over the robust range.
N_BINS = 4096
# Clip limit high: maximum pixels in one bin. The register default is 19200,
# the whole 160x120 frame, so as a fraction 1.0 means no plateau clipping.
CLIP_LIMIT_HIGH = 1.0
# Clip limit low: the IDD default adds 512 pixels to every non-empty bin, and
# Lepton 3 replaces it with linear percent. Here it is a total fraction of N
# pixels split across the non-empty bins, and off by default.
CLIP_LIMIT_LOW = 0.0
# Linear percent: register default 20 %.
LINEAR_PERCENT = 0.20
# Dampening factor D in 0..256: register default 64, so the previous table
# keeps a weight of 64/256 = 0.25.
DAMPENING = 64
# Optional limit on the gain of the linear leg (not an IDD register), so a
# nearly flat scene cannot blow up the contrast. None = off.
MAX_GAIN = None


def linear_minmax(img, pct=PERCENTILE, max_gain=MAX_GAIN) -> np.ndarray:
    """Linear stretch between two percentiles to 0..255 (float32, not quantised).

    With max_gain set, the stretched span is never narrower than
    (full range of img) / max_gain, centred on the percentile range."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, pct)
    span = hi - lo
    if max_gain is not None and max_gain > 0:
        full = float(np.ptp(img)) + 1e-6
        if span < full / float(max_gain):
            mid = 0.5 * (lo + hi)
            half = 0.5 * full / float(max_gain)
            lo, hi = mid - half, mid + half
            span = hi - lo
    if span <= 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    x = np.clip((img - lo) / span, 0.0, 1.0)
    return (x * 255.0).astype(np.float32)


def heq(
    img,
    clip_limit_high=CLIP_LIMIT_HIGH,
    clip_limit_low=CLIP_LIMIT_LOW,
    linear_percent=LINEAR_PERCENT,
    dampening=DAMPENING,
    n_bins=N_BINS,
    pct=PERCENTILE,
    max_gain=MAX_GAIN,
    prev_lut=None,
    return_lut=False,
):
    """Plateau histogram-equalisation AGC of one frame to uint8.

    prev_lut is the lookup table returned for the previous frame
    (return_lut=True); pass it to get temporal dampening. Use the Agc class
    to have that done for you."""
    img = np.asarray(img, dtype=np.float32)
    n_bins = int(n_bins)
    clip_h = float(clip_limit_high)
    clip_l = float(clip_limit_low)
    lp = float(np.clip(float(linear_percent), 0.0, 1.0))
    damp_n = int(np.clip(int(dampening), 0, 256))

    flat = img.ravel()
    n_pix = flat.size
    if n_pix == 0:
        out = np.zeros_like(img, dtype=np.uint8)
        return (out, None) if return_lut else out

    lo, hi = np.percentile(flat, pct)
    if hi <= lo + 1e-9:
        lo = float(flat.min())
        hi = float(flat.max())
    if hi <= lo + 1e-9:
        out = np.zeros_like(img, dtype=np.uint8)
        return (out, None) if return_lut else out

    edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    idx = np.clip(np.digitize(flat, edges[1:-1], right=False), 0, n_bins - 1)
    hist = np.bincount(idx, minlength=n_bins).astype(np.float64)

    # clip limit high: plateau every bin
    hist = np.minimum(hist, max(1.0, clip_h * float(n_pix)))

    # clip limit low: total extra mass clip_l * N, spread over the non-empty bins
    if clip_l > 0:
        nonempty = hist > 0
        n_ne = int(nonempty.sum())
        if n_ne > 0:
            hist[nonempty] += (clip_l * float(n_pix)) / float(n_ne)

    total = hist.sum()
    if total <= 0:
        out = np.zeros_like(img, dtype=np.uint8)
        return (out, None) if return_lut else out

    lut = np.cumsum(hist) / total * 255.0

    if prev_lut is not None and damp_n > 0:
        prev = np.asarray(prev_lut, dtype=np.float64)
        if prev.shape == lut.shape:
            w_prev = damp_n / 256.0
            lut = w_prev * prev + (1.0 - w_prev) * lut

    heq_img = lut[idx].reshape(img.shape).astype(np.float32)
    lin_img = linear_minmax(img, pct=pct, max_gain=max_gain)
    out = np.clip(lp * lin_img + (1.0 - lp) * heq_img + 0.5, 0, 255).astype(np.uint8)
    if return_lut:
        return out, lut.astype(np.float32)
    return out


class Agc:
    """Stateful HEQ AGC for a stream: keeps the lookup table for dampening.

    >>> agc = Agc()
    >>> gray = agc(frame)       # frame: (H, W) uint16
    """

    def __init__(self, **settings):
        self.settings = settings
        self.lut = None

    def reset(self) -> None:
        self.lut = None

    def __call__(self, img) -> np.ndarray:
        out, lut = heq(img, prev_lut=self.lut, return_lut=True, **self.settings)
        if lut is not None:
            self.lut = lut
        return out
