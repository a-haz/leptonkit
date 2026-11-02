"""Side-by-side RGB and thermal panels for a ThermalRig session.

For every stride-th tir/raw_<t_ns>.y16 the nearest rgb/rgb_<t_ns>.jpg (same
boot clock) is found and pairs/pair_<t_ns>.png is written: the RGB frame next
to the colour-mapped thermal frame, with the time offset and the temperature
range printed on top. pairs/index.csv lists every pair.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from leptonkit import y16

ROT_K = {"none": 0, "ccw90": 1, "180": 2, "cw90": 3}   # np.rot90 k
PANEL_H = 480
BAR_H = 22


def ts_of(p: Path) -> int:
    return int(re.search(r"_(\d+)\.", p.name).group(1))


def tir_panel(c: np.ndarray, k: int):
    """Colour-mapped panel of a temperature image, with its 2 and 98 percentile range."""
    import cv2

    if k:
        c = np.rot90(c, k=k)
    lo, hi = np.percentile(c, 2), np.percentile(c, 98)
    g = (np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    s = PANEL_H / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * s), PANEL_H), interpolation=cv2.INTER_NEAREST)
    return img, float(lo), float(hi)


def pair_session(session, rot: str = "none", rgb_rot: str = "none", stride: int = 9) -> list[str]:
    """Write the pair panels; returns the rows of index.csv (header first)."""
    import cv2

    sess = Path(session).expanduser()
    tirs = sorted(sess.glob("tir/raw_*.y16"), key=ts_of)[::stride]
    rgbs = sorted(sess.glob("rgb/rgb_*.jpg"), key=ts_of)
    if not tirs or not rgbs:
        raise FileNotFoundError(f"missing streams in {sess}: {len(tirs)} tir / {len(rgbs)} rgb")
    rgb_ts = np.array([ts_of(p) for p in rgbs])
    out = sess / "pairs"
    out.mkdir(exist_ok=True)
    rows = ["tir_ns,rgb_ns,dt_ms,file"]
    for tp in tirs:
        t = ts_of(tp)
        rp = rgbs[int(np.argmin(np.abs(rgb_ts - t)))]
        dt_ms = (ts_of(rp) - t) / 1e6
        try:
            c = y16.raw_to_celsius(y16.read_y16(tp))
        except ValueError as e:
            print("skip:", e)
            continue
        rgb = cv2.imread(str(rp))
        if rgb is None:                     # e.g. a JPEG cut short when the app was killed
            print("skip: cannot read", rp)
            continue
        tir, lo, hi = tir_panel(c, ROT_K[rot])
        if ROT_K[rgb_rot]:
            rgb = np.ascontiguousarray(np.rot90(rgb, k=ROT_K[rgb_rot]))
        s = PANEL_H / rgb.shape[0]
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), PANEL_H))
        panel = np.zeros((PANEL_H + BAR_H, rgb.shape[1] + tir.shape[1], 3), np.uint8)
        panel[BAR_H:, :rgb.shape[1]] = rgb
        panel[BAR_H:, rgb.shape[1]:] = tir
        cv2.putText(panel, f"t={t}  dt={dt_ms:+.0f}ms  TIR {lo:.1f}..{hi:.1f}C  rot={rot}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        f = out / f"pair_{t}.png"
        cv2.imwrite(str(f), panel)
        rows.append(f"{t},{ts_of(rp)},{dt_ms:.1f},{f.name}")
    (out / "index.csv").write_text("\n".join(rows) + "\n")
    return rows
