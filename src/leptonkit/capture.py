"""Capture a fixed number of radiometric frames to a folder.

Writes y16_stack.npy (N, 120, 160) uint16, timestamps_unix.npy, white-hot and
black-hot previews of the middle frame and, with ffmpeg installed, an MP4.
"""
from __future__ import annotations

import os
import time

import numpy as np

from leptonkit import agc, y16
from leptonkit.device import open_lepton
from leptonkit.video import write_mp4


def summarize(stack: np.ndarray, seconds: float) -> dict:
    raw = stack.astype(np.float64)
    out = {
        "frames": int(len(stack)),
        "shape": list(stack.shape),
        "seconds": round(seconds, 2),
        "fps": round(len(stack) / seconds, 2) if seconds > 0 else None,
        "raw_min_median_max": [float(raw.min()), float(np.median(raw)), float(raw.max())],
        "radiometric": y16.looks_radiometric(stack),
    }
    if out["radiometric"]:
        c = y16.raw_to_celsius(stack)
        out["celsius_min_median_max"] = [round(float(c.min()), 2), round(float(np.median(c)), 2),
                                         round(float(c.max()), 2)]
    return out


def save_capture(out_dir, frames, stamps, fps: float = 9.0, mp4: bool = True) -> dict:
    """Save frames and stamps plus previews; returns what was written."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    stack = np.stack(frames).astype(np.uint16)
    np.save(os.path.join(out_dir, "y16_stack.npy"), stack)
    np.save(os.path.join(out_dir, "timestamps_unix.npy"), np.asarray(stamps, dtype=np.float64))
    mid = agc.heq(stack[len(stack) // 2])
    up = lambda im: cv2.resize(im, (640, 480), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    cv2.imwrite(os.path.join(out_dir, "preview_whitehot.png"), up(mid))
    cv2.imwrite(os.path.join(out_dir, "preview_blackhot.png"), up(255 - mid))
    written = ["y16_stack.npy", "timestamps_unix.npy", "preview_whitehot.png", "preview_blackhot.png"]
    if mp4:
        stream = agc.Agc()
        if write_mp4(os.path.join(out_dir, "capture_whitehot.mp4"), (stream(f) for f in stack), fps=fps):
            written.append("capture_whitehot.mp4")
    return {"dir": os.path.abspath(out_dir), "files": written}


def capture(dev: str, n_frames: int, out_dir, mp4: bool = True) -> dict:
    cap = open_lepton(dev)
    frames, stamps = [], []
    t0 = time.time()
    try:
        for _ in range(n_frames):
            ok, f = cap.read()
            if not ok:
                continue
            frames.append(y16.to_u16(f))
            stamps.append(time.time())
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("no frames captured")
    elapsed = time.time() - t0
    info = summarize(np.stack(frames), elapsed)
    info.update(save_capture(out_dir, frames, stamps, fps=round(info["fps"] or 9.0, 1), mp4=mp4))
    return info
