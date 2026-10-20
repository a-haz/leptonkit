"""Find and open the PureThermal / Lepton on Linux (V4L2)."""
from __future__ import annotations

import glob
import re
import subprocess

from leptonkit.y16 import HEIGHT, WIDTH


def _v4l2(args, timeout=5.0) -> str:
    try:
        return subprocess.run(["v4l2-ctl"] + args, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def device_info(dev: str) -> tuple[str, str]:
    """(output of v4l2-ctl --all, output of v4l2-ctl --list-formats-ext) for one node."""
    return _v4l2(["-d", dev, "--all"]), _v4l2(["-d", dev, "--list-formats-ext"])


def offers_y16(formats: str) -> bool:
    return "Y16" in formats and f"{WIDTH}x{HEIGHT}" in formats


def find_lepton() -> str | None:
    """The first /dev/video* node that offers Y16 at 160x120, else None.

    Matching by format rather than by node number means a webcam that took
    /dev/video2 is never recorded as thermal. Falls back to the node listed
    under "PureThermal" in v4l2-ctl --list-devices."""
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(re.sub(r"\D", "", d) or 0)):
        _, formats = device_info(dev)
        if offers_y16(formats):
            return dev
    lines = _v4l2(["--list-devices"]).splitlines()
    for i, line in enumerate(lines):
        if "PureThermal" in line:
            for nxt in lines[i + 1:i + 4]:
                m = re.search(r"(/dev/video\d+)", nxt)
                if m:
                    return m.group(1)
    return None


def open_lepton(dev: str):
    """Open the device with OpenCV for raw Y16 160x120 frames. Returns a cv2.VideoCapture."""
    import cv2

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise OSError(f"cannot open {dev} (is another program using it?)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return cap
