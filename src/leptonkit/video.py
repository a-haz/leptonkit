"""Write 8-bit frames to an MP4 with ffmpeg, when ffmpeg is installed."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np


def write_mp4(path, frames, fps: float = 9.0, size=(640, 480)) -> bool:
    """Encode uint8 grayscale frames (upscaled with nearest neighbour) to H.264.

    Returns False without writing anything when ffmpeg is not on the PATH, and
    False when ffmpeg fails (for example a build without libx264)."""
    import cv2

    if shutil.which("ffmpeg") is None:
        return False
    w, h = size
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "gray",
         "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    try:
        for f in frames:
            up = cv2.resize(np.asarray(f, dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            ff.stdin.write(up.tobytes())
    except BrokenPipeError:
        pass   # ffmpeg exited early and printed why; the return code says it failed
    finally:
        try:
            ff.stdin.close()
        except BrokenPipeError:
            pass
        ff.wait()
    return ff.returncode == 0
