import numpy as np
import pytest

from leptonkit import video


def test_write_mp4_returns_false_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(video.shutil, "which", lambda _: None)
    out = tmp_path / "out.mp4"
    frames = [np.zeros((120, 160), np.uint8)]
    assert video.write_mp4(out, frames) is False
    assert not out.exists()


def test_write_mp4_encodes_grayscale_frames(tmp_path):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    frames = [np.full((120, 160), i * 40, np.uint8) for i in range(5)]
    out = tmp_path / "out.mp4"
    assert video.write_mp4(out, frames, fps=9.0, size=(160, 120)) is True
    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_mp4_returns_false_when_ffmpeg_fails(monkeypatch, tmp_path):
    class FakeProc:
        returncode = 1

        def __init__(self, *args, **kwargs):
            self.stdin = _FakeStdin()

        def wait(self):
            return self.returncode

    class _FakeStdin:
        def write(self, data):
            pass

        def close(self):
            pass

    monkeypatch.setattr(video.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.subprocess, "Popen", FakeProc)
    out = tmp_path / "out.mp4"
    frames = [np.zeros((120, 160), np.uint8)]
    assert video.write_mp4(out, frames, size=(160, 120)) is False
