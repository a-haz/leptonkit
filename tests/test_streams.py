import json
import threading

import pytest

from leptonkit import streams


def test_find_devices_by_format(monkeypatch):
    info = {
        "/dev/video0": ("Card type: Integrated Camera", "'MJPG' 640x480 1280x720"),
        "/dev/video2": ("Card type: PureThermal (fw:v1.3.0)", "'Y16 ' 160x120 'UYVY' 160x120"),
        "/dev/video4": ("Card type: USB Camera 046d:0825", "'MJPG' 640x480"),
        "/dev/video10": ("Card type: metadata", ""),
    }
    monkeypatch.setattr(streams.glob, "glob", lambda pat: list(info))
    monkeypatch.setattr(streams, "device_info", lambda d: info[d])
    assert streams.find_devices() == ("/dev/video2", "/dev/video0")
    assert streams.find_devices(rgb_match="046d:0825") == ("/dev/video2", "/dev/video4")


def test_report_rates():
    stop = threading.Event()
    a = streams.Stream("lepton", stop)
    a.n, a.stamps = 3, [10.0, 10.5, 11.0]
    b = streams.Stream("rgb", stop)
    b.err = "cannot open /dev/video9"
    rep = streams.report([a, b], {"lepton": "/dev/video2", "rgb": "/dev/video9"}, 9.0, 12.0)
    assert rep["lepton"] == {"frames": 3, "error": None, "fps": 2.0}
    assert rep["rgb"] == {"frames": 0, "error": "cannot open /dev/video9", "fps": 0.0}
    assert rep["devices"]["lepton"] == "/dev/video2"


def test_ctrl_c_still_saves(tmp_path, monkeypatch):
    saved = []

    class FakeLepton(streams.Stream):
        def __init__(self, dev, out, stop_evt):
            super().__init__("lepton", stop_evt)

        def run(self):
            self.stop_evt.wait()
            saved.append("y16_stack.npy")      # stands in for the files written after the stop

    def interrupted(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(streams, "find_devices", lambda rgb_match=None: ("/dev/video2", None))
    monkeypatch.setattr(streams, "LeptonStream", FakeLepton)
    monkeypatch.setattr(streams.time, "sleep", interrupted)
    try:
        rep = streams.record(str(tmp_path), duration=60.0)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped record() before the streams were stopped and saved")
    assert saved == ["y16_stack.npy"]
    assert json.loads((tmp_path / "streams.json").read_text())["lepton"] == rep["lepton"]


def test_rgb_video_keeps_every_stamped_frame(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    stop = threading.Event()

    class FakeCapture:
        """A webcam that ignores the 640x480 request and sends 1280x720."""

        def __init__(self, *a):
            self.left = 3

        def set(self, *a):
            return False

        def isOpened(self):
            return True

        def read(self):
            if self.left == 0:
                stop.set()
                return False, None
            self.left -= 1
            return True, np.full((720, 1280, 3), 60 * self.left, np.uint8)

        def release(self):
            pass

    read_back = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    rgb = streams.RgbStream("/dev/video0", str(tmp_path), stop)
    rgb.run()
    cap = read_back(str(tmp_path / "rgb.avi"))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert rgb.n == len(rgb.stamps) == n == 3
