import threading

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
