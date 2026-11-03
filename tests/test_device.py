import subprocess

from leptonkit import device

Y16 = "'Y16 ' 160x120 'UYVY' 160x120"
MJPG = "'MJPG' 640x480 1280x720"


def fake_nodes(monkeypatch, formats, list_devices=""):
    monkeypatch.setattr(device.glob, "glob", lambda pat: list(formats))
    monkeypatch.setattr(device, "device_info", lambda d: ("", formats[d]))
    monkeypatch.setattr(device, "_v4l2", lambda args, timeout=5.0: list_devices)


def test_offers_y16():
    assert device.offers_y16(Y16)
    assert not device.offers_y16(MJPG)
    assert not device.offers_y16("'Y16 ' 80x60")
    assert not device.offers_y16("")


def test_find_lepton_by_format_in_numeric_order(monkeypatch):
    # video10 would sort before video2 as a string; the node number decides.
    fake_nodes(monkeypatch, {"/dev/video10": Y16, "/dev/video0": MJPG, "/dev/video2": Y16})
    assert device.find_lepton() == "/dev/video2"


def test_find_lepton_falls_back_to_list_devices(monkeypatch):
    listing = (
        "Integrated Camera (usb-0000:00:14.0-8):\n"
        "\t/dev/video0\n"
        "\t/dev/video1\n"
        "\n"
        "PureThermal (fw:v1.3.0): PureThermal (usb-0000:00:14.0-2):\n"
        "\t/dev/video4\n"
        "\t/dev/video5\n"
    )
    fake_nodes(monkeypatch, {"/dev/video0": MJPG, "/dev/video4": ""}, listing)
    assert device.find_lepton() == "/dev/video4"


def test_find_lepton_none(monkeypatch):
    fake_nodes(monkeypatch, {"/dev/video0": MJPG}, "Integrated Camera (usb-1):\n\t/dev/video0\n")
    assert device.find_lepton() is None


def test_missing_v4l2_ctl_gives_empty_output(monkeypatch):
    def no_tool(*args, **kwargs):
        raise FileNotFoundError("v4l2-ctl")

    monkeypatch.setattr(device.subprocess, "run", no_tool)
    assert device.device_info("/dev/video0") == ("", "")

    def too_slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("v4l2-ctl", 5.0)

    monkeypatch.setattr(device.subprocess, "run", too_slow)
    assert device._v4l2(["--list-devices"]) == ""
