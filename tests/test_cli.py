import json

import numpy as np
import pytest

from leptonkit import cli, y16


def frames(n=3):
    c = np.full((n, 120, 160), 22.0)
    c[:, 40:80, 60:100] = 35.0
    return y16.celsius_to_raw(c)


def test_stats_on_stack(tmp_path, capsys):
    p = tmp_path / "y16_stack.npy"
    np.save(p, frames())
    assert cli.main(["stats", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["frames"] == 3
    assert out["radiometric"] is True
    assert out["celsius_min"] == 22.0
    assert out["celsius_max"] == 35.0


def test_render_y16_and_stack(tmp_path):
    raw = tmp_path / "raw_100.y16"
    y16.write_y16(raw, frames(1)[0])
    stack = tmp_path / "y16_stack.npy"
    np.save(stack, frames(2))
    out = tmp_path / "png"
    assert cli.main(["render", str(raw), str(stack), "--out", str(out), "--scale", "2"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["raw_100.png", "y16_stack_00000.png", "y16_stack_00001.png"]
    import cv2

    img = cv2.imread(str(out / "raw_100.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (240, 320)
    assert img[120, 160] > img[10, 10]


def test_record_and_streams_honour_lepton_dev(tmp_path, monkeypatch):
    from leptonkit import streams, trigger

    seen = {}

    class FakeRecorder:
        def __init__(self, out, device=None, **kw):
            seen["record"] = device

        def run(self):
            pass

    monkeypatch.setattr(trigger, "Recorder", FakeRecorder)
    monkeypatch.setattr(streams, "record", lambda out, lepton_dev=None, **kw: seen.update(streams=lepton_dev))
    monkeypatch.setenv("LEPTON_DEV", "/dev/video7")
    assert cli.main(["record", "--out", str(tmp_path)]) == 0
    assert cli.main(["streams", str(tmp_path / "s")]) == 0
    assert seen == {"record": "/dev/video7", "streams": "/dev/video7"}
    assert cli.main(["record", "--out", str(tmp_path), "--device", "/dev/video3"]) == 0
    assert seen["record"] == "/dev/video3"                   # --device still wins


@pytest.mark.parametrize("argv", [
    ["capture", "--frames", "0"],
    ["capture", "--frames", "-5"],
    ["pair", "session", "--stride", "0"],
    ["render", "raw_1.y16", "--out", "png", "--scale", "0"],
])
def test_counts_must_be_positive(argv, monkeypatch, capsys):
    from leptonkit import capture

    monkeypatch.setenv("LEPTON_DEV", "/dev/video9")
    monkeypatch.setattr(capture, "capture", lambda *a, **kw: pytest.fail("capture ran"))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2 and "at least 1" in capsys.readouterr().err


def test_record_needs_an_ordered_body_band(tmp_path, monkeypatch):
    from leptonkit import trigger

    monkeypatch.setattr(trigger, "Recorder", lambda *a, **kw: pytest.fail("recorder started"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["record", "--out", str(tmp_path), "--body-lo", "38", "--body-hi", "28.5"])
    assert "--body-lo" in str(exc.value.code)
