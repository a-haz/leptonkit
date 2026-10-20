import json

import numpy as np

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
