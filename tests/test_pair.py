import cv2
import numpy as np

from leptonkit import cli, y16
from leptonkit.pair import pair_session


def make_session(root, n_tir=10, n_rgb=4):
    (root / "tir").mkdir(parents=True)
    (root / "rgb").mkdir()
    t0 = 2_228_986_602_698_631
    for i in range(n_tir):
        c = np.full((120, 160), 21.0)
        c[40:80, 60 + i:80 + i] = 35.0
        y16.write_y16(root / "tir" / f"raw_{t0 + i * 115_000_000}.y16", y16.celsius_to_raw(c))
    for j in range(n_rgb):
        img = np.full((480, 640, 3), 40 * j, np.uint8)
        cv2.imwrite(str(root / "rgb" / f"rgb_{t0 + j * 300_000_000}.jpg"), img)
    return t0


def test_pairs_nearest_rgb(tmp_path):
    t0 = make_session(tmp_path)
    rows = pair_session(tmp_path, stride=3)
    assert rows[0] == "tir_ns,rgb_ns,dt_ms,file"
    assert len(rows) == 1 + 4   # thermal frames 0, 3, 6, 9
    tir_ns, rgb_ns, dt_ms, name = rows[2].split(",")
    assert int(tir_ns) == t0 + 3 * 115_000_000
    assert int(rgb_ns) == t0 + 1 * 300_000_000   # 345 ms is nearest to 300 ms
    assert float(dt_ms) == -45.0
    img = cv2.imread(str(tmp_path / "pairs" / name))
    assert img.shape == (480 + 22, 640 + 640, 3)
    assert (tmp_path / "pairs" / "index.csv").read_text().count("\n") == 5


def test_rotation_changes_panel_width(tmp_path):
    make_session(tmp_path)
    rows = pair_session(tmp_path, rot="ccw90", stride=10)
    img = cv2.imread(str(tmp_path / "pairs" / rows[1].split(",")[3]))
    assert img.shape == (480 + 22, 640 + 360, 3)


def test_skips_frames_of_another_size(tmp_path, capsys):
    t0 = make_session(tmp_path, n_tir=2)
    np.zeros(256 * 192, "<u2").tofile(tmp_path / "tir" / f"raw_{t0 + 1}.y16")
    rows = pair_session(tmp_path, stride=1)
    assert len(rows) == 1 + 2
    assert "skip:" in capsys.readouterr().out


def test_skips_an_unreadable_rgb_frame(tmp_path, capsys):
    t0 = make_session(tmp_path, n_tir=10, n_rgb=4)
    (tmp_path / "rgb" / f"rgb_{t0 + 3 * 300_000_000}.jpg").write_bytes(b"\xff\xd8 cut short")
    rows = pair_session(tmp_path, stride=3)
    assert len(rows) == 1 + 3            # thermal frame 9 has only the broken JPEG nearby
    assert "skip: cannot read" in capsys.readouterr().out


def test_cli_pair(tmp_path, capsys):
    make_session(tmp_path)
    assert cli.main(["pair", str(tmp_path), "--stride", "5"]) == 0
    assert "2 pairs" in capsys.readouterr().out
