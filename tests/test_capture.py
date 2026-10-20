import numpy as np

from leptonkit import capture, y16


def test_summarize_reports_celsius():
    c = np.full((4, 120, 160), 20.0)
    c[:, 50:60, 50:60] = 36.5
    info = capture.summarize(y16.celsius_to_raw(c), seconds=0.5)
    assert info["frames"] == 4
    assert info["fps"] == 8.0
    assert info["radiometric"] is True
    assert info["celsius_min_median_max"] == [20.0, 20.0, 36.5]


def test_summarize_non_radiometric():
    info = capture.summarize(np.full((2, 120, 160), 8000, np.uint16), seconds=1.0)
    assert info["radiometric"] is False
    assert "celsius_min_median_max" not in info


def test_save_capture_writes_stack_and_previews(tmp_path):
    frames = [y16.celsius_to_raw(np.full((120, 160), 20.0 + i)) for i in range(5)]
    stamps = [1000.0 + 0.11 * i for i in range(5)]
    res = capture.save_capture(tmp_path / "cap", frames, stamps, mp4=False)
    stack = np.load(tmp_path / "cap" / "y16_stack.npy")
    assert stack.shape == (5, 120, 160) and stack.dtype == np.uint16
    np.testing.assert_allclose(np.load(tmp_path / "cap" / "timestamps_unix.npy"), stamps)
    assert (tmp_path / "cap" / "preview_whitehot.png").exists()
    assert (tmp_path / "cap" / "preview_blackhot.png").exists()
    assert "capture_whitehot.mp4" not in res["files"]
