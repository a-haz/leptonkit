# leptonkit

leptonkit is a small Python package for the radiometric 16-bit (Y16) output
of a FLIR Lepton on a PureThermal board under Linux. It finds the camera,
captures raw frames, records events when a person walks through the view,
records the Lepton and a webcam on one clock, converts raw values to degrees
C, and turns raw frames into 8-bit images with a Lepton-style histogram
equalisation AGC. I wrote it for my own research with a thermal camera.

## Hardware

- FLIR Lepton 3.5 on a PureThermal board, as a UVC camera over USB.
  Temperatures need radiometry with TLinear enabled, which makes every value
  centikelvin. `capture` reports `"radiometric": false` and `record` logs a
  warning when the values do not look like centikelvin.
- Optional: any USB webcam that offers MJPG at 640x480, for `streams`.
- Linux with V4L2 and `v4l2-ctl` (package v4l-utils) for device detection.
  `ffmpeg` is used for MP4 previews when it is installed.

## Install

```
pip install .
```

This installs numpy, opencv-python-headless and a `leptonkit` command.

## Running tests

The tests run on Python 3.9 or newer and do not need a camera, since they
work on synthetic frames. From the repository root, optionally in a fresh
virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install '.[test]'
pytest
```

## Usage

```
leptonkit find                                  # print the Lepton's /dev/videoN
leptonkit capture --frames 135 --out cap1       # about 15 s of raw frames
leptonkit stats cap1/y16_stack.npy              # min, median and max in C
leptonkit render cap1/y16_stack.npy --out png --palette inferno --scale 4
leptonkit record --out ~/thermal_events         # person-triggered events
leptonkit streams walk1 --duration 120          # Lepton and webcam together
leptonkit pair path/to/session --rot ccw90      # RGB and thermal side by side
```

The camera is found by the formats it offers (Y16 at 160x120), not by its
node number. Pass `--device /dev/videoN` or set `LEPTON_DEV` to choose it.
Only one program can read the camera at a time.

`record` keeps a live view at http://127.0.0.1:8089/ with all palettes side
by side (`--live-port 0` turns it off). It triggers on new pixels in the body
temperature band (28.5 to 38 C by default) compared with about a second
earlier, so a warm object that does not move never starts an event. Each
event keeps about 5 s from before the trigger, ends after about 5 s of quiet
and is capped at about 90 s. It stops recording when free space drops below
`--min-free-gb` or the events folder grows past `--max-total-gb`.

`pair` works on a session folder from the ThermalRig Android app
(`tir/raw_<t_ns>.y16` and `rgb/rgb_<t_ns>.jpg`). It writes
`pairs/pair_<t_ns>.png` and `pairs/index.csv` inside the session folder.

From Python:

```python
import numpy as np
from leptonkit import agc, y16

stack = np.load("cap1/y16_stack.npy")      # (N, 120, 160) uint16
celsius = y16.raw_to_celsius(stack)         # raw / 100 - 273.15
stream = agc.Agc()                          # keeps state for temporal dampening
gray = [stream(f) for f in stack]           # uint8 images
```

## AGC

`leptonkit.agc.heq` follows the AGC chapter of the FLIR Lepton Software
Interface Description Document: a histogram of the frame with a plateau
(clip limit high), an optional clip limit low, a lookup table from the
cumulative histogram, temporal dampening of that table between frames, and a
blend with a linear stretch (linear percent). The defaults are the IDD
register defaults for Lepton 3.x where they apply: no plateau clipping,
20 % linear and a dampening factor of 64. Clip limit low, which Lepton 3
replaces with linear percent, is off. It runs on the host from the raw
frames and is not bit-exact with the image the camera produces itself.

## File formats

`capture`, `record` events and `streams` write:

- `y16_stack.npy`: numpy array (N, 120, 160) of uint16 raw values.
- `timestamps_unix.npy`: float64 seconds since the epoch, one per frame,
  taken when the frame was read on the host.
- `preview_*.png`, `*.mp4`: 8-bit previews only.
- `record` also writes `meta.json` per event, and `events.jsonl` and
  `recorder.log` in the output folder.
- `streams` also writes `rgb.avi` (MJPG), `rgb_timestamps_unix.npy` and
  `streams.json` with frame counts and rates.

A `.y16` file is one headerless frame of little-endian uint16 values, row by
row, as written by the ThermalRig app.

## License

MIT, see LICENSE.
