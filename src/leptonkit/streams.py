"""Record the Lepton and an optional USB webcam together on one host clock.

  Lepton : radiometric Y16 160x120 (about 8.7 fps) -> y16_stack.npy (N, 120, 160) uint16
           + timestamps_unix.npy
  RGB    : webcam, MJPG 640x480                     -> rgb.avi + rgb_timestamps_unix.npy

Every frame is stamped with time.time() when it is read, so both streams share
one clock. Devices are found by what they offer, not by /dev/videoN.
Recording stops after the duration, when a file named STOP appears in the
output folder or on Ctrl-C. streams.json reports frame counts, rates and errors.
"""
from __future__ import annotations

import glob
import json
import os
import re
import threading
import time

import numpy as np

from leptonkit import agc, y16
from leptonkit.device import device_info, offers_y16


def find_devices(rgb_match: str | None = None) -> tuple[str | None, str | None]:
    """(lepton, rgb) V4L2 nodes. The webcam is the first other node offering MJPG at
    640x480, or the first whose v4l2-ctl --all output contains rgb_match."""
    lep = rgb = None
    nodes = sorted(glob.glob("/dev/video*"), key=lambda d: int(re.sub(r"\D", "", d) or 0))
    for dev in nodes:
        card, fmts = device_info(dev)
        if lep is None and offers_y16(fmts):
            lep = dev
        elif rgb is None:
            if rgb_match is not None:
                if rgb_match in card:
                    rgb = dev
            elif "MJPG" in fmts and "640x480" in fmts:
                rgb = dev
    return lep, rgb


class Stream(threading.Thread):
    def __init__(self, name, stop_evt):
        super().__init__(daemon=True)
        self.name_ = name
        self.stop_evt = stop_evt
        self.n = 0
        self.err = None
        self.stamps = []


class LeptonStream(Stream):
    def __init__(self, dev, out, stop_evt):
        super().__init__("lepton", stop_evt)
        self.dev = dev
        self.out = out
        self.frames = []

    def run(self):
        import cv2

        from leptonkit.device import open_lepton

        try:
            cap = open_lepton(self.dev)
        except OSError as e:
            self.err = str(e)
            return
        while not self.stop_evt.is_set():
            ok, f = cap.read()
            t = time.time()
            if not ok:
                continue
            self.frames.append(y16.to_u16(f))
            self.stamps.append(t)
            self.n += 1
        cap.release()
        if self.frames:
            stack = np.stack(self.frames)
            np.save(os.path.join(self.out, "y16_stack.npy"), stack)
            np.save(os.path.join(self.out, "timestamps_unix.npy"), np.array(self.stamps))
            g = agc.heq(stack.mean(0))
            cv2.imwrite(os.path.join(self.out, "preview_whitehot.png"),
                        cv2.resize(g, (640, 480), interpolation=cv2.INTER_NEAREST))


class RgbStream(Stream):
    def __init__(self, dev, out, stop_evt):
        super().__init__("rgb", stop_evt)
        self.dev = dev
        self.out = out

    def run(self):
        import cv2

        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            self.err = f"cannot open {self.dev}"
            return
        wr = cv2.VideoWriter(os.path.join(self.out, "rgb.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
        first = True
        while not self.stop_evt.is_set():
            ok, f = cap.read()
            t = time.time()
            if not ok:
                continue
            if first:
                first = False
                cv2.imwrite(os.path.join(self.out, "rgb_first.jpg"), f)
            wr.write(f)
            self.stamps.append(t)
            self.n += 1
        cap.release()
        wr.release()
        np.save(os.path.join(self.out, "rgb_timestamps_unix.npy"), np.array(self.stamps))


def report(streams, devices, t0, t1) -> dict:
    rep = {}
    for s in streams:
        fps = 0.0
        if s.n > 1 and s.stamps[-1] > s.stamps[0]:
            fps = round((s.n - 1) / (s.stamps[-1] - s.stamps[0]), 2)
        rep[s.name_] = {"frames": s.n, "error": s.err, "fps": fps}
    rep["devices"] = devices
    rep["t_start_unix"] = t0
    rep["t_end_unix"] = t1
    return rep


def record(out, duration=150.0, lepton=True, rgb=True, lepton_dev=None, rgb_dev=None, rgb_match=None) -> dict:
    os.makedirs(out, exist_ok=True)
    found_lep, found_rgb = find_devices(rgb_match)
    lep_dev = lepton_dev or found_lep
    rgb_dev = rgb_dev or found_rgb
    print(f"[streams] lepton={lep_dev} rgb={rgb_dev}", flush=True)
    stop = threading.Event()
    streams = []
    if lepton:
        if lep_dev is None:
            raise RuntimeError("no Lepton (Y16 160x120) device found; refusing to record without thermal")
        streams.append(LeptonStream(lep_dev, out, stop))
    if rgb and rgb_dev:
        streams.append(RgbStream(rgb_dev, out, stop))
    t0 = time.time()
    for s in streams:
        s.start()
    last = -1
    try:
        while time.time() - t0 < duration and not os.path.exists(os.path.join(out, "STOP")):
            time.sleep(0.5)
            el = int(time.time() - t0)
            if el % 10 == 0 and el != last:
                last = el
                print(f"[streams] t={el:5d}s " + " ".join(f"{s.name_}={s.n}" for s in streams), flush=True)
    except KeyboardInterrupt:
        print("[streams] interrupted, saving what was recorded", flush=True)
    stop.set()
    for s in streams:
        s.join()   # the threads write their files after stopping; exiting earlier would cut them off
    rep = report(streams, {"lepton": lep_dev, "rgb": rgb_dev}, t0, time.time())
    with open(os.path.join(out, "streams.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("[streams] done", json.dumps(rep), flush=True)
    return rep
