"""Person-triggered recording.

The trigger looks for new body-temperature pixels compared with about one
second earlier, so a static warm object (a PC, a radiator) never triggers and
only a moving warm blob does. Each event keeps a pre-roll from before the
trigger, ends after a quiet period, and is capped in length; after a capped
event the trigger re-arms only once the scene has been quiet for a while.

MotionGate holds that logic with no camera or file access, so it can be
tested on synthetic frames. run() adds the camera, the event files, a disk
budget and an optional live view in the browser.
"""
from __future__ import annotations

import datetime
import json
import os
import signal
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from leptonkit import agc, y16


@dataclass
class TriggerSettings:
    body_lo_c: float = 28.5      # skin as seen by the Lepton, lower bound
    body_hi_c: float = 38.0      # upper bound
    mask_lag: int = 9            # compare with the mask this many frames ago (about 1 s)
    motion_trigger: int = 40     # new warm pixels that count as motion
    motion_quiet: int = 15       # below this a frame counts as quiet
    trigger_frames: int = 3      # consecutive motion frames to start (rides out FFC glitches)
    quiet_frames: int = 45       # consecutive quiet frames to stop (about 5 s)
    preroll: int = 45            # frames kept from before the trigger (about 5 s)
    max_event: int = 810         # hard cap per event (about 90 s)
    min_event: int = 9           # events with fewer active frames are discarded
    rearm_quiet: int = 45        # quiet frames needed after a capped event
    warmup: int = 30             # frames to settle before arming


@dataclass
class Event:
    frames: list = field(default_factory=list)
    stamps: list = field(default_factory=list)
    hots: list = field(default_factory=list)
    peak_motion: int = 0
    capped: bool = False


class MotionGate:
    def __init__(self, settings: TriggerSettings | None = None):
        self.s = settings or TriggerSettings()
        self.raw_lo = int(y16.celsius_to_raw(self.s.body_lo_c))
        self.raw_hi = int(y16.celsius_to_raw(self.s.body_hi_c))
        self.reset()

    def reset(self) -> None:
        self.ring = deque(maxlen=self.s.preroll)
        self.masks = deque(maxlen=self.s.mask_lag + 1)
        self.warm = 0
        self.consec = 0
        self.quiet = 0
        self.rearm = 0
        self.event: Event | None = None

    @property
    def in_event(self) -> bool:
        return self.event is not None

    def push(self, frame, t: float, can_start: bool = True):
        """Feed one uint16 frame. Returns (status, finished event or None).

        status: hot and motion pixel counts, in_event, and a note
        ('trigger', 'blocked', 'discarded', 'capped' or None)."""
        s = self.s
        a = np.asarray(frame)
        mask = (a >= self.raw_lo) & (a <= self.raw_hi)
        hot = int(mask.sum())
        motion = int((mask & ~self.masks[0]).sum()) if len(self.masks) > s.mask_lag else 0
        self.masks.append(mask)
        self.ring.append((a, t, hot))
        self.warm += 1
        note = None
        done = None
        if self.warm < s.warmup:
            pass
        elif self.event is None:
            if self.rearm > 0:
                self.rearm = self.rearm - 1 if motion < s.motion_quiet else s.rearm_quiet
            else:
                self.consec = self.consec + 1 if motion > s.motion_trigger else 0
                if self.consec >= s.trigger_frames:
                    if not can_start:
                        self.consec = 0
                        note = "blocked"
                    else:
                        self.event = Event(frames=[f for f, _, _ in self.ring],
                                           stamps=[x for _, x, _ in self.ring],
                                           hots=[h for _, _, h in self.ring],
                                           peak_motion=motion)
                        self.quiet = 0
                        note = "trigger"
        else:
            ev = self.event
            ev.frames.append(a)
            ev.stamps.append(t)
            ev.hots.append(hot)
            ev.peak_motion = max(ev.peak_motion, motion)
            self.quiet = self.quiet + 1 if motion < s.motion_quiet else 0
            if self.quiet >= s.quiet_frames or len(ev.frames) >= s.max_event:
                active = len(ev.frames) - self.quiet - s.preroll
                ev.capped = len(ev.frames) >= s.max_event
                if active >= s.min_event:
                    done = ev
                    note = "capped" if ev.capped else None
                else:
                    note = "discarded"
                if ev.capped:
                    self.rearm = s.rearm_quiet
                self.event = None
                self.consec = 0
        status = {"hot": hot, "motion": motion, "in_event": self.in_event, "note": note}
        return status, done

    def flush(self) -> Event | None:
        """End the stream (camera lost): returns the running event if long enough, and resets."""
        ev = self.event
        self.reset()
        if ev is not None and len(ev.frames) >= self.s.min_event:
            return ev
        return None


# ---------------------------------------------------------------- live view

CMAPS = ["inferno", "turbo", "jet", "hot", "plasma", "viridis"]
PALETTES = CMAPS + ["whitehot", "blackhot"]


class LiveView:
    """MJPEG view of the latest frame at http://host:port/ (palette grid or one palette)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = None   # (gray8, tmax_c, hot, motion, in_event)

    def publish(self, frame, in_event, hot, motion) -> None:
        g = y16.stretch_percentile(frame, 2, 98)
        tmax = float(y16.raw_to_celsius(frame.max()))
        with self.lock:
            self.state = (g, tmax, hot, motion, in_event)

    @staticmethod
    def colorize(g, cmap):
        import cv2

        if cmap == "whitehot":
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if cmap == "blackhot":
            return cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(g, getattr(cv2, "COLORMAP_" + cmap.upper(), cv2.COLORMAP_INFERNO))

    def jpeg(self, cmap):
        import cv2

        with self.lock:
            st = self.state
        if st is None:
            return None
        g, tmax, hot, motion, in_event = st
        font = cv2.FONT_HERSHEY_SIMPLEX
        if cmap == "grid":
            tiles = []
            for p in PALETTES:
                t = cv2.resize(self.colorize(g, p), (320, 240), interpolation=cv2.INTER_NEAREST)
                cv2.putText(t, p, (8, 24), font, 0.7, (0, 0, 0), 3)
                cv2.putText(t, p, (8, 24), font, 0.7, (255, 255, 255), 1)
                tiles.append(t)
            img = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)])
        else:
            img = cv2.resize(self.colorize(g, cmap), (640, 480), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        txt = f"max {tmax:.1f}C  hot {hot}  motion {motion}"
        cv2.putText(img, txt, (8, h - 10), font, 0.55, (0, 0, 0), 3)
        cv2.putText(img, txt, (8, h - 10), font, 0.55, (240, 240, 240), 1)
        if in_event:
            cv2.circle(img, (w - 25, 22), 9, (0, 0, 255), -1)
            cv2.putText(img, "REC", (w - 80, 28), font, 0.6, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def page(self) -> bytes:
        buttons = "".join(f"<button onclick=\"s('{p}')\">{p}</button>" for p in ["grid"] + PALETTES)
        names = ",".join(f"'{p}'" for p in PALETTES)
        html = ("<html><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                "<div style='padding:6px'>" + buttons +
                " <span style='opacity:.6'>(click a tile to zoom, click again for the grid)</span></div>"
                "<img id='v' src='/stream?cmap=grid' style='width:100%;max-width:1280px;"
                "image-rendering:pixelated;cursor:pointer'><script>"
                f"var cur='grid',v=document.getElementById('v'),N=[{names}];"
                "function s(c){cur=c;v.src='/stream?cmap='+c}"
                "v.onclick=function(e){if(cur!=='grid'){s('grid');return}"
                "var r=v.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)/r.width*4),"
                "y=Math.floor((e.clientY-r.top)/r.height*2);s(N[Math.min(y*4+x,7)])}"
                "</script></body></html>")
        return html.encode()

    def serve(self, host: str, port: int):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                cmap = "grid"
                if "cmap=" in self.path:
                    c = self.path.split("cmap=")[1].split("&")[0]
                    if c in PALETTES:
                        cmap = c
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            jpg = view.jpeg(cmap)
                            if jpg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                            time.sleep(0.11)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/frame.jpg"):
                    jpg = view.jpeg(cmap)
                    self.send_response(200 if jpg else 503)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    if jpg:
                        self.wfile.write(jpg)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(view.page())

        srv = ThreadingHTTPServer((host, port), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


# ---------------------------------------------------------------- recorder

class Recorder:
    def __init__(self, out_dir, device=None, settings=None, min_free_gb=5.0, max_total_gb=10.0,
                 live_host="127.0.0.1", live_port=8089, fps=9.0):
        self.out = os.path.abspath(out_dir)
        self.events_dir = os.path.join(self.out, "events")
        os.makedirs(self.events_dir, exist_ok=True)
        self.device = device
        self.gate = MotionGate(settings)
        self.min_free_gb = min_free_gb
        self.max_total_gb = max_total_gb
        self.live_host = live_host
        self.live_port = live_port
        self.fps = fps
        self.view = LiveView()
        self.log_f = open(os.path.join(self.out, "recorder.log"), "a", buffering=1)

    def log(self, msg: str) -> None:
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        self.log_f.write(line + "\n")
        print(line, flush=True)

    def budget_ok(self) -> bool:
        st = os.statvfs(self.events_dir)
        free = st.f_bavail * st.f_frsize / 1e9
        total = 0
        for root, _, files in os.walk(self.events_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return free >= self.min_free_gb and total / 1e9 <= self.max_total_gb

    def save(self, ev: Event) -> str:
        from leptonkit.video import write_mp4

        ts = datetime.datetime.fromtimestamp(ev.stamps[0]).strftime("%Y%m%d_%H%M%S")
        d = base = os.path.join(self.events_dir, f"evt_{ts}")
        n = 1
        while True:   # never write into an earlier event (same second, a clock step, a second recorder)
            try:
                os.makedirs(d)
                break
            except FileExistsError:
                n += 1
                d = f"{base}_{n}"
        stack = np.stack(ev.frames).astype(np.uint16)
        np.save(os.path.join(d, "y16_stack.npy"), stack)
        np.save(os.path.join(d, "timestamps_unix.npy"), np.array(ev.stamps))
        try:
            stream = agc.Agc()
            write_mp4(os.path.join(d, "event_whitehot.mp4"), (stream(f) for f in stack), fps=self.fps)
        except Exception as e:  # the raw stack is what matters
            self.log(f"mp4 encode failed: {e}")
        meta = {
            "start": ev.stamps[0], "end": ev.stamps[-1], "n_frames": len(ev.frames),
            "duration_s": round(ev.stamps[-1] - ev.stamps[0], 1),
            "peak_hot_px": int(max(ev.hots)), "peak_motion_px": int(ev.peak_motion),
            "peak_temp_c": round(float(y16.raw_to_celsius(stack.max())), 1),
            "capped": ev.capped, "settings": asdict(self.gate.s),
        }
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(self.out, "events.jsonl"), "a") as f:
            f.write(json.dumps({"dir": d, **{k: v for k, v in meta.items() if k != "settings"}}) + "\n")
        self.log(f"event saved: {d} ({meta['duration_s']} s, peak {meta['peak_temp_c']} C, "
                 f"motion {meta['peak_motion_px']} px)")
        return d

    def open_camera(self):
        from leptonkit.device import find_lepton, open_lepton

        while True:
            dev = self.device or find_lepton()
            if dev:
                try:
                    cap = open_lepton(dev)
                    ok, _ = cap.read()
                    if ok:
                        self.log(f"camera opened: {dev}")
                        return cap
                    cap.release()
                except OSError:
                    pass
            self.log("camera not available, retrying in 5 s")
            time.sleep(5)

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit))
        s = self.gate.s
        self.log(f"recorder start, body band {s.body_lo_c}-{s.body_hi_c} C, "
                 f"trigger {s.motion_trigger} new warm px vs {s.mask_lag} frames ago")
        if self.live_port:
            try:
                self.view.serve(self.live_host, self.live_port)
                self.log(f"live view on http://{self.live_host}:{self.live_port}/")
            except OSError as e:
                self.log(f"live view failed: {e}")
        cap = self.open_camera()
        checked = False
        try:
            while True:
                ok, f = cap.read()
                now = time.time()
                if not ok:
                    self.log("read failed, reconnecting")
                    cap.release()
                    ev = self.gate.flush()
                    if ev is not None:
                        self.save(ev)
                    checked = False
                    cap = self.open_camera()
                    continue
                a = y16.to_u16(f)
                if not checked:
                    checked = True
                    if not y16.looks_radiometric(a):
                        self.log(f"warning: median raw {int(np.median(a))} does not look like centikelvin; "
                                 "the body band assumes radiometric (TLinear) output")
                can_start = True
                if not self.gate.in_event and self.gate.consec + 1 >= s.trigger_frames:
                    can_start = self.budget_ok()
                status, done = self.gate.push(a, now, can_start=can_start)
                self.view.publish(a, status["in_event"], status["hot"], status["motion"])
                note = status["note"]
                if note == "trigger":
                    self.log(f"trigger: {status['motion']} new warm px (warm total {status['hot']})")
                elif note == "blocked":
                    self.log("disk budget reached, event not recorded")
                elif note == "discarded":
                    self.log("short blip discarded")
                elif note == "capped":
                    self.log("event length cap hit, re-arming after the scene clears")
                if done is not None:
                    self.save(done)
        except (KeyboardInterrupt, SystemExit):
            ev = self.gate.flush()
            if ev is not None:
                self.save(ev)
            self.log("recorder stopped")
        finally:
            cap.release()
