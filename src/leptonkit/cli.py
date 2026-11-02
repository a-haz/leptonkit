"""leptonkit command line."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

from leptonkit import __version__, agc, y16


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _chosen_device(arg: str | None) -> str | None:
    """--device, else $LEPTON_DEV, else None (auto-detect)."""
    return arg or os.environ.get("LEPTON_DEV") or None


def _device(arg: str | None) -> str:
    from leptonkit.device import find_lepton

    dev = _chosen_device(arg) or find_lepton()
    if not dev:
        sys.exit("leptonkit: no device offering Y16 at 160x120 found (pass --device)")
    return dev


def cmd_find(args) -> int:
    from leptonkit.device import find_lepton

    dev = find_lepton()
    if not dev:
        print("no Lepton found", file=sys.stderr)
        return 1
    print(dev)
    return 0


def cmd_capture(args) -> int:
    from leptonkit.capture import capture

    out = args.out or _stamp()
    info = capture(_device(args.device), args.frames, out, mp4=not args.no_mp4)
    print(json.dumps(info, indent=1))
    return 0


def cmd_record(args) -> int:
    from leptonkit.trigger import Recorder, TriggerSettings

    settings = TriggerSettings(body_lo_c=args.body_lo, body_hi_c=args.body_hi, motion_trigger=args.motion_px)
    Recorder(args.out, device=_chosen_device(args.device), settings=settings, min_free_gb=args.min_free_gb,
             max_total_gb=args.max_total_gb, live_host=args.live_host, live_port=args.live_port).run()
    return 0


def cmd_pair(args) -> int:
    from leptonkit.pair import pair_session

    rows = pair_session(args.session, rot=args.rot, rgb_rot=args.rgb_rot, stride=args.stride)
    print(f"{len(rows) - 1} pairs -> {Path(args.session) / 'pairs'}")
    return 0


def cmd_streams(args) -> int:
    from leptonkit.streams import record

    try:
        record(args.out, duration=args.duration, lepton=not args.no_lepton, rgb=not args.no_rgb,
               lepton_dev=_chosen_device(args.device), rgb_dev=args.rgb_device, rgb_match=args.rgb_match)
    except RuntimeError as e:
        print(f"leptonkit: {e}", file=sys.stderr)
        return 1
    return 0


def _load_frames(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        stack = np.load(path)
        return stack[None] if stack.ndim == 2 else stack
    return y16.read_y16(path)[None]


def cmd_render(args) -> int:
    import cv2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = agc.Agc()
    n = 0
    for src in args.inputs:
        src = Path(src)
        for i, frame in enumerate(_load_frames(src)):
            gray = stream(frame) if args.agc == "heq" else y16.stretch_percentile(frame, 1, 99)
            if args.palette == "blackhot":
                gray = 255 - gray
            img = gray if args.palette in ("whitehot", "blackhot") else cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            if args.scale > 1:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
            name = f"{src.stem}.png" if src.suffix != ".npy" else f"{src.stem}_{i:05d}.png"
            cv2.imwrite(str(out / name), img)
            n += 1
    print(f"{n} images -> {out}")
    return 0


def cmd_stats(args) -> int:
    frames = _load_frames(Path(args.input))
    c = y16.raw_to_celsius(frames)
    print(json.dumps({
        "frames": int(len(frames)),
        "radiometric": y16.looks_radiometric(frames),
        "celsius_min": round(float(c.min()), 2),
        "celsius_median": round(float(np.median(c)), 2),
        "celsius_max": round(float(c.max()), 2),
    }, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leptonkit", description="FLIR Lepton (PureThermal) Y16 tools")
    p.add_argument("--version", action="version", version=f"leptonkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("find", help="print the V4L2 node of the Lepton")
    s.set_defaults(func=cmd_find)

    s = sub.add_parser("capture", help="capture N raw frames to a folder")
    s.add_argument("--device", help="V4L2 node (default: $LEPTON_DEV or auto-detect)")
    s.add_argument("--frames", type=int, default=135, help="number of frames (default 135, about 15 s)")
    s.add_argument("--out", help="output folder (default: a timestamped folder here)")
    s.add_argument("--no-mp4", action="store_true", help="skip the MP4 preview")
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser("record", help="record events when a person moves through the view")
    s.add_argument("--out", required=True, help="folder for events/, events.jsonl and recorder.log")
    s.add_argument("--device", help="V4L2 node (default: $LEPTON_DEV or auto-detect, re-detected after a replug)")
    s.add_argument("--body-lo", type=float, default=28.5, help="lower bound of the body band, C")
    s.add_argument("--body-hi", type=float, default=38.0, help="upper bound of the body band, C")
    s.add_argument("--motion-px", type=int, default=40, help="new warm pixels that count as motion")
    s.add_argument("--min-free-gb", type=float, default=5.0, help="stop recording below this free space")
    s.add_argument("--max-total-gb", type=float, default=10.0, help="stop recording above this total size")
    s.add_argument("--live-host", default="127.0.0.1", help="address for the live view")
    s.add_argument("--live-port", type=int, default=8089, help="port for the live view, 0 = off")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("pair", help="RGB and thermal side-by-side panels for a ThermalRig session")
    s.add_argument("session")
    s.add_argument("--rot", choices=["none", "ccw90", "180", "cw90"], default="none",
                   help="rotation that makes the thermal image upright")
    s.add_argument("--rgb-rot", choices=["none", "ccw90", "180", "cw90"], default="none",
                   help="rotation that makes the RGB image upright")
    s.add_argument("--stride", type=int, default=9, help="use every Nth thermal frame (9 = about 1 per second)")
    s.set_defaults(func=cmd_pair)

    s = sub.add_parser("streams", help="record the Lepton and a webcam on one clock")
    s.add_argument("out")
    s.add_argument("--duration", type=float, default=150.0, help="seconds (or create OUT/STOP to end early)")
    s.add_argument("--device", help="Lepton V4L2 node (default: $LEPTON_DEV or auto-detect)")
    s.add_argument("--rgb-device", help="webcam V4L2 node (default: auto-detect)")
    s.add_argument("--rgb-match", help="pick the webcam whose v4l2-ctl --all output contains this text")
    s.add_argument("--no-rgb", action="store_true")
    s.add_argument("--no-lepton", action="store_true")
    s.set_defaults(func=cmd_streams)

    s = sub.add_parser("render", help="render .y16 files or a y16_stack.npy to PNG")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--agc", choices=["heq", "linear"], default="heq")
    s.add_argument("--palette", choices=["whitehot", "blackhot", "inferno"], default="whitehot")
    s.add_argument("--scale", type=int, default=1, help="integer upscale factor")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("stats", help="temperature range of a .y16 file or y16_stack.npy")
    s.add_argument("input")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
