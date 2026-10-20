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


def _device(arg: str | None) -> str:
    from leptonkit.device import find_lepton

    dev = arg or os.environ.get("LEPTON_DEV") or find_lepton()
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
