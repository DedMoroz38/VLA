"""Grab one 640x480 frame from each real camera, for checking the sim match.

    /opt/anaconda3/bin/python calib/capture_real.py            # -> calib/real/{side,wrist}_now_480.png
    /opt/anaconda3/bin/python calib/capture_real.py --side 0 --wrist 1

Needs OpenCV (the base anaconda python has it; the mujoco-so101 env does not).
OpenCV opens cameras by index, and macOS can renumber them when devices are
plugged in or out. Check the saved frames: on this Mac 0 was the Logitech
C920 (side) and 1 the icspring module (wrist); the FaceTime camera is the
remaining index.
"""
import argparse
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent


def grab(index, warmup=40):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} did not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(warmup):            # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {index} returned no frames")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=0)
    ap.add_argument("--wrist", type=int, default=1)
    ap.add_argument("--suffix", default="now_480")
    args = ap.parse_args()
    (HERE / "real").mkdir(exist_ok=True)
    for name, idx in [("side", args.side), ("wrist", args.wrist)]:
        f = grab(idx)
        out = HERE / "real" / f"{name}_{args.suffix}.png"
        cv2.imwrite(str(out), f)
        print(f"{name}: camera {idx} {f.shape[1]}x{f.shape[0]} -> {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
