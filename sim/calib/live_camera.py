"""Show a real camera live at 640x480, the resolution the sim cameras render at.

    /opt/anaconda3/bin/python calib/live_camera.py            # side cam (index 0)
    /opt/anaconda3/bin/python calib/live_camera.py --index 1  # wrist cam

Keys: S saves the current frame to calib/real/<name>_live_NNN.png, Q or Esc quits.
"""
import argparse
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="OpenCV camera index (0 = side C920 on this Mac)")
    ap.add_argument("--name", default=None, help="label for the window and saved frames")
    args = ap.parse_args()
    name = args.name or {0: "side", 1: "wrist"}.get(args.index, f"cam{args.index}")

    cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.index} did not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{name}: camera {args.index} streaming at {w}x{h}  (S = save frame, Q = quit)")

    title = f"{name} cam - {w}x{h}"
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out = HERE / "real" / f"{name}_live_{saved:03d}.png"
            cv2.imwrite(str(out), frame)
            print("saved", out.relative_to(HERE.parent))
            saved += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
