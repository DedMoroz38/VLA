"""Open the rig's cameras by NAME, not by OpenCV index.

On macOS, OpenCV's camera indices change with what is plugged in (with the C920
unplugged, index 0 became the wrist camera). AVFoundation device names don't, so
cameras are opened by (part of) their name through PyAV:

    side   "C920"       Logitech HD Pro Webcam C920
    wrist  "icspring"   icspring camera

    python calib/cameras.py            # list the connected cameras

An int still means an OpenCV index (the old behaviour).
"""
import threading
import time

import numpy as np

W, H = 640, 480
NAMES = {"side": "C920", "wrist": "icspring"}


def list_cameras():
    """AVFoundation video device names, in AVFoundation's order."""
    import av
    import av.logging
    old = av.logging.get_level()
    av.logging.set_level(av.logging.INFO)          # the device list is logged at INFO
    try:
        with av.logging.Capture() as logs:
            try:
                av.open("", format="avfoundation", options={"list_devices": "true"})
            except Exception:
                pass
    finally:
        av.logging.set_level(old)
    names, video = [], False
    for _, _, msg in logs:
        msg = msg.strip()
        if msg.startswith("AVFoundation video devices"):
            video = True
        elif msg.startswith("AVFoundation audio devices"):
            video = False
        elif video and msg.startswith("[") and "]" in msg:
            name = msg.split("]", 1)[1].strip()
            if not name.startswith("Capture screen"):
                names.append(name)
    return names


def find_camera(part):
    names = list_cameras()
    hits = [n for n in names if part.lower() in n.lower()]
    if not hits:
        raise SystemExit(f"no camera matching '{part}' is connected. Connected: {names or 'none'}")
    return hits[0]


def to_640x480(f):
    """The cameras' 640x480 mode is a centre 4:3 crop of the 16:9 frame; do the same if we got 16:9."""
    import cv2
    h, w = f.shape[:2]
    if (w, h) != (W, H):
        cw = h * 4 // 3
        f = cv2.resize(f[:, (w - cw) // 2:(w - cw) // 2 + cw], (W, H), interpolation=cv2.INTER_AREA)
    return f


class Grabber:
    """Background reader that keeps only the newest frame (BGR, 640x480).
    spec: a name fragment ('C920', 'icspring', ...) or an int OpenCV index."""

    def __init__(self, spec):
        self.frame, self.n, self.run = None, 0, True
        self.lock = threading.Lock()
        if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
            import cv2
            self.cap = cv2.VideoCapture(int(spec), cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                raise SystemExit(f"camera {spec} did not open")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            self.name, target = f"OpenCV index {spec}", self._loop_cv
        else:
            import av
            self.name = find_camera(spec)
            for _ in range(100):
                try:
                    self.cap = av.open(self.name, format="avfoundation",
                                       options={"video_size": f"{W}x{H}", "framerate": "30"})
                    break
                except BlockingIOError:
                    time.sleep(0.02)
            else:
                raise SystemExit(f"camera '{self.name}' did not open")
            target = self._loop_av
        threading.Thread(target=target, daemon=True).start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 5:
            time.sleep(0.02)
        if self.frame is None:
            raise SystemExit(f"camera '{self.name}' gave no frames")

    def _put(self, f):
        f = to_640x480(f)
        with self.lock:
            self.frame, self.n = f, self.n + 1

    def _loop_cv(self):
        while self.run:
            ok, f = self.cap.read()
            if ok:
                self._put(f)

    def _loop_av(self):
        s = self.cap.streams.video[0]
        while self.run:
            try:
                for pkt in self.cap.demux(s):
                    for fr in pkt.decode():
                        self._put(fr.to_ndarray(format="bgr24"))
                    break
            except BlockingIOError:
                time.sleep(0.002)

    def get(self):
        with self.lock:
            return self.frame.copy(), self.n

    def close(self):
        self.run = False
        time.sleep(0.1)
        if hasattr(self.cap, "release"):
            self.cap.release()
        else:
            self.cap.close()


if __name__ == "__main__":
    for i, n in enumerate(list_cameras()):
        role = [r for r, part in NAMES.items() if part.lower() in n.lower()]
        print(f"  {n}" + (f"   <- {role[0]} camera" if role else ""))
