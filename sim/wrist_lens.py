"""The real wrist camera's lens, for sim renders.

MuJoCo renders an ideal pinhole. The real wrist camera (calibrated with
calib/calibrate_intrinsics.py -> calib/intrinsics/wrist.json) has the measured
focal lengths and optical centre, which so101_real.xml gives wrist_cam directly,
plus lens distortion (about 12 px at the image's left/right edges), which
MuJoCo can't model. WristLens.distort() remaps a pinhole render of wrist_cam
into what the real camera would see.

    lens = WristLens()                  # loads calib/intrinsics/wrist.json
    img = lens.distort(renderer.render())                     # RGB/BGR, any size with 4:3 aspect
    mask = lens.distort(mask, nearest=True)                   # segmentation / masks

The render must come from a camera with the same intrinsics (wrist_cam in
so101_real.xml). Renders at other 4:3 sizes (e.g. 320x240) are handled by
scaling the maps.

Orientation: the wrist image is used rotated 90 deg anticlockwise (WRIST_ROTATION,
chosen 2026-11-21), jaws on the right: orient_wrist() is applied after distort() in the
dataset export, eval and the teleop display, so a 640x480 render becomes 480 wide x 640
tall. The real wrist camera must be rotated the same way at deployment -- in LeRobot:
OpenCVCameraConfig(..., width=480, height=640, rotation=Cv2Rotation.ROTATE_270) (i.e.
rotation=-90; LeRobot's width/height are after rotation).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = HERE / "calib/intrinsics/wrist.json"
WRIST_ROTATION = "ccw90"          # 90 deg anticlockwise; LeRobot Cv2Rotation.ROTATE_270 (-90)


def orient_wrist(img):
    """Rotate a (distorted) wrist image to the orientation the dataset and policy use."""
    return np.ascontiguousarray(np.rot90(img, 1))           # k=1: anticlockwise


class WristLens:
    def __init__(self, path=CALIB):
        c = json.loads(Path(path).read_text())
        self.calib = c
        self.W, self.H = c["image_size"]
        self.K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
        self.D = np.array(c["distortion"], float)
        self.fisheye = c.get("model") == "fisheye"
        self._maps = {}

    def mujoco_camera_attrs(self):
        """XML attributes that give a MuJoCo camera these intrinsics (pinhole part).
        MuJoCo's principalpixel is the offset of the image centre from the optical centre,
        i.e. the opposite sign of OpenCV's (cx - W/2, cy - H/2) -- verified by projection test."""
        c = self.calib
        return (f'resolution="{self.W} {self.H}" focalpixel="{c["fx"]:.3f} {c["fy"]:.3f}" '
                f'principalpixel="{self.W / 2 - c["cx"]:.3f} {self.H / 2 - c["cy"]:.3f}" '
                f'sensorsize="{self.W * 1e-5:.5f} {self.H * 1e-5:.5f}"')

    def _map(self, w, h):
        import cv2                                          # lazy: orient_wrist() works without OpenCV
        if (w, h) not in self._maps:
            s = w / self.W
            if abs(h / self.H - s) > 1e-6:
                raise ValueError(f"render is {w}x{h}; the lens is calibrated for a {self.W}x{self.H} (4:3) image")
            # for each output (distorted) pixel: where it lies in the ideal pinhole image
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            pts = np.stack([u, v], -1).reshape(-1, 1, 2) / s            # in calibration pixels
            if self.fisheye:
                und = cv2.fisheye.undistortPoints(pts, self.K, self.D.reshape(-1, 1), P=self.K)
            else:
                und = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
            und = und.reshape(h, w, 2) * s
            self._maps[(w, h)] = (und[..., 0].copy(), und[..., 1].copy())
        return self._maps[(w, h)]

    def distort(self, img, nearest=False):
        import cv2
        h, w = img.shape[:2]
        mx, my = self._map(w, h)
        return cv2.remap(img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def coverage(self):
        """Range of ideal-image pixels the distorted frame needs (to check no render margin is needed)."""
        mx, my = self._map(self.W, self.H)
        return float(mx.min()), float(mx.max()), float(my.min()), float(my.max())
