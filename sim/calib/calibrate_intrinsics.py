"""Checkerboard calibration of a real camera's intrinsics and lens distortion, at the
640x480 mode the sim and the policy use.

    cd sim
    PY=/opt/anaconda3/envs/lerobot/bin/python
    $PY calib/calibrate_intrinsics.py board                 # 1. print calib/intrinsics/checkerboard_A4.pdf
    $PY calib/calibrate_intrinsics.py capture --cam wrist   # 2. collect ~20 views (live window)
    $PY calib/calibrate_intrinsics.py calibrate --cam wrist # 3. solve -> calib/intrinsics/wrist.json

    (--cam side for the C920; --model fisheye if the standard model fits a very wide lens badly)

1. board: an A4 landscape PDF, 10 x 7 squares of 25 mm (9 x 6 inner corners).
   Print at 100% / "actual size" (no "fit to page"). Measure one square with a
   ruler: if it isn't 25.0 mm, pass the real size with --square-mm when
   calibrating. Tape it flat on something stiff (a board, a book cover): a
   wavy page makes the result worse.

2. capture: live view with the board's corners drawn when found. A grid shows
   where corners have been seen so far, so you can tell which parts of the
   image still need the board. Distortion is strongest near the edges and
   corners, so those matter most.
     SPACE  save this view (only when the board is found)
     A      auto-capture on/off: saves by itself when the board is found, held
            still for ~0.5 s, and clearly different from the views already saved
     D      delete the last saved view          Q / Esc  quit
   Aim for 15-25 views: board near and far, tilted up to ~45 deg in different
   directions, and pushed into every corner and edge of the image. The wrist
   camera moves with the arm: hold the board in front of it.

3. calibrate: OpenCV calibrateCamera on the saved views (sub-pixel corners). Views
   with a reprojection error far above the rest are dropped and it's re-solved.
   Writes <cam>.json (fx, fy, cx, cy, distortion, RMS, fovy for the sim) and
   <cam>_undistort.png (a saved view next to its undistorted version: the
   board's lines should come out straight). Good: RMS below ~0.5 px.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")   # before cv2 loads: avoids noisy OpenCL cache errors
import cv2  # noqa: E402
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "intrinsics"
W, H = 640, 480
COLS, ROWS = 9, 6                 # inner corners
SQUARE_MM = 25.0
CELL = 40                         # coverage grid cell, px


# ---------------------------------------------------------------- 1. board
def make_board(path, square_mm=SQUARE_MM):
    from PIL import Image
    dpi = 300
    mm = dpi / 25.4
    page_w, page_h = int(297 * mm), int(210 * mm)
    sq = square_mm * mm
    nx, ny = COLS + 1, ROWS + 1
    x0 = (page_w - nx * sq) / 2
    y0 = (page_h - ny * sq) / 2
    xs, ys = (np.arange(page_w) - x0) / sq, (np.arange(page_h) - y0) / sq
    inside = (xs[None, :] >= 0) & (xs[None, :] < nx) & (ys[:, None] >= 0) & (ys[:, None] < ny)
    black = inside & ((np.floor(xs)[None, :] + np.floor(ys)[:, None]) % 2 == 0)
    img = Image.fromarray(np.where(black, 0, 255).astype(np.uint8), "L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PDF", resolution=dpi)
    print(f"wrote {path}: A4 landscape, {nx} x {ny} squares of {square_mm} mm ({COLS} x {ROWS} inner corners).")
    print("Print at 100% / actual size, then check one square with a ruler.")


# ---------------------------------------------------------------- detection
def find_corners(gray, fast=False):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | (0 if fast else cv2.CALIB_CB_ACCURACY)
    ok, c = cv2.findChessboardCornersSB(gray, (COLS, ROWS), flags)
    return c.reshape(-1, 2) if ok else None


def view_signature(c):
    """Where the board is, how big, and how it's tilted: to tell views apart."""
    ctr = c.mean(0)
    span = np.ptp(c, 0)
    a, b = c[0], c[COLS - 1]
    d = c[-1]
    return np.r_[ctr / [W, H], span / [W, H], np.arctan2(*(b - a)[::-1]) / np.pi,
                 np.linalg.norm(b - a) / max(np.linalg.norm(d - b), 1) / 4]


# ---------------------------------------------------------------- 2. capture
def capture(cam_role, camera_spec):
    sys.path.insert(0, str(HERE))
    from cameras import NAMES, Grabber
    folder = OUT / cam_role / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    g = Grabber(camera_spec or NAMES[cam_role])
    print("camera:", g.name, "| saving to", folder)
    saved = sorted(folder.glob("*.png"))
    cover = np.zeros((H // CELL + 1, W // CELL + 1), int)
    sigs = []
    for p in saved:                                   # resume: count what's already there
        c = find_corners(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), fast=True)
        if c is not None:
            sigs.append(view_signature(c))
            for x, y in c:
                cover[int(y) // CELL, int(x) // CELL] += 1
    auto, still_since, last_c, msg, msg_t = False, None, None, "", 0
    win = f"intrinsics - {cam_role} camera ({g.name})"
    try:
        while True:
            frame, _ = g.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            c = find_corners(gray, fast=True)
            view = frame.copy()
            # coverage grid: green where corners have been seen, more = brighter
            ov = view.copy()
            for (r, k), n in np.ndenumerate(cover):
                if n:
                    cv2.rectangle(ov, (int(k * CELL), int(r * CELL)), (int(k * CELL + CELL - 1), int(r * CELL + CELL - 1)),
                                  (0, int(min(255, 80 + 25 * int(n))), 0), -1)
            view = cv2.addWeighted(ov, 0.35, view, 0.65, 0)
            now = time.time()
            if c is not None:
                cv2.drawChessboardCorners(view, (COLS, ROWS), c.reshape(-1, 1, 2).astype(np.float32), True)
                moving = last_c is None or np.abs(c - last_c).max() > 2.0
                still_since = now if moving or still_since is None else still_since
                last_c = c
            else:
                still_since, last_c = None, None
            n_cells = (cover > 0).sum()
            cv2.putText(view, f"saved {len(sigs)}  coverage {100 * n_cells / cover.size:.0f}%  auto {'ON' if auto else 'off'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(view, "SPACE save  A auto  D delete last  Q quit", (8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if msg and now - msg_t < 2:
                cv2.putText(view, msg, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            want = k == ord(" ")
            if auto and c is not None and still_since is not None and now - still_since > 0.5:
                s = view_signature(c)
                if not sigs or min(np.linalg.norm(np.array(sigs) - s, axis=1)) > 0.12:
                    want = True
            if want:
                if c is None:
                    msg, msg_t = "board not found - not saved", now
                else:
                    idx = int(saved[-1].stem) + 1 if saved else 0
                    p = folder / f"{idx:03d}.png"
                    cv2.imwrite(str(p), frame)
                    saved.append(p)
                    sigs.append(view_signature(c))
                    for x, y in c:
                        cover[int(y) // CELL, int(x) // CELL] += 1
                    msg, msg_t = f"saved {p.name}", now
                    still_since = None
            elif k in (ord("a"), ord("A")):
                auto = not auto
            elif k in (ord("d"), ord("D")) and saved:
                p = saved.pop()
                c_old = find_corners(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), fast=True)
                p.unlink()
                if c_old is not None:
                    sigs.pop()
                    for x, y in c_old:
                        cover[int(y) // CELL, int(x) // CELL] -= 1
                msg, msg_t = f"deleted {p.name}", now
            elif k in (ord("q"), 27):
                break
    finally:
        g.close()
        cv2.destroyAllWindows()
    print(f"{len(saved)} views in {folder}")


# ---------------------------------------------------------------- 3. calibrate
def calibrate(cam_role, square_mm, model, frames=None):
    folder = Path(frames) if frames else OUT / cam_role / "frames"
    paths = sorted(folder.glob("[0-9]*.png"))              # saved views are numbered: 000.png ...
    obj = np.zeros((ROWS * COLS, 3), np.float32)
    obj[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * square_mm / 1000.0
    imgpts, objpts, used = [], [], []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g.shape != (H, W):
            raise SystemExit(f"{p.name} is {g.shape[1]}x{g.shape[0]}, expected {W}x{H}")
        c = find_corners(g)
        if c is None:
            print(f"  {p.name}: board not found, skipped")
            continue
        imgpts.append(c.astype(np.float32).reshape(-1, 1, 2))
        objpts.append(obj.reshape(-1, 1, 3))
        used.append(p)
    if len(used) < 8:
        raise SystemExit(f"only {len(used)} usable views; capture at least ~15")

    def solve(ip, op):
        if model == "fisheye":
            K, D = np.eye(3), np.zeros((4, 1))
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
            rms, K, D, rv, tv = cv2.fisheye.calibrate(op, ip, (W, H), K, D, flags=flags,
                                                       criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9))
            proj = lambda i: cv2.fisheye.projectPoints(op[i], rv[i], tv[i], K, D)[0]
        else:
            rms, K, D, rv, tv = cv2.calibrateCamera(op, ip, (W, H), None, None)
            proj = lambda i: cv2.projectPoints(op[i], rv[i], tv[i], K, D)[0]
        per = np.array([np.sqrt(np.mean(np.sum((proj(i).reshape(-1, 2) - ip[i].reshape(-1, 2)) ** 2, 1)))
                        for i in range(len(ip))])
        return rms, K, D, per

    rms, K, D, per = solve(imgpts, objpts)
    keep = per < max(3 * np.median(per), 0.3)
    if (~keep).any():
        print("dropped (error far above the rest):", [used[i].name for i in np.flatnonzero(~keep)])
        imgpts = [x for x, k in zip(imgpts, keep) if k]
        objpts = [x for x, k in zip(objpts, keep) if k]
        used = [x for x, k in zip(used, keep) if k]
        rms, K, D, per = solve(imgpts, objpts)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    allc = np.concatenate([x.reshape(-1, 2) for x in imgpts])
    cov = np.zeros((H // CELL + 1, W // CELL + 1), bool)
    cov[(allc[:, 1] // CELL).astype(int), (allc[:, 0] // CELL).astype(int)] = True
    res = dict(camera=cam_role, model=model, image_size=[W, H], views=len(used), rms_px=round(float(rms), 4),
               fx=round(float(fx), 3), fy=round(float(fy), 3), cx=round(float(cx), 3), cy=round(float(cy), 3),
               distortion=[round(float(v), 6) for v in np.ravel(D)],
               distortion_names=(["k1", "k2", "k3", "k4"] if model == "fisheye" else ["k1", "k2", "p1", "p2", "k3"]),
               fovy_deg=round(float(np.degrees(2 * np.arctan(H / 2 / fy))), 3),
               fovx_deg=round(float(np.degrees(2 * np.arctan(W / 2 / fx))), 3),
               principal_offset_px=[round(float(cx - W / 2), 2), round(float(cy - H / 2), 2)],
               corner_coverage_pct=round(100 * float(cov.mean()), 1),
               per_view_rms_px={p.name: round(float(e), 3) for p, e in zip(used, per)},
               square_mm=square_mm, date=time.strftime("%Y-%m-%d %H:%M:%S"))
    OUT.mkdir(exist_ok=True)
    (OUT / f"{cam_role}.json").write_text(json.dumps(res, indent=2))

    # preview: the view whose corners reach furthest out, raw vs undistorted (same K: lines go straight)
    far = int(np.argmax([np.linalg.norm(x.reshape(-1, 2) - [W / 2, H / 2], axis=1).max() for x in imgpts]))
    img = cv2.imread(str(used[far]))
    if model == "fisheye":
        und = cv2.fisheye.undistortImage(img, K, D, Knew=K)
    else:
        und = cv2.undistort(img, K, D)
    cv2.imwrite(str(OUT / f"{cam_role}_undistort.png"), np.hstack([img, und]))

    print(f"\n{cam_role} camera, {model} model, {len(used)} views, corners cover {res['corner_coverage_pct']}% of the image")
    print(f"  RMS reprojection error {rms:.3f} px   ({'good' if rms < 0.5 else 'ok' if rms < 1.0 else 'poor: recapture'})")
    print(f"  fx {fx:.1f}  fy {fy:.1f}  ->  fovy {res['fovy_deg']:.2f} deg, fovx {res['fovx_deg']:.2f} deg")
    print(f"  optical centre ({cx:.1f}, {cy:.1f})  = image centre + ({cx - W / 2:+.1f}, {cy - H / 2:+.1f}) px")
    print(f"  distortion {dict(zip(res['distortion_names'], res['distortion']))}")
    print(f"wrote {OUT / (cam_role + '.json')} and {cam_role}_undistort.png")
    if res["corner_coverage_pct"] < 60:
        print("  note: corners cover little of the image; add views with the board near the edges/corners")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("board", "capture", "calibrate"))
    ap.add_argument("--cam", choices=("wrist", "side"), default="wrist")
    ap.add_argument("--camera", default=None, help="camera name fragment or OpenCV index (default by --cam)")
    ap.add_argument("--square-mm", type=float, default=SQUARE_MM, help="measured printed square size")
    ap.add_argument("--model", choices=("standard", "fisheye"), default="standard")
    ap.add_argument("--frames", default=None, help="(calibrate) folder of views instead of intrinsics/<cam>/frames")
    args = ap.parse_args()
    if args.step == "board":
        make_board(OUT / "checkerboard_A4.pdf", args.square_mm)
    elif args.step == "capture":
        capture(args.cam, args.camera)
    else:
        calibrate(args.cam, args.square_mm, args.model, args.frames)


if __name__ == "__main__":
    main()
