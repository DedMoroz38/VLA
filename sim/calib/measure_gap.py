"""Measure the sim-to-real visual gap, factor by factor, on matched real / sim pairs.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/measure_gap.py calib/pairs/20261123_202747

Input: a capture_pairs.py folder (real and sim side + wrist frames at the same measured joints and
cube pose). The sim is re-rendered with segmentation, so every pixel is labelled (floor, table,
container, cube, robot arm, fixed jaw, moving jaw), and each factor is measured on the region where
it means something:

  geometry    pixel offset of each object's outline, real vs sim (chamfer search on real edges):
              side: static scene (= camera pose), arm, cube; wrist: fixed jaw (= camera mount),
              moving jaw, cube
  colour      per region: CIELAB lightness gain and a/b (colour) offset, real minus sim
  contrast    per region: lightness spread ratio
  shadows     table: share of pixels darker than 85% of the table's median, and how dark (p5/p50)
  sharpness   Gaussian blur sigma that makes sim edges as soft as real ones (contrast-normalised
              gradient across object outlines)
  noise       per-pixel noise sigma on flat table (Immerkaer estimator)
  vignetting  wrist: real/sim table lightness vs distance from the image centre

Output, in the pairs folder: gap.json (per pair and summary) and gap.png (overview).
"""
import argparse
import json
import pathlib
import sys

import cv2
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))

import joint_units  # noqa: E402
from pick_place_task import reset_to_ready, set_eraser_pose  # noqa: E402
from run_policy_real import H, W, SimView  # noqa: E402
from wrist_lens import orient_wrist  # noqa: E402

CATS = ["floor", "table", "container", "cube", "arm", "jaw_fixed", "jaw_moving"]
CAMS = {"side": "ext_cam", "wrist": "wrist_cam"}
SIGMAS = np.arange(0, 6.01, 0.25)


# -- labels ---------------------------------------------------------------------------------

def geom_categories(m):
    """geom id -> index into CATS (-1: ignore)."""
    out = np.full(m.ngeom, -1, int)
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        name, root = m.body(b).name, m.body(m.body_rootid[b]).name
        if b == 0:
            c = "floor"
        elif root in ("table", "cube", "container"):
            c = root
        elif name == "gripper":
            c = "jaw_fixed"
        elif name == "moving_jaw_so101_v1":
            c = "jaw_moving"
        elif root == "base":
            c = "arm"
        else:
            continue
        out[g] = CATS.index(c)
    return out


class Labeller:
    def __init__(self, sim):
        self.sim = sim
        self.cat = geom_categories(sim.m)
        self.r = mujoco.Renderer(sim.m, H, W)
        self.r.enable_segmentation_rendering()
        sim._render(sim.d, "ext_cam")                       # makes sim.lens

    def labels(self, q_sim, pose):
        s = self.sim
        reset_to_ready(s.m, s.d)
        s.d.qpos[:6] = q_sim
        set_eraser_pose(s.m, s.d, *pose, object_body=s.obj)
        mujoco.mj_forward(s.m, s.d)
        out = {}
        for tag, cam in CAMS.items():
            self.r.update_scene(s.d, camera=cam)
            seg = self.r.render()
            lab = np.full(seg.shape[:2], -1, np.int16)
            geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
            lab[geom] = self.cat[seg[..., 0][geom]]
            if tag == "wrist":
                lab = orient_wrist(s.lens.distort(lab, nearest=True))
            out[tag] = lab
        return out


# -- measurements ---------------------------------------------------------------------------

def interior(mask, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def outline(mask):
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def shift(mask, dx, dy, sc=1.0, c0=(0.0, 0.0)):
    """Move a mask by a chamfer_fit result: scale sc about c0, then shift by (dx, dy)."""
    M = np.float32([[sc, 0, (1 - sc) * c0[0] + dx], [0, sc, (1 - sc) * c0[1] + dy]])
    return cv2.warpAffine(mask.astype(np.uint8), M, mask.shape[::-1], flags=cv2.INTER_NEAREST).astype(bool)


def real_edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    e = cv2.Canny(g, 30, 90)
    return cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)


def chamfer_fit(dist, pts, rng, scales=(1.0,), trunc=12.0):
    """Shift (dx, dy) and scale (about the outline's centroid) of sim outline points that best
    land them on real edges. Returns (dx, dy, scale, mean residual px, centroid) or None when the
    outline is too small to trust."""
    if len(pts) < 40:
        return None
    pts = pts.astype(np.float32)
    c0 = pts.mean(0)
    hh, ww = dist.shape

    def cost(dx, dy, sc):
        q = np.round((pts - c0) * sc + c0 + (dx, dy)).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < ww) & (q[:, 1] >= 0) & (q[:, 1] < hh)
        if ok.mean() < 0.7:
            return np.inf
        return np.minimum(dist[q[ok, 1], q[ok, 0]], trunc).mean()

    best = (0, 0, 1.0, np.inf)
    for sc in scales:
        for dy in range(-rng, rng + 1, 4):
            for dx in range(-rng, rng + 1, 4):
                c = cost(dx, dy, sc)
                if c < best[3]:
                    best = (dx, dy, sc, c)
    if not np.isfinite(best[3]):
        return None
    ds = (scales[1] - scales[0]) / 2 if len(scales) > 1 else 0
    for sc in sorted({best[2] - ds, best[2], best[2] + ds}):
        for dy in range(best[1] - 4, best[1] + 5):
            for dx in range(best[0] - 4, best[0] + 5):
                c = cost(dx, dy, sc)
                if c < best[3]:
                    best = (dx, dy, sc, c)
    return int(best[0]), int(best[1]), float(best[2]), float(best[3]), (float(c0[0]), float(c0[1]))


def lab(img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return x[..., 0] * 100 / 255, x[..., 1] - 128, x[..., 2] - 128


def region_colour(real, sim, m):
    if m.sum() < 200:
        return None
    (Lr, ar, br), (Ls, as_, bs) = lab(real), lab(sim)
    return {"L_real": float(np.median(Lr[m])), "L_sim": float(np.median(Ls[m])),
            "L_gain": float(np.median(Lr[m]) / max(np.median(Ls[m]), 1)),
            "a_off": float(np.median(ar[m]) - np.median(as_[m])),
            "b_off": float(np.median(br[m]) - np.median(bs[m])),
            "contrast_ratio": float((np.percentile(Lr[m], 90) - np.percentile(Lr[m], 10) + 1)
                                    / (np.percentile(Ls[m], 90) - np.percentile(Ls[m], 10) + 1)),
            "n_px": int(m.sum())}


SHADOW_LEVELS = (0.95, 0.90, 0.85, 0.80)


def shadows(img, m):
    """Share of table pixels darker than x * the lit table (its 80th-percentile lightness), and the
    darkest 2% relative to it."""
    if m.sum() < 500:
        return None
    L = lab(img)[0][m]
    lit = np.percentile(L, 80)
    out = {f"below_{int(x * 100)}": float((L < x * lit).mean()) for x in SHADOW_LEVELS}
    out["p2_over_lit"] = float(np.percentile(L, 2) / lit)
    return out


def noise_sigma(img, m):
    """Immerkaer (1996) noise estimate on the grey image, over the flat pixels of mask m."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, k))
    gm = cv2.GaussianBlur(g, (0, 0), 2)
    flat = m & (np.hypot(*np.gradient(gm)) < 0.6)
    if flat.sum() < 500:
        return None
    return float(np.sqrt(np.pi / 2) / 6 * r[flat].mean())


def edge_sharpness(img, band):
    """Steepest edges in `band` (top 2% of gradient magnitude) over the band's contrast. The band
    is wide enough to hold each image's own edges, so it does not depend on exact alignment."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    v = g[band]
    contrast = np.percentile(v, 98) - np.percentile(v, 2)
    return float(np.percentile(np.hypot(gx, gy)[band], 98) / max(contrast, 1.0))


def blur_match(real, sim, band):
    """Gaussian sigma (px) at which sim edges in `band` are as soft as the real ones."""
    if band.sum() < 150:
        return None
    target = edge_sharpness(real, band)
    s = np.array([edge_sharpness(sim if sig == 0 else cv2.GaussianBlur(sim, (0, 0), sig), band)
                  for sig in SIGMAS])
    if target >= s[0]:
        return 0.0                                          # real is as sharp as sim or sharper
    if target <= s[-1]:
        return float(SIGMAS[-1])
    i = int(np.argmax(s <= target))
    return float(np.interp(target, [s[i], s[i - 1]], [SIGMAS[i], SIGMAS[i - 1]]))


def vignetting(real, sim, m, nbins=6):
    hh, ww = m.shape
    yy, xx = np.mgrid[:hh, :ww]
    r = np.hypot(xx - ww / 2, yy - hh / 2) / np.hypot(ww / 2, hh / 2)
    ratio = lab(real)[0] / np.maximum(lab(sim)[0], 1)
    out = []
    for i in range(nbins):
        sel = m & (r >= i / nbins) & (r < (i + 1) / nbins)
        out.append(float(np.median(ratio[sel])) if sel.sum() > 300 else None)
    return out


# -- per pair -------------------------------------------------------------------------------

def measure_pair(e, folder, lbl, jmap):
    q_sim = jmap.real_to_sim(np.array(e["joints_real"], np.float32))
    labels = lbl.labels(q_sim, tuple(e["pose"]))
    res = {"n": e["n"], "pose_id": e["pose_id"], "dist_mm": e["gripper_to_cube_mm"]}
    for tag in CAMS:
        real = cv2.imread(str(folder / e["files"][f"real_{tag}"]))
        sim = cv2.imread(str(folder / e["files"][f"sim_{tag}"]))
        L = labels[tag]
        dist = real_edges(real)
        r = {"geometry": {}, "colour": {}}
        objs = {"side": {"static": [0, 1, 2], "arm": [4, 5, 6], "cube": [3]},
                "wrist": {"jaw_fixed": [5], "jaw_moving": [6], "cube": [3]}}[tag]
        offsets = {}
        for name, cats in objs.items():
            if name == "static":                            # tile, tape, table-edge lines of the sim image
                g = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 40, 120) > 0
                pts = np.argwhere(g & np.isin(L, cats) & ~cv2.dilate((L >= 3).astype(np.uint8),
                                                                      np.ones((9, 9), np.uint8)).astype(bool))
            else:
                pts = np.argwhere(outline(np.isin(L, cats)))
            scales = (1.0,) if tag == "side" else tuple(np.round(np.arange(0.8, 1.51, 0.05), 3))
            o = chamfer_fit(dist, pts[:, ::-1], 24 if tag == "side" else 48, scales)
            r["geometry"][name] = None if o is None else {"dx": o[0], "dy": o[1], "scale": round(o[2], 3),
                                                          "resid_px": round(o[3], 2)}
            offsets[name] = o
        # colour / contrast per region (interior, shifted onto the real object when it was located)
        for c, name in enumerate(CATS):
            m = L == c
            o = offsets.get(name) or offsets.get("arm" if name in ("jaw_fixed", "jaw_moving") and tag == "side"
                                                 else "static" if name in ("floor", "table", "container") else "")
            m_real = shift(m, o[0], o[1], o[2], o[4]) if o else m
            both = interior(m, 6) & interior(m_real, 6)     # agree in sim and on the (shifted) real one
            r["colour"][name] = region_colour(real, sim, both)
        table = interior(L == 1, 8)
        r["shadows"] = {"real": shadows(real, table), "sim": shadows(sim, table)}
        r["noise"] = {"real": noise_sigma(real, table), "sim": noise_sigma(sim, table)}
        # cube and arm/jaw outlines (high-contrast edges), widened to hold the real ones too
        k = np.ones((2 * (6 if tag == "side" else 30) + 1,) * 2, np.uint8)
        band = cv2.dilate(outline(np.isin(L, [3, 4, 5, 6])).astype(np.uint8), k).astype(bool)
        r["blur_sigma"] = blur_match(real, sim, band)
        if tag == "side":                                   # also the dark cube's own outline (its edges
            cb = cv2.dilate(outline(L == 3).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            r["blur_sigma_cube"] = blur_match(real, sim, cb)  # are the policy's cue)
        if tag == "wrist":
            r["vignetting"] = vignetting(real, sim, interior(L == 1, 4))
        res[tag] = r
    return res, labels


# -- summary and figure ---------------------------------------------------------------------

def q(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None
    return {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
            "p90": round(float(np.percentile(v, 90)), 3), "n": int(len(v))}


def summarise(rows):
    S = {}
    for tag in CAMS:
        s = {"geometry": {}, "colour": {}}
        for name in rows[0][tag]["geometry"]:
            g = [r[tag]["geometry"][name] for r in rows]
            s["geometry"][name] = {k: q([x and x[k] for x in g]) for k in ("dx", "dy", "scale", "resid_px")}
        for name in CATS:
            c = [r[tag]["colour"][name] for r in rows]
            if any(c):
                s["colour"][name] = {k: q([x and x[k] for x in c])
                                     for k in ("L_real", "L_sim", "L_gain", "a_off", "b_off", "contrast_ratio")}
        keys = [f"below_{int(x * 100)}" for x in SHADOW_LEVELS] + ["p2_over_lit"]
        s["shadows"] = {src: {k: q([r[tag]["shadows"][src] and r[tag]["shadows"][src][k] for r in rows])
                              for k in keys} for src in ("real", "sim")}
        s["noise"] = {src: q([r[tag]["noise"][src] for r in rows]) for src in ("real", "sim")}
        s["blur_sigma"] = q([r[tag]["blur_sigma"] for r in rows])
        if tag == "side":
            s["blur_sigma_cube"] = q([r[tag]["blur_sigma_cube"] for r in rows])
        if tag == "wrist":
            v = np.array([[np.nan if x is None else x for x in r[tag]["vignetting"]] for r in rows], float)
            s["vignetting_by_radius"] = [None if np.isnan(v[:, i]).all() else round(float(np.nanmedian(v[:, i])), 3)
                                         for i in range(v.shape[1])]
        S[tag] = s
    return S


def overlay(img, L, cats=(3, 4, 5, 6), col=(0, 0, 255)):
    out = img.copy()
    out[outline(np.isin(L, cats))] = col
    return out


def figure(folder, rows, labels_by_n, pick):
    tiles = []
    for n in pick:
        e = next(r for r in rows if r["n"] == n)
        row = []
        for tag in CAMS:
            files = {s: cv2.imread(str(folder / f"{n:03d}_{s}_{tag}.png")) for s in ("real", "sim")}
            L = labels_by_n[n][tag]
            real = overlay(files["real"], L)
            sim = overlay(files["sim"], L)
            if tag == "wrist":
                real, sim = (cv2.resize(x, (360, 480)) for x in (real, sim))
            g = e[tag]["geometry"]
            key = "cube"
            txt = f"{tag} n{n} cube off " + (f"{g[key]['dx']:+d},{g[key]['dy']:+d}px" if g.get(key) else "n/a")
            cv2.putText(real, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(real, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            row += [real, sim]
        tiles.append(np.hstack(row))
    cv2.imwrite(str(folder / "gap.png"), cv2.resize(np.vstack(tiles), None, fx=0.5, fy=0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs")
    a = ap.parse_args()
    folder = (SIM / a.pairs) if not pathlib.Path(a.pairs).is_absolute() else pathlib.Path(a.pairs)
    index = json.loads((folder / "pairs.json").read_text())
    sim = SimView(SIM / index["scene"])
    lbl = Labeller(sim)
    jmap = joint_units.JointMap.load()
    rows, labels_by_n = [], {}
    for e in index["pairs"]:
        r, L = measure_pair(e, folder, lbl, jmap)
        rows.append(r)
        labels_by_n[e["n"]] = L
        g = r["wrist"]["geometry"].get("cube")
        print(f"pair {e['n']:3d} pose {e['pose_id']}: side blur {r['side']['blur_sigma']}, wrist blur "
              f"{r['wrist']['blur_sigma']}, wrist cube offset {g and (g['dx'], g['dy'])}", flush=True)
    summary = summarise(rows)
    (folder / "gap.json").write_text(json.dumps({"summary": summary, "pairs": rows}, indent=1))
    close = sorted(rows, key=lambda r: r["dist_mm"])
    figure(folder, rows, labels_by_n, [close[0]["n"], close[len(close) // 3]["n"], close[-1]["n"]])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
