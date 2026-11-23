"""Build a textured copy of the real lab floor for the cube scene.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python calib/floor/build_floor.py              # build + write the scene
    /opt/anaconda3/envs/lerobot/bin/python calib/floor/build_floor.py --no-write   # build + check only

Inputs: calib/floor/side_1080p_now.png (a 1080p frame from the calibrated side
camera, C920) and ../photos/floor.jpeg (a phone photo of the same floor from
above the robot).

1. C920 -> floor: every side-camera pixel above the table's back edge is traced
   to the floor plane (z = FLOOR_Z, 72 cm below the table top) with the tuned
   ext_cam, giving a metric top-down image of the strip the camera sees.
2. Photo -> floor: SIFT matches between the photo and that strip, RANSAC
   homography (the floor is a plane), so the whole photo is warped onto the
   same metric grid. Non-floor parts of the photo are masked by hand-drawn
   polygons (table, robot and cables, a stool, a chair leg).
3. Colours: a robust 3x3 + bias transform maps the photo's colours to the
   C920's, fitted where both see the floor.
4. Texture = colour-corrected photo, plus the C920's own pixels where only it
   sees the floor (feathered seam); unseen floor is filled with real 1 m blocks
   copied from the seen area, aligned to the measured 0.25 m tile grid.
5. Lighting: render the sim side camera, compare its floor with the real frame
   and scale the texture per channel so they match.
6. Scene: a thin visual-only box carries the texture at z = FLOOR_Z; the
   physical floor plane moves to FLOOR_Z with the mean floor colour, and the
   table legs are shortened to reach it. Backup: *.xml.before_floor.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import floorlib as F  # noqa: E402

SIM = HERE.parent.parent
ASSETS = F.SCENE.parent / "assets"
PHOTO = SIM.parent / "photos/floor.jpeg"
FRAME = HERE / "side_1080p_now.png"
TEX_NAME = "floor_real.png"

# non-floor parts of the photo, in its 1200 x 900 preview coordinates (x right, y down)
PHOTO_NONFLOOR = [
    [(0, 718), (1200, 730), (1200, 900), (0, 900)],                               # table, bin, foreground
    [(560, 650), (760, 650), (790, 700), (1065, 760), (1065, 790), (560, 790)],   # robot base + cables
    [(70, 0), (290, 0), (290, 135), (70, 135)],                                   # white stool
    [(0, 0), (70, 0), (70, 90), (0, 90)],                                         # chair leg
]


def log(*a):
    print(*a, flush=True)


def c920_ortho(img):
    H, W = img.shape[:2]
    X, Y = F.grid()
    u, v, front = F.project(np.stack([X, Y, np.full_like(X, F.FLOOR_Z)], -1), W, H)
    ok = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui, vi = np.clip(u.round().astype(int), 0, W - 1), np.clip(v.round().astype(int), 0, H - 1)
    ok &= F.floor_mask_in_image(W, H)[vi, ui]
    o = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    o[~ok] = 0
    return o, ok


def register_photo(photo, ortho, ok):
    s = 2000 / max(photo.shape[:2])
    small = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cl = cv2.createCLAHE(3.0, (8, 8))
    g1 = cl.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    g2 = cl.apply(cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY))
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, cv2.erode(ok.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)))
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.8 * b.distance]
    p1 = np.float32([k1[a.queryIdx].pt for a in good])
    p2 = np.float32([k2[a.trainIdx].pt for a in good])
    Hs, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 6.0, maxIters=20000, confidence=0.999)
    err = np.linalg.norm(cv2.perspectiveTransform(p1[inl.ravel() > 0][None], Hs)[0] - p2[inl.ravel() > 0], axis=1)
    log(f"photo registration: {len(good)} matches, {int(inl.sum())} inliers, "
        f"reprojection {np.median(err) * F.RES * 1000:.1f} mm median")
    return Hs @ np.diag([s, s, 1.0])


def warp_photo(photo, Hfull, shape):
    Hp, Wp = photo.shape[:2]
    k = Wp / 1200.0
    bad = np.zeros((Hp, Wp), np.uint8)
    for pl in PHOTO_NONFLOOR:
        cv2.fillPoly(bad, [np.int32(np.array(pl) * k)], 255)
    tex = cv2.warpPerspective(photo, Hfull, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    cov = cv2.warpPerspective(255 - bad, Hfull, (shape[1], shape[0]), flags=cv2.INTER_NEAREST) > 0
    cov = cv2.erode(cov.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    tex[~cov] = 0
    return tex, cov


def colour_fit(P, pv, C, cv):
    both = cv2.erode((pv & cv).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    Pb, Cb = cv2.GaussianBlur(P, (0, 0), 6), cv2.GaussianBlur(C, (0, 0), 6)
    idx = np.random.default_rng(0).choice(np.flatnonzero(both.ravel()), 60000, replace=False)
    x, y = Pb.reshape(-1, 3)[idx], Cb.reshape(-1, 3)[idx]
    A = np.c_[x, np.ones(len(x))]
    w = np.ones(len(x))
    for _ in range(8):
        M = np.linalg.lstsq(A * w[:, None], y * w[:, None], rcond=None)[0]
        r = np.linalg.norm(A @ M - y, axis=1)
        sc = 1.4826 * np.median(r) + 1e-6
        w = np.where(r < 2.5 * sc, 1.0, 2.5 * sc / r)
    log(f"colour transform photo -> C920: median residual {np.median(np.linalg.norm(A @ M - y, axis=1)):.1f} grey levels")
    return M


def apply_colour(P, M):
    out = np.c_[P.reshape(-1, 3), np.ones(P.shape[0] * P.shape[1])] @ M
    return np.clip(out, 0, 255).reshape(P.shape)


def tile_phase(img, valid, axis, period):
    """Offset (px) of the tile boundaries along one texture axis: strongest periodic edge response."""
    g = cv2.GaussianBlur(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
    d = np.abs(np.diff(g, axis=axis))
    vm = valid[1:, :] & valid[:-1, :] if axis == 0 else valid[:, 1:] & valid[:, :-1]
    prof = np.where(vm, d, 0).sum(1 - axis) / np.maximum(vm.sum(1 - axis), 1)
    scores = [prof[p::period].sum() for p in range(period)]
    return int(np.argmax(scores)) + 1


def fill_blocks(tex, known, block, phase_i, phase_j, period, seed=0):
    """Fill unknown pixels block by block with real blocks copied from the fully known area.
    Targets sit on a block grid; sources may start on any tile line (multiples of `period`),
    so copied tiles stay aligned with the real ones."""
    Hh, Ww = known.shape
    rows = list(range(phase_i - block, Hh, block))
    cols = list(range(phase_j - block, Ww, block))
    src = [(i, j) for i in range(phase_i % period, Hh - block + 1, period)
           for j in range(phase_j % period, Ww - block + 1, period)
           if known[i:i + block, j:j + block].all()]
    log(f"fill: {len(src)} fully seen {block * F.RES:.2f} m blocks to copy from")
    if not src:
        return tex
    rng = np.random.default_rng(seed)
    out, last = tex.copy(), {}
    for i in rows:
        for j in cols:
            i0, j0, i1, j1 = max(i, 0), max(j, 0), min(i + block, Hh), min(j + block, Ww)
            hole = ~known[i0:i1, j0:j1]
            if not hole.any():
                continue
            choices = [s for s in src if s != last.get((i - block, j)) and s != last.get((i, j - block))] or src
            si, sj = choices[rng.integers(len(choices))]
            last[(i, j)] = (si, sj)
            patch = tex[si + (i0 - i):si + (i1 - i), sj + (j0 - j):sj + (j1 - j)]
            out[i0:i1, j0:j1][hole] = patch[hole]
    return out


def complete_cells(tex, known, phase_i, phase_j, period, min_frac=0.3):
    """Tiles are one colour per 0.25 m cell. Complete every cell that is at least min_frac seen
    from its own seen part (inpainting), so whatever is still unknown is made of whole cells and
    the block fill's seams fall on real tile lines."""
    out, kn = tex.copy(), known.copy()
    Hh, Ww = known.shape
    n = 0
    for i in range(phase_i - period, Hh, period):
        for j in range(phase_j - period, Ww, period):
            i0, j0, i1, j1 = max(i, 0), max(j, 0), min(i + period, Hh), min(j + period, Ww)
            k = known[i0:i1, j0:j1]
            frac = k.mean()
            if frac == 1 or frac < min_frac:
                continue
            cell = np.clip(out[i0:i1, j0:j1], 0, 255).astype(np.uint8)
            filled = cv2.inpaint(cell, (~k).astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA).astype(np.float32)
            # put back some of the carpet's mottle: random seen pixels' deviation from the cell mean
            seen = cell[k].astype(np.float32)
            noise = seen[np.random.default_rng(i * 7919 + j).integers(len(seen), size=(~k).sum())] - seen.mean(0)
            filled[~k] += 0.7 * noise
            out[i0:i1, j0:j1] = np.where(k[..., None], out[i0:i1, j0:j1], filled)
            kn[i0:i1, j0:j1] = True
            n += 1
    log(f"completed {n} partly seen tile cells")
    return out, kn


def feather(a_val, width_px):
    """0..1 weight that rises from the edge of a_val over width_px."""
    dist = cv2.distanceTransform(a_val.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(dist / width_px, 0, 1)


def render_side(scene_text):
    """Render the side camera at 1080p for a scene XML text (written next to the scene so paths resolve)."""
    path = F.SCENE.with_name("_floor_tmp.xml")
    path.write_text(scene_text)
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink()
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 1080, 1920)
    r.update_scene(d, camera="ext_cam")
    img = r.render()[..., ::-1].copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="ext_cam")
    seg = r.render()
    floor_ids = [m.geom(n).id for n in ("floor_real", "floor") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
    is_floor = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], floor_ids)
    r.close()
    return img, is_floor


def scene_stat_without_floor():
    """MuJoCo sets the camera near/far clip and the shadow-map area from the model's size
    (stat.extent). A 4 m floor patch triples that size: the wrist camera's near clip went from
    19 to 56 mm (cutting off the jaws) and shadows got 3x coarser. So the scene keeps the size it
    had without the floor, fixed with a <statistic> element."""
    bak = F.SCENE.with_suffix(".xml.before_floor")
    src = (bak if bak.exists() else F.SCENE).read_text()
    src = re.sub(r'\s*<statistic[^>]*/>', "", src)
    src = re.sub(r'\s*<geom name="floor_real"[^>]*/>', "", src)
    tmp = F.SCENE.with_name("_floor_stat_tmp.xml")
    tmp.write_text(src)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    return m.stat.center.copy(), float(m.stat.extent)


def scene_with_floor(xml, mean_rgb, center, half):
    """Insert/replace the textured floor in the scene XML text."""
    xml = re.sub(r'\s*<texture name="floor_real_tex"[^>]*/>', "", xml)
    xml = re.sub(r'\s*<material name="floor_real_mat"[^>]*/>', "", xml)
    xml = re.sub(r'\s*<geom name="floor_real"[^>]*/>', "", xml)
    xml = re.sub(r'\s*<material name="floor_plain"[^>]*/>', "", xml)
    asset = (f'\n    <!-- Real lab floor (calib/floor/build_floor.py): photo + C920, metric, 2 mm/px -->'
             f'\n    <texture name="floor_real_tex" type="2d" file="assets/{TEX_NAME}"/>'
             f'\n    <material name="floor_real_mat" texture="floor_real_tex" texuniform="false" texrepeat="1 1"'
             f' specular="0" shininess="0" reflectance="0"/>'
             f'\n    <material name="floor_plain" rgba="{mean_rgb[0]:.3f} {mean_rgb[1]:.3f} {mean_rgb[2]:.3f} 1"'
             f' specular="0" shininess="0" reflectance="0"/>')
    xml = xml.replace("</asset>", asset + "\n  </asset>", 1)
    # physical floor plane: at FLOOR_Z, plain mean floor colour (only seen beyond the textured patch)
    xml = re.sub(r'<geom name="floor" size="0 0 0.05" pos="[^"]*" type="plane" material="[^"]*"/>',
                 f'<geom name="floor" size="0 0 0.05" pos="0 0 {F.FLOOR_Z}" type="plane" material="floor_plain"/>', xml)
    geom = (f'\n    <geom name="floor_real" type="box" pos="{center[0]:.4f} {center[1]:.4f} {F.FLOOR_Z - 0.0005 + 0.0006:.4f}"'
            f' size="{half[0]:.4f} {half[1]:.4f} 0.0005" material="floor_real_mat" contype="0" conaffinity="0"'
            f' group="1"/>')
    xml = re.sub(r'(<geom name="floor" [^>]*/>)', r"\1" + geom.replace("\\", "\\\\"), xml, count=1)
    # keep the model's size (and so near clip + shadow-map area) as it was without the floor
    xml = re.sub(r'\s*<statistic[^>]*/>', "", xml)
    sc, se = scene_stat_without_floor()
    stat = (f'\n  <!-- fixed to the size without the floor patch: keeps near clip ({se * 0.01 * 1000:.0f} mm)'
            f' and shadows as before (calib/floor/build_floor.py) -->'
            f'\n  <statistic center="{sc[0]:.4f} {sc[1]:.4f} {sc[2]:.4f}" extent="{se:.4f}"/>')
    xml = re.sub(r'(<mujoco model="[^"]*">)', lambda mm: mm.group(1) + stat, xml, count=1)
    # table legs down to the new floor (top at z = -0.02 under the 2 cm table top)
    leg_half = (-0.02 - F.FLOOR_Z) / 2
    xml = re.sub(r'(<geom name="table_leg_\w+" type="box" pos="[-\d.]+ [-\d.]+ )[-\d.]+(" size="0.02 0.02 )[\d.]+(")',
                 lambda mm: f"{mm.group(1)}{-0.02 - leg_half:.4f}{mm.group(2)}{leg_half:.4f}{mm.group(3)}", xml)
    return xml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true", help="don't modify the scene XML")
    args = ap.parse_args()

    img = cv2.imread(str(FRAME))
    photo = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(PHOTO)).convert("RGB")), cv2.COLOR_RGB2BGR)
    C, cv_ = c920_ortho(img)
    Hfull = register_photo(photo, C, cv_)
    P, pv = warp_photo(photo, Hfull, cv_.shape)
    M = colour_fit(P.astype(np.float32), pv, C.astype(np.float32), cv_)
    Pc = apply_colour(P.astype(np.float32), M)

    # C920-only floor (outside the photo); reject dark non-floor things (cables, chair) there
    hsv = cv2.cvtColor(C, cv2.COLOR_BGR2HSV)
    c_only = cv_ & ~pv & (hsv[..., 2] > 70)
    c_only = cv2.erode(c_only.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    w = feather(pv, 25)[..., None]                              # 5 cm feather inside the photo's edge
    tex = np.where(pv[..., None], Pc, 0).astype(np.float32)
    tex = np.where(c_only[..., None], C, tex)
    blend = (pv & cv_ & (hsv[..., 2] > 70))[..., None] & (w < 1)   # where both see the floor, near the edge
    tex = np.where(blend, w * Pc + (1 - w) * C, tex)
    known = pv | c_only

    period = int(round(0.25 / F.RES))
    pi_, pj_ = tile_phase(tex, pv, 0, period), tile_phase(tex, pv, 1, period)
    log(f"tile grid: 0.25 m period, boundaries at texture rows {pi_} + k*{period}, cols {pj_} + k*{period}")
    tex, known = complete_cells(tex, known, pi_, pj_, period)
    full, filled = tex, known.copy()
    for k, blk in enumerate((4 * period, 2 * period, period)):   # 1 m blocks, then 0.5 m, then 0.25 m
        if filled.all():
            break
        full = fill_blocks(full, filled, blk, pi_, pj_, period, seed=k)
        filled = known | (full.sum(-1) > 0)

    # lighting: render with the uncorrected texture and match the real floor per channel
    xml0 = F.SCENE.read_text()
    mean_bgr = full[known].mean(0)
    center = ((F.X0 + F.X1) / 2, (F.Y0 + F.Y1) / 2)
    half = ((F.X1 - F.X0) / 2, (F.Y1 - F.Y0) / 2)
    gains = np.ones(3)
    for it in range(3):
        out = np.clip(full * gains, 0, 255).astype(np.uint8)
        png = out.transpose(1, 0, 2)[::-1]                    # PNG cols = +x, rows = +y at the top
        cv2.imwrite(str(ASSETS / TEX_NAME), png)
        mrgb = np.clip(mean_bgr[::-1] * gains[::-1] / 255.0, 0, 1)
        xml = scene_with_floor(xml0, mrgb, center, half)
        sim_img, is_floor = render_side(xml)
        realmask = F.floor_mask_in_image(img.shape[1], img.shape[0]) & is_floor
        # leave out the arm and other non-floor things in the real frame: big local differences
        diff = np.abs(cv2.GaussianBlur(sim_img, (0, 0), 3).astype(float) - cv2.GaussianBlur(img, (0, 0), 3).astype(float)).sum(-1)
        use = realmask & (diff < np.percentile(diff[realmask], 85))
        ratio = np.median(img[use].astype(float), 0) / np.maximum(np.median(sim_img[use].astype(float), 0), 1)
        mad = np.abs(sim_img[realmask].astype(float) - img[realmask].astype(float)).mean(0)
        log(f"lighting pass {it + 1}: real/sim floor ratio (BGR) {ratio.round(3)}, mean abs diff {mad.round(1)}")
        if np.abs(ratio - 1).max() < 0.01:
            break
        gains *= ratio
    np.save(HERE / "colour_and_gain.npy", np.r_[M.ravel(), gains])

    # comparison image: real | sim | |difference|, floor strip, 640x480 crop like the policy sees
    crop = lambda a: cv2.resize(a[:, 240:1680], (640, 480), interpolation=cv2.INTER_AREA)[:200]
    dimg = cv2.applyColorMap(np.clip(np.abs(sim_img.astype(int) - img.astype(int)).sum(-1) / 1.5, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(HERE / "floor_compare.png"), np.vstack([crop(img), crop(sim_img), crop(dimg)]))
    top = cv2.resize(np.clip(full * gains, 0, 255).astype(np.uint8), None, fx=0.25, fy=0.25)
    cv2.imwrite(str(HERE / "floor_texture_preview.jpg"), top)
    log("wrote", (ASSETS / TEX_NAME).relative_to(SIM), "and calib/floor/floor_compare.png")

    if args.no_write:
        log("--no-write: scene XML unchanged")
        return
    bak = F.SCENE.with_suffix(".xml.before_floor")
    if not bak.exists():
        shutil.copy(F.SCENE, bak)
        log("backup:", bak.name)
    F.SCENE.write_text(xml)
    mujoco.MjModel.from_xml_path(str(F.SCENE))
    log("wrote", F.SCENE.name, "(reloads OK)")


if __name__ == "__main__":
    main()
