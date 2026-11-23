"""Build the real placement container (transparent open-top box with rounded corners,
outlined on the tabletop by a band of green tape stuck round its base) and put it into
the cube scene (pick_place_scene_real.xml).

    cd sim
    /opt/anaconda3/envs/mujoco-so101/bin/python container/build_container.py            # LENGTH / WIDTH below
    /opt/anaconda3/envs/mujoco-so101/bin/python container/build_container.py --length 0.23 --length-from front
    /opt/anaconda3/envs/mujoco-so101/bin/python container/build_container.py --check    # just render the check images

Resizing: set LENGTH and WIDTH below (or --length / --width) and re-run. LENGTH_FROM and
WIDTH_FROM pick which side moves; the opposite side stays where the container is placed
now (PLACED_*), so the real container and tape only need moving on that one side. The tape
is built from the container's outline, so it follows the resize automatically.

Measured: outer footprint 220 x 130 mm, corners rounded with a 17 mm radius, 55 mm tall,
transparent. Not yet measured (defaults are guesses, pass the real values): wall and floor
thickness, the tape colour, how clear the plastic is (--alpha).

What it writes:
  - assets/container_shell.obj, assets/container_tape.obj: visual meshes (the shell is
    one watertight solid: outer wall, rim, inner wall, inner floor, base; the tape is a
    flat band 0.4 mm thick on the tabletop round the container's base, square corners)
  - the <body name="container"> and its contact pairs into the scene, between
    "container:begin/end" markers (re-running replaces them; backup .before_container)
Collision is boxes and cylinders (MuJoCo collides with a mesh's convex hull, which for an
open box would be a solid block): two floor slabs plus corner discs, four straight walls,
and each 90-degree corner as a run of short tangent wall segments.

The success test (pick_place_task.object_in_container) reads the container's position
from the body and its inside tolerances from <numeric name="container_tol">.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROBOT_DIR = SIM / "mujoco_menagerie/robotstudio_so101"
SCENE = ROBOT_DIR / "pick_place_scene_real.xml"
ASSETS = ROBOT_DIR / "assets"

CUBE_HALF = 0.02
CORNER_SEGMENTS = 24          # visual mesh: segments per 90-degree corner
CORNER_COLLIDERS = 4          # collision: wall segments per corner
STIFF = 'solref="0.01 1" solimp="0.95 0.99 0.001"'   # as the cube-table pair

# ---- container size: edit these and re-run ----------------------------------------------
LENGTH = 0.2030          # outer length, m: along x (toward / away from the robot's front)
WIDTH = 0.1310           # outer width, m: along y (sideways)
# Which side moves when the size differs from the placed footprint (the opposite side stays put):
#   LENGTH_FROM: "front"  = the far end (away from the robot's base) moves
#                "back"   = the end nearest the base moves
#                "centre" = both ends move by half (centre stays)
#   WIDTH_FROM:  "outer"  = the side away from the robot moves
#                "inner"  = the side facing the robot moves
#                "centre" = both sides move by half
LENGTH_FROM = "centre"
WIDTH_FROM = "centre"
# Where the container is placed now: the fixed sides are this footprint's. live_overlay.py's
# container mode (Tab) rewrites LENGTH, WIDTH and these lines as you resize.
PLACED_CENTRE = (0.2095, -0.2245)
PLACED_LENGTH, PLACED_WIDTH = 0.2030, 0.1310

# ---- colours (RGB 0-1); container/match_color.py fits these to the real side camera -------
TAPE_COLOR = (0.196, 0.368, 0.270)
CONTAINER_COLOR = (0.714, 0.727, 0.673)
CONTAINER_ALPHA = 0.25


def centre_for(length, width, length_from, width_from):
    """Container centre that keeps the non-moving sides of the placed footprint where they are."""
    dl, dw = length - PLACED_LENGTH, width - PLACED_WIDTH
    cx = PLACED_CENTRE[0] + {"front": dl / 2, "back": -dl / 2, "centre": 0.0}[length_from]
    cy = PLACED_CENTRE[1] + {"outer": -dw / 2, "inner": dw / 2, "centre": 0.0}[width_from]
    return cx, cy


def rounded_rect(hx, hy, r, n=CORNER_SEGMENTS):
    """Outline (counter-clockwise from +x side) of a rectangle with half-sizes hx, hy and corner radius r."""
    pts = []
    for cx, cy, a0 in ((hx - r, hy - r, 0), (-hx + r, hy - r, 90), (-hx + r, -hy + r, 180), (hx - r, -hy + r, 270)):
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / max(n, 1))      # n=0, r=0: one sharp corner point
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


class Mesh:
    def __init__(self):
        self.v, self.f = [], []

    def add_verts(self, pts):
        i = len(self.v)
        self.v.extend(pts)
        return i

    def wall(self, ring, z0, z1, outward=True):
        """Side faces of an extruded closed outline (shares vertices round the ring, so it shades smoothly)."""
        n = len(ring)
        lo = self.add_verts([(x, y, z0) for x, y in ring])
        hi = self.add_verts([(x, y, z1) for x, y in ring])
        for k in range(n):
            a, b = k, (k + 1) % n
            q = (lo + a, lo + b, hi + b, hi + a) if outward else (lo + b, lo + a, hi + a, hi + b)
            self.f += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]

    def annulus(self, outer, inner, z, up=True):
        """Flat ring between two outlines with the same vertex count."""
        n = len(outer)
        o = self.add_verts([(x, y, z) for x, y in outer])
        i = self.add_verts([(x, y, z) for x, y in inner])
        for k in range(n):
            a, b = k, (k + 1) % n
            q = (o + a, o + b, i + b, i + a)
            if not up:
                q = q[::-1]
            self.f += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]

    def disc(self, ring, z, up=True):
        c = self.add_verts([(0.0, 0.0, z)])
        r = self.add_verts([(x, y, z) for x, y in ring])
        n = len(ring)
        for k in range(n):
            a, b = r + k, r + (k + 1) % n
            self.f.append((c, a, b) if up else (c, b, a))

    def write_obj(self, path, comment):
        lines = [f"# {comment}"] + [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.v]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.f]
        path.write_text("\n".join(lines) + "\n")


def build_meshes(a):
    hx, hy, r, t, ft, h = a.length / 2, a.width / 2, a.radius, a.wall, a.floor, a.height
    outer = rounded_rect(hx, hy, r)
    inner = rounded_rect(hx - t, hy - t, r - t)
    shell = Mesh()
    shell.wall(outer, 0.0, h, outward=True)
    shell.annulus(outer, inner, h, up=True)              # rim
    shell.wall(inner, ft, h, outward=False)              # inner wall faces inwards
    shell.disc(inner, ft, up=True)                       # inside floor
    shell.disc(outer, 0.0, up=False)                     # base
    shell.write_obj(ASSETS / "container_shell.obj",
                    f"container shell {a.length * 1000:.0f}x{a.width * 1000:.0f}x{h * 1000:.0f} mm, r {r * 1000:.0f} mm "
                    "(sim/container/build_container.py)")

    # tape on the tabletop round the base: four straight strips laid against the walls, so the
    # band is a rectangle with square corners (inner and outer), not following the rounded
    # corners -- the table shows in the small gap between each rounded corner and the tape
    g0, g1 = a.tape_gap, a.tape_gap + a.tape_width
    t_in, t_out = rounded_rect(hx + g0, hy + g0, 0.0, n=0), rounded_rect(hx + g1, hy + g1, 0.0, n=0)
    z0, z1 = 0.0001, 0.0005                              # 0.4 mm thick, just above the table (no z-fighting)
    tape = Mesh()
    tape.wall(t_out, z0, z1, outward=True)
    tape.wall(t_in, z0, z1, outward=False)
    tape.annulus(t_out, t_in, z1, up=True)
    tape.annulus(t_out, t_in, z0, up=False)
    tape.write_obj(ASSETS / "container_tape.obj", f"tabletop tape band {a.tape_width * 1000:.0f} mm round the container")


def collision_geoms(a):
    """(name, xml attrs) of the container's collision geoms, in the body frame (origin: base centre)."""
    hx, hy, r, t, ft, h = a.length / 2, a.width / 2, a.radius, a.wall, a.floor, a.height
    g = []
    box = lambda name, p, s, yaw=0.0: g.append((name, (
        f'type="box" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" size="{s[0]:.5f} {s[1]:.5f} {s[2]:.5f}"'
        + (f' euler="0 0 {yaw:.5f}"' if yaw else ""))))
    # floor: a cross of two slabs and a disc in each corner. (A resting cube micro-rocks on the
    # overlapping pieces at ~0.01 rad/s; the success test's at-rest thresholds allow for that, see
    # <numeric name="container_rest">. Kept as recorded: the 100 cube demos replay exactly with it.)
    box("container_floor_x", (0, 0, ft / 2), (hx - t, hy - r, ft / 2))
    box("container_floor_y", (0, 0, ft / 2), (hx - r, hy - t, ft / 2))
    zc, hw = h / 2, h / 2
    box("container_wall_px", (hx - t / 2, 0, zc), (t / 2, hy - r, hw))
    box("container_wall_nx", (-hx + t / 2, 0, zc), (t / 2, hy - r, hw))
    box("container_wall_py", (0, hy - t / 2, zc), (hx - r, t / 2, hw))
    box("container_wall_ny", (0, -hy + t / 2, zc), (hx - r, t / 2, hw))
    rm = r - t / 2                                       # wall centre-line radius at the corners
    half_seg = rm * np.tan(np.radians(90 / CORNER_COLLIDERS / 2)) * 1.05
    for ci, (cx, cy, a0) in enumerate(((hx - r, hy - r, 0), (-hx + r, hy - r, 90),
                                       (-hx + r, -hy + r, 180), (hx - r, -hy + r, 270))):
        g.append((f"container_floor_c{ci}", f'type="cylinder" pos="{cx:.5f} {cy:.5f} {ft / 2:.5f}" '
                                            f'size="{r - t:.5f} {ft / 2:.5f}"'))
        for k in range(CORNER_COLLIDERS):
            ang = np.radians(a0 + 90 * (k + 0.5) / CORNER_COLLIDERS)
            p = (cx + rm * np.cos(ang), cy + rm * np.sin(ang), zc)
            box(f"container_wall_c{ci}{k}", p, (t / 2, half_seg, hw), yaw=ang)
    return g


def body_xml(a, geoms):
    px, py = a.pos
    tol = (a.length / 2 - a.wall - CUBE_HALF - 0.005, a.width / 2 - a.wall - CUBE_HALF - 0.005)
    col = " ".join(f"{v:g}" for v in a.color)
    tape = " ".join(f"{v:g}" for v in a.tape_color)
    body = [
        f'    <!-- container:begin (sim/container/build_container.py; edit there and re-run) -->',
        f'    <!-- Placement container: {a.length * 1000:.0f} x {a.width * 1000:.0f} mm outer, corners r {a.radius * 1000:.0f} mm,',
        f'         {a.height * 1000:.0f} mm tall, transparent (wall thickness a guess; colours fitted to the real camera),',
        f'         outlined by a {a.tape_width * 1000:.0f} mm band of green tape (square corners) stuck to the table round its base. Fixed to the',
        f'         tabletop beside the robot, opposite the cube\'s start area, fully inside the side',
        f'         camera\'s frame. Visual meshes; collision boxes/cylinders in group 3. -->',
        f'    <body name="container" pos="{px:.4f} {py:.4f} 0" euler="0 0 {np.radians(a.yaw):.5f}">',
        f'      <geom name="container_shell" type="mesh" mesh="container_shell" material="container_mat" '
        f'contype="0" conaffinity="0" mass="0"/>',
        f'      <geom name="container_tape" type="mesh" mesh="container_tape" material="container_tape_mat" '
        f'contype="0" conaffinity="0" mass="0"/>',
    ]
    body += [f'      <geom name="{n}" {attrs} group="3" rgba="{col} 1"/>' for n, attrs in geoms]
    body += ['    </body>', '    <!-- container:end -->']

    top = [
        '  <!-- container:begin (sim/container/build_container.py) -->',
        '  <asset>',
        '    <mesh name="container_shell" file="container_shell.obj"/>',
        '    <mesh name="container_tape" file="container_tape.obj"/>',
        f'    <!-- clear plastic: mostly see-through, with a little gloss -->',
        f'    <material name="container_mat" rgba="{col} {a.alpha:g}" specular="0.4" shininess="0.6" reflectance="0"/>',
        f'    <!-- tape and container colours: fitted to the real side camera (container/match_color.py) -->',
        f'    <material name="container_tape_mat" rgba="{tape} 1" specular="0" shininess="0" reflectance="0"/>',
        '  </asset>',
        '  <custom>',
        '    <!-- object_in_container(): cube centre within +/-x, +/-y of the container centre',
        '         (inside half-size minus the cube half-size minus 5 mm) and z within [zmin, zmax] -->',
        f'    <numeric name="container_tol" data="{tol[0]:.4f} {tol[1]:.4f} 0 {a.height:.4f}"/>',
        '    <!-- object_in_container(): "at rest" = linear speed < 0.005 m/s and angular speed < 0.05 rad/s',
        '         (a resting cube micro-rocks at ~0.01 rad/s on the container floor; the eraser scenes keep',
        '         their original angular-only 0.01 rad/s test) -->',
        '    <numeric name="container_rest" data="0.005 0.05"/>',
        '  </custom>',
        '  <contact>',
        '    <!-- cube-container: stiff like cube-table, so a dropped cube does not sink into the floor -->',
        '    <exclude body1="cube" body2="container"/>',
    ]
    top += [f'    <pair geom1="cube_geom" geom2="{n}" condim="3" {STIFF} friction="1 1 0.005 0.0001 0.0001"/>'
            for n, _ in geoms]
    top += ['  </contact>', '  <!-- container:end -->']
    return "\n".join(body), "\n".join(top)


def patch_scene(body, top):
    s = SCENE.read_text()
    bak = SCENE.with_suffix(".xml.before_container")
    if not bak.exists():
        shutil.copy(SCENE, bak)
    s = re.sub(r"\n[ ]*<!-- container:begin.*?container:end -->", "", s, flags=re.S)
    s = s.replace("  </worldbody>", body + "\n  </worldbody>", 1)
    s = s.replace("</mujoco>", top + "\n</mujoco>", 1)
    SCENE.write_text(s)


def check(out_dir):
    """Render the side camera (ready pose) and the wrist camera, and report whether the container
    is fully in the side camera's frame and which of its points the arm can reach top-down."""
    import mujoco
    sys.path.insert(0, str(SIM))
    from PIL import Image

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("ready").id)
    mujoco.mj_forward(m, d)
    cid = m.camera("ext_cam").id
    p, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    f = 240 / np.tan(np.radians(m.cam_fovy[cid] / 2))
    for name in ("container_shell", "container_tape"):
        g = m.geom(name).id
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        world = d.geom_xpos[g] + m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T
        cam = (world - p) @ R
        u = 320 + f * cam[:, 0] / -cam[:, 2]
        v = 240 - f * cam[:, 1] / -cam[:, 2]
        inside = u.min() > 0 and u.max() < 640 and v.min() > 0 and v.max() < 480
        print(f"side camera: {name[10:]:5s} spans u {u.min():.0f}..{u.max():.0f}, v {v.min():.0f}..{v.max():.0f} "
              f"(frame 640x480) -> {'fully in view' if inside else 'CUT OFF'}")

    m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, camera="ext_cam")
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(r.render()).save(out_dir / "side_cam.png")
    print("wrote", out_dir / "side_cam.png")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=LENGTH, help="outer length, m (along world x at yaw 0)")
    ap.add_argument("--width", type=float, default=WIDTH, help="outer width, m")
    ap.add_argument("--length-from", choices=("front", "back", "centre"), default=LENGTH_FROM,
                    help="which end moves when the length changes (see LENGTH_FROM)")
    ap.add_argument("--width-from", choices=("outer", "inner", "centre"), default=WIDTH_FROM,
                    help="which side moves when the width changes (see WIDTH_FROM)")
    ap.add_argument("--radius", type=float, default=0.017, help="outer corner radius, m")
    ap.add_argument("--height", type=float, default=0.055, help="outer height, m")
    ap.add_argument("--wall", type=float, default=0.002, help="wall thickness, m (guess)")
    ap.add_argument("--floor", type=float, default=0.002, help="floor thickness, m (guess)")
    ap.add_argument("--tape-width", type=float, default=0.018, help="tape band width, m")
    ap.add_argument("--tape-color", type=float, nargs=3, default=TAPE_COLOR, help="tape RGB 0-1")
    ap.add_argument("--tape-gap", type=float, default=0.0, help="gap between the container's base and the tape, m")
    ap.add_argument("--color", type=float, nargs=3, default=CONTAINER_COLOR, help="container tint RGB 0-1")
    ap.add_argument("--alpha", type=float, default=CONTAINER_ALPHA, help="container opacity 0-1 (transparent plastic)")
    ap.add_argument("--pos", type=float, nargs=2, default=None,
                    help="centre on the tabletop, m (overrides the placement from PLACED_* and the *_FROM sides)")
    ap.add_argument("--yaw", type=float, default=0.0, help="rotation about z, deg (0: long side along x)")
    ap.add_argument("--check", action="store_true", help="only render the check images")
    return ap


def default_args():
    """The script's settings (the variables at the top) as an args namespace, for other tools."""
    return make_parser().parse_args([])


def build(a):
    """Write the meshes and the container into the scene for settings `a` (pos=None: from PLACED_*)."""
    if a.pos is None:
        a.pos = centre_for(a.length, a.width, a.length_from, a.width_from)
    build_meshes(a)
    patch_scene(*body_xml(a, collision_geoms(a)))


def save_size(length, width, centre):
    """Store a new size and placement in the variables at the top of this file (used by
    calib/live_overlay.py's container mode), so a plain re-run rebuilds the same container."""
    src = Path(__file__).read_text()
    subs = [(r"^LENGTH = [0-9.]+", f"LENGTH = {length:.4f}"),
            (r"^WIDTH = [0-9.]+", f"WIDTH = {width:.4f}"),
            (r"^PLACED_CENTRE = \([^)]*\)", f"PLACED_CENTRE = ({centre[0]:.4f}, {centre[1]:.4f})"),
            (r"^PLACED_LENGTH, PLACED_WIDTH = [0-9.]+, [0-9.]+", f"PLACED_LENGTH, PLACED_WIDTH = {length:.4f}, {width:.4f}")]
    _set_vars(subs)


def save_colors(tape=None, container=None, alpha=None):
    """Store fitted colours in the variables at the top of this file (container/match_color.py)."""
    t3 = lambda c: "(" + ", ".join(f"{v:.3f}" for v in c) + ")"
    subs = []
    if tape is not None:
        subs.append((r"^TAPE_COLOR = \([^)]*\)", f"TAPE_COLOR = {t3(tape)}"))
    if container is not None:
        subs.append((r"^CONTAINER_COLOR = \([^)]*\)", f"CONTAINER_COLOR = {t3(container)}"))
    if alpha is not None:
        subs.append((r"^CONTAINER_ALPHA = [0-9.]+", f"CONTAINER_ALPHA = {alpha:.3f}"))
    _set_vars(subs)


def _set_vars(subs):
    src = Path(__file__).read_text()
    for pat, rep in subs:
        src, n = re.subn(pat, rep, src, count=1, flags=re.M)
        assert n == 1, pat
    Path(__file__).write_text(src)


def main():
    a = make_parser().parse_args()
    if a.pos is None:
        a.pos = centre_for(a.length, a.width, a.length_from, a.width_from)
    if not a.check:
        cx, cy = a.pos
        print(f"container {a.length * 1000:.0f} x {a.width * 1000:.0f} mm, centre ({cx * 1000:.1f}, {cy * 1000:.1f}) mm: "
              f"x {(cx - a.length / 2) * 1000:.1f}..{(cx + a.length / 2) * 1000:.1f}, "
              f"y {(cy - a.width / 2) * 1000:.1f}..{(cy + a.width / 2) * 1000:.1f} mm (base); tape outer edge "
              f"x {(cx - a.length / 2 - a.tape_gap - a.tape_width) * 1000:.1f}..{(cx + a.length / 2 + a.tape_gap + a.tape_width) * 1000:.1f}, "
              f"y {(cy - a.width / 2 - a.tape_gap - a.tape_width) * 1000:.1f}..{(cy + a.width / 2 + a.tape_gap + a.tape_width) * 1000:.1f} mm")
        build(a)
        print("wrote", ASSETS / "container_shell.obj", ASSETS / "container_tape.obj", "and the container into", SCENE.name)
    check(HERE / "check")


if __name__ == "__main__":
    main()
