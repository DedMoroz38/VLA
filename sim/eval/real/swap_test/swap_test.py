"""Offline swap test: does the policy locate the cube from the REAL side camera?
Side image = first frame (arm at 'ready', real cube on pose j) of a recorded --sim-wrist episode.
Wrist image = sim render at 'ready' with the cube on pose i. Plan once per (i, j); see whether the
planned motion follows the side image's cube (j) or the wrist's cube (i). Reference: the same
with the SIM side image of pose j."""
import json, sys, pathlib, numpy as np, cv2
SIM = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM))
import run_policy_real as r, joint_units
OUT = pathlib.Path(__file__).parent
EP = SIM / "eval/real/smolvla_cube_020000_simwrist"
pf = json.load(open(SIM / "eval/cube_poses.json")); by = {p["id"]: p for p in pf["poses"]}
sim = r.SimView(SIM / pf["scene"]); jm = joint_units.JointMap.load()
ready_sim = sim.ready_sim; ready = jm.sim_to_real(ready_sim).astype(np.float32)
pol = r.Policy(SIM / "outputs/train/smolvla_cube/020000/pretrained_model", None, None, r.task_for(sim.obj))
ids = [300, 302, 305, 308, 311, 313, 500, 509]          # spread over the test region
def real_side(pid):
    cap = cv2.VideoCapture(str(EP / f"pose{pid:04d}.mp4")); ok, f = cap.read(); cap.release()
    return cv2.cvtColor(f[:, :640], cv2.COLOR_BGR2RGB)
pose = lambda pid: (by[pid]["x"], by[pid]["y"], by[pid]["yaw_rad"])
start = sim.gripper_path(ready_sim[None])[0]
sim_imgs = {pid: sim.images(ready_sim, pose(pid)) for pid in ids}
real_ext = {pid: real_side(pid) for pid in ids}
rows = []
for side_src in ("real", "sim"):
    for i in ids:              # wrist pose
        for j in ids:          # side-image pose
            ext = real_ext[j] if side_src == "real" else sim_imgs[j]["ext_cam"]
            chunk = pol.chunk(ready, {"ext_cam": ext, "wrist_cam": sim_imgs[i]["wrist_cam"]})
            path = sim.gripper_path(jm.real_to_sim(chunk))
            rows.append({"side": side_src, "wrist_pose": i, "side_pose": j,
                         "dpan_deg": float(chunk[-1, 0] - ready[0]),
                         "end_xy": path[-1, :2].tolist()})
        print(side_src, "wrist pose", i, "done", flush=True)
json.dump(rows, open(OUT / "swap_results.json", "w"), indent=1)
# analysis: does the plan's sideways swing (shoulder pan) follow the cube's direction in the side image or in the wrist image?
az = {pid: np.degrees(np.arctan2(by[pid]["y"], by[pid]["x"])) for pid in ids}
for side_src in ("real", "sim"):
    R = [x for x in rows if x["side"] == side_src]
    d = np.array([x["dpan_deg"] for x in R]); a_side = np.array([az[x["side_pose"]] for x in R]); a_wr = np.array([az[x["wrist_pose"]] for x in R])
    X = np.c_[a_side - a_side.mean(), a_wr - a_wr.mean(), np.ones(len(d))]
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    print(f"\n{side_src.upper()} side image: pan swing in the 1.5 s plan vs the cube's direction (deg per deg)")
    print(f"  follows the SIDE image's cube: {coef[0]:+.2f}  (corr {np.corrcoef(a_side, d)[0,1]:+.2f})")
    print(f"  follows the WRIST image's cube: {coef[1]:+.2f}  (corr {np.corrcoef(a_wr, d)[0,1]:+.2f})")
    print(f"  pan swing range {d.min():+.1f} .. {d.max():+.1f} deg")
# a picture: for each side-image pose, real side frame + the plans' end points (mean over wrist poses)
tiles = []
for j in ids:
    ext = cv2.cvtColor(real_ext[j], cv2.COLOR_RGB2BGR).copy()
    for poly in sim.cube_outline(*pose(j)):
        cv2.polylines(ext, [np.round(poly).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
    for side_src, col in (("real", (0, 0, 255)), ("sim", (0, 200, 0))):
        for x in rows:
            if x["side"] == side_src and x["side_pose"] == j:
                p = sim.project(np.array([[*x["end_xy"], 0.05]]))[0]
                cv2.circle(ext, tuple(np.round(p).astype(int)), 3, col, -1)
    s = sim.project(start[None])[0]; cv2.drawMarker(ext, tuple(np.round(s).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(ext, f"side image: pose {j}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tiles.append(cv2.resize(ext, (320, 240)))
cv2.imwrite(str(OUT / "swap_test.png"), np.vstack([np.hstack(tiles[k:k + 4]) for k in (0, 4)]))
print("wrote", OUT / "swap_test.png")
