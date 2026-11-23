"""Trace one closed-loop rollout of a policy against the scripted expert at the same pose.

Prints, every 15 control steps, the policy's commanded joints and gripper beside
the expert's (demo frame at the same time index), plus the eraser's height and
position, to show where the two diverge. Run on a GPU node from sim_assets:
    python diag_onepose.py CKPT demos/onepose_trimmed/scripted_000.npz
"""
import sys

import mujoco
import numpy as np
import torch

from demo_io import CONTROL_SUBSTEPS, STATE_SPEC, TASK_DESCRIPTION, load_demo
from eval_policy import load_policy
from lerobot.policies.utils import prepare_observation_for_inference
from pick_place_task import SCENE_PATH, reset_to_ready, set_eraser_pose

ckpt, demo_path = sys.argv[1], sys.argv[2]
width, height = 320, 240
demo = load_demo(demo_path)
model = mujoco.MjModel.from_xml_path(SCENE_PATH if len(sys.argv) < 4 else sys.argv[3])
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height, width)
policy, pre, post = load_policy(ckpt, "cuda")
eid = model.body("eraser").id
lo, hi = model.actuator_ctrlrange[:6].T

x, y, _ = demo.obj_pos[0]
w, _, _, z = demo.obj_quat[0]
reset_to_ready(model, data)
set_eraser_pose(model, data, x, y, 2 * np.arctan2(z, w))
policy.reset(); pre.reset(); post.reset()
print("demo length", len(demo), "frames; gripper action: expert closes at",
      int(np.argmax(demo.action[:, 5] < 0.3)), "opens again at",
      int(len(demo) - np.argmax(demo.action[::-1, 5] < 0.3)))
print("   t | policy action (pan lift elbow wflex wroll grip) | expert action                        | grip q | eraser x y z")
for t in range(500):
    obs = {"observation.state": data.qpos[:6].astype(np.float32)}
    for cam in ("ext_cam", "wrist_cam"):
        renderer.update_scene(data, camera=cam)
        obs[f"observation.images.{cam}"] = renderer.render().copy()
    with torch.inference_mode():
        b = prepare_observation_for_inference(obs, torch.device("cuda"), TASK_DESCRIPTION, "so101")
        a = post(policy.select_action(pre(b))).squeeze(0).float().cpu().numpy()
    if t % 15 == 0:
        e = demo.action[min(t, len(demo) - 1)]
        p = data.xpos[eid]
        print(f"{t:4d} | {np.round(a, 2)} | {np.round(e, 2)} | {data.qpos[5]:5.2f} | "
              f"{p[0]:.3f} {p[1]:+.3f} {p[2]:.3f}")
    data.ctrl[:6] = np.clip(a, lo, hi)
    for _ in range(CONTROL_SUBSTEPS):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
