"""On-disk format for SO-101 pick-and-place demonstrations.

One demonstration is one compressed .npz holding per-control-step arrays plus a
JSON metadata blob. Everything is sampled at the *control* rate, not the
physics rate: the controller holds one action for CONTROL_SUBSTEPS physics
steps, and we log exactly one frame per such hold.

Frame convention (matters for replay -- see replay_demo.py):

    frame k  =  the state observed at time[k], together with the action that
                was commanded *from* that state and then integrated forward for
                CONTROL_SUBSTEPS physics steps.

So mj_state[k] is the state *before* action[k] is applied, which makes replay a
straight loop: restore mj_state[0], then for each k set ctrl = action[k] and
step CONTROL_SUBSTEPS times.

Camera images are deliberately not stored. mj_state is mjSTATE_INTEGRATION, so
any view of the episode (wrist cam, external cam, a camera that didn't exist at
record time) can be re-rendered exactly from the file -- which is how the
LeRobot export in step 5 will get its frames, at whatever resolution it wants.
"""
import json
import pathlib

import mujoco
import numpy as np

SCHEMA_VERSION = 1

# Physics steps per control step. The scene uses a 5 ms timestep, so 6 substeps
# gives a 30 ms (33.3 Hz) control period -- the rate at which actions are
# recorded and must be replayed.
CONTROL_SUBSTEPS = 6

SITE_NAME = "gripperframe"
OBJECT_BODY = "eraser"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

TASK_DESCRIPTION = "Pick up the eraser and place it in the container"
# The language instruction stored with every episode (and fed to the policy), per scene object.
TASKS = {
    "eraser": TASK_DESCRIPTION,
    "cube": "Pick and place the black cube into the container",
}


def task_for(object_body):
    """Task instruction for a scene's pick object ('eraser' or 'cube')."""
    return TASKS[object_body]


def control_dt(model):
    return CONTROL_SUBSTEPS * model.opt.timestep


def site_quat(data, site_id):
    """World-frame orientation of a site as a wxyz quaternion."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return quat


class DemoRecorder:
    """Accumulates one episode's worth of control-step frames in memory."""

    def __init__(self, model, site_name=SITE_NAME, object_body=OBJECT_BODY):
        self.model = model
        self.site_id = model.site(site_name).id
        self.object_bid = model.body(object_body).id
        self.site_name = site_name
        self.object_body = object_body
        self.nstate = mujoco.mj_stateSize(model, STATE_SPEC)
        self._frames = []

    def __len__(self):
        return len(self._frames)

    def clear(self):
        self._frames.clear()

    def capture(self, data, action):
        """Log the current state plus the action about to be applied to it.

        Call this after the state is settled for this control step and after
        the action is decided, but *before* writing data.ctrl and stepping.
        """
        state = np.zeros(self.nstate)
        mujoco.mj_getState(self.model, data, state, STATE_SPEC)
        self._frames.append({
            "time": float(data.time),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "mj_state": state,
            "action": np.asarray(action, dtype=float).copy(),
            # First 6 qpos/qvel entries are the arm's 6 actuated joints; the
            # remaining entries are the eraser's free joint.
            "joint_pos": data.qpos[:6].copy(),
            "joint_vel": data.qvel[:6].copy(),
            "ee_pos": data.site_xpos[self.site_id].copy(),
            "ee_quat": site_quat(data, self.site_id),
            "obj_pos": data.xpos[self.object_bid].copy(),
            "obj_quat": data.xquat[self.object_bid].copy(),
        })

    def arrays(self):
        if not self._frames:
            raise ValueError("nothing recorded")
        keys = self._frames[0].keys()
        return {k: np.array([f[k] for f in self._frames]) for k in keys}

    def save(self, path, scene_path, success, source, extra_metadata=None):
        """Write the episode to `path` (.npz). Returns the path written."""
        model = self.model
        arrays = self.arrays()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "task": task_for(self.object_body),
            "source": source,                      # "teleop" | "scripted" | ...
            "success": bool(success),
            "scene_path": str(scene_path),
            "num_frames": len(self._frames),
            "timestep": float(model.opt.timestep),
            "control_substeps": CONTROL_SUBSTEPS,
            "control_dt": control_dt(model),
            "fps": 1.0 / control_dt(model),
            "duration_s": arrays["time"][-1] - arrays["time"][0],
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "state_spec": "mjSTATE_INTEGRATION",
            "ee_site": self.site_name,
            "object_body": self.object_body,
            "joint_names": [model.joint(i).name for i in range(6)],
            "actuator_names": [model.actuator(i).name for i in range(model.nu)],
            "action_space": "position targets for the 6 arm actuators (rad)",
            "cameras": [model.camera(i).name for i in range(model.ncam)],
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata_json=json.dumps(metadata, indent=2),
                            **arrays)
        return path


class Demo:
    """A loaded demonstration: `demo.action`, `demo.ee_pos`, `demo.meta[...]`."""

    def __init__(self, arrays, metadata, path=None):
        self._arrays = arrays
        self.meta = metadata
        self.path = path

    def __len__(self):
        return len(self._arrays["time"])

    def __getattr__(self, name):
        try:
            return self._arrays[name]
        except KeyError:
            raise AttributeError(name) from None

    def keys(self):
        return list(self._arrays)


def load_demo(path):
    path = pathlib.Path(path)
    with np.load(path, allow_pickle=False) as f:
        metadata = json.loads(str(f["metadata_json"]))
        arrays = {k: f[k] for k in f.files if k != "metadata_json"}
    return Demo(arrays, metadata, path)


def next_demo_path(directory, prefix="demo", suffix=".npz"):
    """First unused `<directory>/<prefix>_NNN<suffix>`."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        candidate = directory / f"{prefix}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
