"""Replay real SO-100 commands in MuJoCo and measure how far the sim follower drifts from the real one.

Real data: a LeRobot SO-100 dataset (e.g. lerobot/svla_so100_pickplace) with `action` = leader command and
`observation.state` = follower position, both [6] in SO-100 units (5 joints in degrees, gripper 0-100).

The sim arm is the bare menagerie model (no table, no cubes). The command is held for one frame (zero-order hold)
and the physics runs at the model timestep. Per-joint sign and zero offset are inputs: LeRobot puts degree zero at
the middle of the operator's calibrated range, so the offset is per setup and has to be fitted, not assumed.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from . import scene
from .smolvla_export import JAW_RANGE

N_ARM = 5


@dataclass
class Params:
    kp: float = 50.0             # menagerie so_arm100.xml defaults
    dampratio: float = 1.0
    armature: float = 0.1
    frictionloss: float = 0.1
    damping: float = 0.0         # passive joint damping
    delay: int = 0               # command delay, in frames


def build(p: Params):
    spec = mujoco.MjSpec.from_file(scene.ARM_XML)
    for a in spec.actuators[:N_ARM]:
        a.gainprm[0] = p.kp
        a.biasprm[1] = -p.kp
        a.biasprm[2] = p.dampratio   # positive = dampratio, resolved at compile
    for j in spec.joints[:N_ARM]:
        j.armature = p.armature
        j.frictionloss = p.frictionloss
        j.damping[0] = p.damping     # a vector in MjSpec (mujoco 3.13); [0] = linear damping
    return spec.compile()


def to_sim(u, sign, offset):
    """SO-100 units [..., 6] -> sim radians. sign/offset apply to the 5 arm joints."""
    u = np.asarray(u, dtype=np.float64)
    q = np.empty_like(u)
    q[..., :N_ARM] = np.asarray(sign) * np.deg2rad(u[..., :N_ARM]) + np.asarray(offset)
    lo, hi = JAW_RANGE
    q[..., N_ARM] = lo + u[..., N_ARM] / 100.0 * (hi - lo)
    return q


def replay(model, q_cmd, q0, fps=30, delay=0):
    """Hold each command for 1/fps s. Returns sim joint positions [T, 6] sampled at the frame starts."""
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    T = len(q_cmd)
    out = np.empty((T, model.nq))
    steps_done = 0
    for t in range(T):
        out[t] = data.qpos
        data.ctrl[:] = q_cmd[max(t - delay, 0)]
        target = int(round((t + 1) / fps / model.opt.timestep))
        for _ in range(target - steps_done):
            mujoco.mj_step(model, data)
        steps_done = target
    return out


def tracking_error(q_sim, q_real):
    """Mean |error| per arm joint, in degrees."""
    return np.rad2deg(np.abs(q_sim[:, :N_ARM] - q_real[:, :N_ARM]).mean(0))


def load_episodes(parquet_path):
    """{episode_index: (action [T,6], state [T,6])} in SO-100 units."""
    import pandas as pd
    d = pd.read_parquet(parquet_path, columns=["action", "observation.state", "episode_index", "frame_index"])
    d = d.sort_values(["episode_index", "frame_index"])
    eps = {}
    for ep, g in d.groupby("episode_index"):
        eps[int(ep)] = (np.stack(g["action"].to_numpy()), np.stack(g["observation.state"].to_numpy()))
    return eps


def episode_error(p: Params, action, state, sign, offset, fps=30):
    """Replay one real episode from its first measured state; per-joint mean |error| in degrees."""
    model = build(p)
    q_cmd = to_sim(action, sign, offset)
    q_real = to_sim(state, sign, offset)
    q_sim = replay(model, q_cmd, q_real[0], fps=fps, delay=p.delay)
    return tracking_error(q_sim, q_real)
