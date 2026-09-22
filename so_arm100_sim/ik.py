"""Damped least-squares inverse kinematics on the end-effector site.

Used by the scripted expert only. A learned policy never sees this file; it
is what produces the demonstrations and the reference success rates.

Five joints move the site (`Jaw` does not), and the target is a 6-D pose, so
the system is over-determined and solved in the least-squares sense with
Tikhonov damping: dq = J^T (J J^T + lambda I)^-1 e. Orientation error is
down-weighted against position, because on a 5-DOF arm one rotational degree
is structurally unreachable and forcing it would pull the position off.

Everything runs on a scratch `MjData` with `mj_kinematics` + `mj_comPos`
only: no dynamics, no contacts, no side effects on the simulation the
expert is acting in.
"""
import mujoco
import numpy as np

from . import scene


class IK:
    def __init__(self, model, damping=1e-3, rot_weight=0.3, step=0.6,
                 max_iters=200, pos_tol=1e-3, rot_tol=2e-2):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = scene.ee_site_id(model)
        self.qpos_i = scene.arm_qpos_index(model)[:5]
        self.qvel_i = scene.arm_qvel_index(model)[:5]
        self.lo, self.hi = scene.joint_limits(model)
        self.lo, self.hi = self.lo[:5], self.hi[:5]
        self.damping = float(damping)
        self.rot_weight = float(rot_weight)
        self.step = float(step)
        self.max_iters = int(max_iters)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q5):
        """Site position and 3x3 rotation for 5 arm joint angles."""
        self.data.qpos[self.qpos_i] = q5
        mujoco.mj_kinematics(self.model, self.data)
        return (self.data.site_xpos[self.site].copy(),
                self.data.site_xmat[self.site].reshape(3, 3).copy())

    def solve(self, q_init, target_pos, target_rot=None):
        """Return (q5, pos_err_m, rot_err_rad, iters).

        `target_rot` is a 3x3 rotation or None for position-only. `q_init` is
        the warm start; the expert passes the arm's current angles so the
        solution is the nearby one, which keeps successive actions smooth.
        """
        q = np.clip(np.asarray(q_init, float)[:5].copy(), self.lo, self.hi)
        target_pos = np.asarray(target_pos, float)
        pos_err = rot_err = np.inf
        it = 0
        for it in range(1, self.max_iters + 1):
            pos, R = self.fk(q)
            e_pos = target_pos - pos
            pos_err = float(np.linalg.norm(e_pos))
            if target_rot is None:
                rot_err = 0.0
                e = e_pos
            else:
                e_rot = rotation_error(R, target_rot)
                rot_err = float(np.linalg.norm(e_rot))
                e = np.concatenate([e_pos, self.rot_weight * e_rot])
            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
            Jp = self._jacp[:, self.qvel_i]
            if target_rot is None:
                J = Jp
            else:
                J = np.vstack([Jp, self.rot_weight * self._jacr[:, self.qvel_i]])
            JJt = J @ J.T + self.damping * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            # Limit the per-iteration move so the linearisation stays honest.
            n = np.linalg.norm(dq)
            if n > 0.5:
                dq *= 0.5 / n
            q = np.clip(q + self.step * dq, self.lo, self.hi)
        return q, pos_err, rot_err, it


def rotation_error(R_cur, R_des):
    """Small-angle rotation vector taking R_cur to R_des, world frame."""
    R_err = R_des @ R_cur.T
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R_err.reshape(-1))
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def top_down_rotation(close_yaw):
    """Gripper rotation for a top-down grasp whose jaws close along the
    horizontal direction at angle `close_yaw` (world, about +z).

    From the model probe: the site's local -y axis is the finger direction and
    its local x is the closing direction. So local y must point up (+z),
    local x lies in the table plane at `close_yaw`, and local z completes the
    right-handed frame.
    """
    x = np.array([np.cos(close_yaw), np.sin(close_yaw), 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)
