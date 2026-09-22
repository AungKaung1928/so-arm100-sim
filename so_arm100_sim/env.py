"""The environment contract: four tabletop tasks on one arm, one surface.

    obs, reward, done, info = env.step(action)

`obs` is always a dict with a float32 `"state"` vector, plus `"image"`
(H, W, 3 uint8) when the env was built with `image=...`. Two state layouts:

    obs_mode="state"    25 dims, privileged. Joint angles and velocities, the
                        end-effector position, the active cube's position,
                        cube minus end effector, the target position and the
                        jaw gap. This is what a state-based RL policy trains
                        on and what the scripted expert conceptually has.
    obs_mode="proprio"  15 dims, what the real arm knows about itself: joint
                        angles, joint velocities, end-effector position (from
                        its own kinematics). Objects and targets have to come
                        from the image. Image policies use this one.

Two action conventions, both 6-dimensional (five arm joints + jaw):

    action_mode="delta"     in [-1, 1], scaled by DELTA_SCALE and added to
                            the previous commanded joint target. Natural for
                            RL: a zero action holds still.
    action_mode="absolute"  joint targets in radians, clipped to the limits.
                            Natural for imitation: it is what a recorded
                            dataset stores and what a real SO-ARM100 takes.

Control runs at 20 Hz over the model's 2 ms physics step (25 substeps). After
the substeps `mj_forward` is called once so every quantity read for the
observation and reward belongs to the same instant -- `mj_step` integrates
positions to t+1 and leaves site and body poses at t otherwise.

Episodes end on success (`terminated`) or on the step limit (`truncated`);
both flags travel in `info` because a value bootstrap has to tell them
apart. Per-task limits are in EPISODE_STEPS.

Success criteria are geometric and stated once, here, so that the RL, IL and
language projects report the same number:

    reach       ee within REACH_TOL of a point REACH_HOVER above the cube
    push        cube's xy within PUSH_TOL of the target marker
    lift        cube centre LIFT_HEIGHT above its resting height AND both
                jaws in contact with it (a flung cube does not count)
    pick_place  cube xy within PLACE_TOL of the marker, resting on the table,
                and released
"""
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from . import scene
from .dr import DRConfig, Physics, resolve_physics

TASKS = ("reach", "push", "lift", "pick_place")
OBS_MODES = ("state", "proprio")
ACTION_MODES = ("delta", "absolute")

CONTROL_HZ = 20
SUBSTEPS = int(round(1.0 / CONTROL_HZ / scene.PHYSICS_DT))     # 25
CONTROL_DT = 1.0 / CONTROL_HZ

EPISODE_STEPS = {"reach": 100, "push": 150, "lift": 150, "pick_place": 250}

# rad per control step at a saturated delta action. 0.05 rad/step = 1 rad/s.
DELTA_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.10])
QVEL_SCALE = 0.1
INIT_NOISE = 0.02          # rad, on the home pose at reset

REACH_HOVER = 0.03         # m above the cube centre
REACH_TOL = 0.02
PUSH_TOL = 0.025
PUSH_DIST = (0.06, 0.12)   # target distance from the cube at reset
LIFT_HEIGHT = 0.05         # m above the resting cube height; a top-down pose is
                           # only IK-reachable to ~0.09 m at the far edge of the spawn box
PLACE_TOL = 0.025
PLACE_MIN_SEP = 0.06       # target at least this far from the cube at reset
DISTRACTOR_MIN_SEP = 0.06
CUBE_YAW = np.pi / 4       # initial yaw drawn uniformly in [-CUBE_YAW, CUBE_YAW]

STATE_LAYOUT = {
    "qpos":       slice(0, 6),
    "qvel":       slice(6, 12),
    "ee_pos":     slice(12, 15),
    "cube_pos":   slice(15, 18),
    "cube_rel":   slice(18, 21),
    "target_pos": slice(21, 24),
    "jaw_gap":    slice(24, 25),
}
STATE_DIM = 25
PROPRIO_LAYOUT = {"qpos": slice(0, 6), "qvel": slice(6, 12), "ee_pos": slice(12, 15)}
PROPRIO_DIM = 15
ACT_DIM = 6

REWARD_WEIGHTS = {
    "dist": 1.0,          # reach: -distance to the hover point
    "reach": 0.5,         # push / lift / place: -distance ee to cube
    "goal": 1.0,          # push / place: -xy distance cube to marker
    "grasp": 0.25,        # lift / place: both jaws touching the cube
    "height": 2.0,        # lift: fraction of LIFT_HEIGHT achieved, while grasped
    "action_rate": 0.01,  # all: mean squared change of the (delta) action
    "success": 1.0,
}


@dataclass(frozen=True)
class TaskSpec:
    kind: str = "reach"
    color: str = "red"
    distractors: bool = False      # other two cubes on the table too

    @classmethod
    def parse(cls, s):
        """'lift', 'lift:blue', 'lift:blue:distractors'."""
        if isinstance(s, TaskSpec):
            return s
        parts = str(s).split(":")
        kind = parts[0]
        color = parts[1] if len(parts) > 1 and parts[1] else "red"
        distractors = len(parts) > 2 and parts[2] == "distractors"
        if kind not in TASKS:
            raise KeyError(f"unknown task {kind!r}; have {TASKS}")
        if color not in scene.CUBE_COLORS:
            raise KeyError(f"unknown colour {color!r}; have {scene.CUBE_COLORS}")
        return cls(kind, color, distractors)

    def __str__(self):
        s = f"{self.kind}:{self.color}"
        return s + ":distractors" if self.distractors else s


class ArmEnv:
    def __init__(self, task="reach", obs_mode="state", action_mode="delta", seed=0,
                 dr=None, physics=None, image=None, episode_steps=None,
                 terminate_on_success=True):
        """
        task        TaskSpec or its string form
        dr          DRConfig sampled at every reset, or None
        physics     a name from EVAL_PHYSICS or a factors dict; fixed for every
                    episode. Ignored when `dr` is given.
        image       None, or dict(camera="front", height=96, width=128)
        """
        self.task = TaskSpec.parse(task)
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        self.obs_mode, self.action_mode = obs_mode, action_mode
        self.episode_steps = int(episode_steps or EPISODE_STEPS[self.task.kind])
        self.terminate_on_success = bool(terminate_on_success)
        self.dr = dr if dr is None or isinstance(dr, DRConfig) else DRConfig(**dr)
        self.fixed_physics = resolve_physics(physics)
        self.image_cfg = dict(camera="front", height=96, width=128)
        if image:
            self.image_cfg.update(image)
        self.image = bool(image)

        self.model = scene.build_model(3)
        self.data = mujoco.MjData(self.model)
        self.physics = Physics(self.model, self.task.color)

        self._qpos_i = scene.arm_qpos_index(self.model)
        self._qvel_i = scene.arm_qvel_index(self.model)
        self._lo, self._hi = scene.joint_limits(self.model)
        self._home = scene.home_qpos(self.model)
        self._ee = scene.ee_site_id(self.model)
        self._target_site = scene.target_site_id(self.model)
        self._cube_q = {c: scene.cube_qpos_index(self.model, c) for c in scene.CUBE_COLORS}
        self._cube_v = {c: scene.cube_qvel_index(self.model, c) for c in scene.CUBE_COLORS}
        self._cube_body = {c: scene.cube_body(self.model, c) for c in scene.CUBE_COLORS}
        self._cube_geom = {c: scene.cube_geom(self.model, c) for c in scene.CUBE_COLORS}
        self._pads_fixed, self._pads_moving = scene.pad_geoms(self.model)
        self._pad3 = (scene.name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_3"),
                      scene.name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_3"))

        self.rng = np.random.default_rng(seed)
        self._renderer = None
        self._target_q = self._home.copy()
        self._prev_action = np.zeros(ACT_DIM)
        self._buf = deque()
        self._target = np.zeros(3)
        self._t = 0
        self._started = False
        self._success_ever = False
        self._grasped_ever = False

    # -- reset -----------------------------------------------------------

    def _sample_xy(self, avoid=(), min_sep=DISTRACTOR_MIN_SEP, tries=50):
        for _ in range(tries):
            xy = np.array([self.rng.uniform(*scene.SPAWN_X), self.rng.uniform(*scene.SPAWN_Y)])
            if all(np.linalg.norm(xy - a) >= min_sep for a in avoid):
                return xy
        return xy

    def _place_cube(self, color, xy, yaw, half):
        i = self._cube_q[color]
        q = np.zeros(7)
        q[:2] = xy
        q[2] = half
        mujoco.mju_axisAngle2Quat(q[3:], np.array([0.0, 0.0, 1.0]), float(yaw))
        self.data.qpos[i:i + 7] = q
        self.data.qvel[self._cube_v[color]:self._cube_v[color] + 6] = 0.0

    def _park_cube(self, color):
        k = scene.CUBE_COLORS.index(color)
        self._place_cube(color, np.array(scene.PARK_XY[k]), 0.0, scene.CUBE_HALF)

    def _set_target(self, pos, visible):
        self._target = np.asarray(pos, float).copy()
        p = self._target.copy() if visible else np.array([scene.PARK_XY[0][0],
                                                          scene.PARK_XY[0][1] + 0.5, 0.0005])
        if visible:
            p[2] = 0.0005
        self.model.site_pos[self._target_site] = p

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        factors = self.dr.sample(self.rng) if self.dr is not None else dict(self.fixed_physics)
        self.physics.apply(factors)
        half = self.physics.cube_half

        mujoco.mj_resetData(self.model, self.data)
        q0 = self._home + self.rng.uniform(-INIT_NOISE, INIT_NOISE, ACT_DIM)
        q0 = np.clip(q0, self._lo, self._hi)
        self.data.qpos[self._qpos_i] = q0
        self.data.qvel[self._qvel_i] = 0.0

        color = self.task.color
        cube_xy = self._sample_xy()
        yaw = self.rng.uniform(-CUBE_YAW, CUBE_YAW)
        self._place_cube(color, cube_xy, yaw, half)
        placed = [cube_xy]
        for c in scene.CUBE_COLORS:
            if c == color:
                continue
            if self.task.distractors:
                xy = self._sample_xy(avoid=placed)
                self._place_cube(c, xy, self.rng.uniform(-CUBE_YAW, CUBE_YAW), scene.CUBE_HALF)
                placed.append(xy)
            else:
                self._park_cube(c)

        kind = self.task.kind
        if kind == "reach":
            self._set_target(np.array([*cube_xy, half + REACH_HOVER]), visible=False)
        elif kind == "push":
            for _ in range(50):
                d = self.rng.uniform(*PUSH_DIST)
                th = self.rng.uniform(0.0, 2 * np.pi)
                t = cube_xy + d * np.array([np.cos(th), np.sin(th)])
                if (scene.SPAWN_X[0] <= t[0] <= scene.SPAWN_X[1]
                        and scene.SPAWN_Y[0] <= t[1] <= scene.SPAWN_Y[1]):
                    break
            self._set_target(np.array([*t, 0.0]), visible=True)
        elif kind == "lift":
            self._set_target(np.array([*cube_xy, half + LIFT_HEIGHT]), visible=False)
        else:   # pick_place
            t = self._sample_xy(avoid=placed, min_sep=PLACE_MIN_SEP)
            self._set_target(np.array([*t, 0.0]), visible=True)

        self._target_q = q0.copy()
        self.data.ctrl[:] = q0
        # `latency` stale commands ahead of the fresh one. An earlier version
        # held latency + 1 and so applied EVERY command one step late, at
        # latency 0 included; `test_kp_affects_tracking` caught it because a
        # step command produced no motion within the step that issued it.
        self._buf = deque([q0.copy() for _ in range(self.physics.latency)])
        self._prev_action = np.zeros(ACT_DIM)
        self._t = 0
        self._success_ever = False
        self._grasped_ever = False
        self._started = True
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    # -- step ------------------------------------------------------------

    def step(self, action):
        if not self._started:
            raise RuntimeError("call reset() before step()")
        a = np.asarray(action, dtype=float).reshape(ACT_DIM)
        if self.action_mode == "delta":
            a = np.clip(a, -1.0, 1.0)
            target = self._target_q + a * DELTA_SCALE
        else:
            target = a
        target = np.clip(target, self._lo, self._hi)
        self._target_q = target

        # Latency: the command applied now is the one issued `latency` steps ago.
        self._buf.append(target.copy())
        applied = self._buf.popleft()
        self.data.ctrl[:] = applied

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        grasped = self.grasped()
        self._grasped_ever = self._grasped_ever or grasped
        success = self._success(grasped)
        self._success_ever = self._success_ever or success
        terms = self._reward_terms(a, grasped, success)
        reward = float(sum(terms.values()))
        if self.action_mode == "delta":
            self._prev_action = a
        self._t += 1

        terminated = bool(success and self.terminate_on_success)
        truncated = bool(self._t >= self.episode_steps and not terminated)
        done = terminated or truncated
        info = {
            "success": bool(success), "success_ever": bool(self._success_ever),
            "terms": terms, "t": self._t, "terminated": terminated, "truncated": truncated,
            "grasped": bool(grasped), "ee_pos": self.ee_pos(), "cube_pos": self.cube_pos(),
            "target_pos": self._target.copy(), "physics": dict(self.physics.current),
        }
        return self.observe(), reward, done, info

    # -- task logic ------------------------------------------------------

    def _success(self, grasped):
        kind = self.task.kind
        cube = self.cube_pos()
        if kind == "reach":
            return np.linalg.norm(self.ee_pos() - self._target) < REACH_TOL
        if kind == "push":
            return np.linalg.norm(cube[:2] - self._target[:2]) < PUSH_TOL
        if kind == "lift":
            return grasped and cube[2] > self.physics.cube_half + LIFT_HEIGHT
        # pick_place
        return (np.linalg.norm(cube[:2] - self._target[:2]) < PLACE_TOL
                and cube[2] < self.physics.cube_half + 0.01 and not grasped)

    def _reward_terms(self, a, grasped, success):
        w = REWARD_WEIGHTS
        kind = self.task.kind
        ee, cube = self.ee_pos(), self.cube_pos()
        d_ec = float(np.linalg.norm(ee - cube))
        terms = {}
        if kind == "reach":
            terms["dist"] = -w["dist"] * float(np.linalg.norm(ee - self._target))
        elif kind == "push":
            terms["reach"] = -w["reach"] * d_ec
            terms["goal"] = -w["goal"] * float(np.linalg.norm(cube[:2] - self._target[:2]))
        elif kind == "lift":
            terms["reach"] = -w["reach"] * d_ec
            terms["grasp"] = w["grasp"] * float(grasped)
            frac = np.clip((cube[2] - self.physics.cube_half) / LIFT_HEIGHT, 0.0, 1.0)
            terms["height"] = w["height"] * float(frac) * float(grasped)
        else:
            terms["reach"] = -w["reach"] * d_ec * (0.0 if self._grasped_ever else 1.0)
            terms["grasp"] = w["grasp"] * float(grasped)
            terms["goal"] = -w["goal"] * float(np.linalg.norm(cube[:2] - self._target[:2]))
        if self.action_mode == "delta":
            terms["action_rate"] = -w["action_rate"] * float(np.mean((a - self._prev_action) ** 2))
        terms["success"] = w["success"] * float(success)
        return terms

    # -- observation -----------------------------------------------------

    def observe(self):
        q = self.data.qpos[self._qpos_i].copy()
        dq = self.data.qvel[self._qvel_i] * QVEL_SCALE
        nq, npos = self.physics.obs_noise_q, self.physics.obs_noise_pos
        if nq > 0:
            q = q + self.rng.normal(0.0, nq, ACT_DIM)
        ee = self.ee_pos()
        if self.obs_mode == "proprio":
            state = np.concatenate([q, dq, ee])
        else:
            cube, target = self.cube_pos(), self._target.copy()
            if npos > 0:
                cube = cube + self.rng.normal(0.0, npos, 3)
                target = target + self.rng.normal(0.0, npos, 3)
            state = np.concatenate([q, dq, ee, cube, cube - ee, target, [self.jaw_gap()]])
        obs = {"state": state.astype(np.float32)}
        if self.image:
            obs["image"] = self.render(**self.image_cfg)
        return obs

    def render(self, camera="front", height=96, width=128):
        from .render import Renderer
        if (self._renderer is None or self._renderer.height != height
                or self._renderer.width != width):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = Renderer(self.model, height, width)
        return self._renderer.render(self.data, camera)

    # -- privileged accessors (expert, tests, evaluation) ----------------

    def arm_qpos(self):
        return self.data.qpos[self._qpos_i].copy()

    def joint_target(self):
        """The joint targets currently commanded (before latency)."""
        return self._target_q.copy()

    def ee_pos(self):
        return self.data.site_xpos[self._ee].copy()

    def ee_rot(self):
        return self.data.site_xmat[self._ee].reshape(3, 3).copy()

    def cube_pos(self, color=None):
        return self.data.xpos[self._cube_body[color or self.task.color]].copy()

    def cube_yaw(self, color=None):
        i = self._cube_q[color or self.task.color]
        w, x, y, z = self.data.qpos[i + 3:i + 7]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def target_pos(self):
        return self._target.copy()

    def jaw_gap(self):
        a, b = self._pad3
        return float(np.linalg.norm(self.data.geom_xpos[a] - self.data.geom_xpos[b]))

    def grasped(self, color=None):
        """Both jaws in contact with the cube."""
        g = self._cube_geom[color or self.task.color]
        fixed = moving = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            other = c.geom2 if c.geom1 == g else (c.geom1 if c.geom2 == g else -1)
            if other < 0:
                continue
            if other in self._pads_fixed:
                fixed = True
            elif other in self._pads_moving:
                moving = True
            if fixed and moving:
                return True
        return False

    @property
    def t(self):
        return self._t

    @property
    def joint_limits(self):
        return self._lo.copy(), self._hi.copy()

    @property
    def state_dim(self):
        return STATE_DIM if self.obs_mode == "state" else PROPRIO_DIM

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
