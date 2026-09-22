"""A scripted expert for every task, built on IK waypoints.

It exists for three reasons: it produces demonstrations for imitation
learning, it can be queried for a label at *any* state (which is what
DAgger needs and what a human teleoperator cannot give), and its success
rate on nominal and held-out physics is the reference every learned policy
is measured against. `scripts/expert_success.py` measures it.

It is not a controller a real robot would run -- it reads the cube's true
pose from the simulator. That is the privileged information a learned policy
has to recover from the image.

Phases per task. `GRASP_DZ` puts the site where the lower two pad pairs sit
level with the cube's middle and the fingertips clear the table.

    reach        one waypoint: REACH_HOVER above the cube, jaw open
    lift         pregrasp (high above) -> hover (fingertips 1 cm over the
                 cube, wait until laterally aligned) -> descend -> settle and
                 close until both jaws touch -> lift straight up
    push         behind the cube on the target line, raised -> descend ->
                 press the closed pads into the cube's near face and follow it
    pick_place   lift, then carry over the marker -> lower -> open -> retreat

Every waypoint is turned into joint targets by damped-least-squares IK warm
started from the current command, then rate limited to at most `speed` rad
per control step, so consecutive actions are smooth enough to learn from.

Two things a first draft did not do, both found by watching it fail:

  - The grasp's 180-degree ambiguity is resolved explicitly (`_solve`).
  - The position actuators are plain PD with no gravity compensation, so the
    arm settles 1-2 cm from the pose IK asked for, and more under a weakened
    kp. The expert integrates the Cartesian error and shifts its IK target,
    but ONLY while the rate limiter is inactive -- while the command is still
    slewing, the error is motion, not sag, and integrating it overshoots.
    The hover phase exists so that this correction has converged before the
    fingers pass the top of the cube; descending with a 1 cm lateral error
    lands a finger on the cube and shoves it away.
"""
import numpy as np

from . import env as E
from . import scene
from .ik import IK, top_down_rotation

GRASP_DZ = 0.0175         # site height above the cube centre when grasping
FINGER_BELOW_SITE = 0.0284  # lowest pad's bottom edge, measured below the site
HOVER_CLEAR = 0.010       # fingertips this far above the cube top in the hover phase
PREGRASP_DZ = 0.05
LIFT_DZ = 0.065           # site ends ~0.095 m up: 1.5 cm over the success line, inside reach
CARRY_DZ = 0.05
PLACE_DZ = 0.015
PUSH_STANDOFF = 0.045     # m behind the cube on the target line, site to cube centre
PUSH_PRESS = 0.008        # m the pad face is commanded into the cube face while pushing
PAD_FACE = 0.010          # m from the site to a closed pad's outer face (half the closed gap)
PUSH_DZ = 0.023           # site height above the cube centre while pushing; pad 3 just clears the top

WAYPOINT_TOL = 0.012
ALIGN_TOL = 0.004         # lateral alignment required before the final descent
DESCEND_TOL = 0.006
SETTLE = 3                # steps to let the PD catch up before the jaw closes
PHASE_TIMEOUT = 40        # steps; proceed anyway so a stuck phase cannot eat the episode
CLOSE_TIMEOUT = 25
RELEASE_WAIT = 5

CORR_GAIN = 0.15
CORR_MAX = 0.03

PHASES = {
    "reach": ["reach"],
    "push": ["p_high", "p_low", "p_push"],
    "lift": ["pregrasp", "hover", "descend", "close", "lift"],
    "pick_place": ["pregrasp", "hover", "descend", "close", "lift", "carry", "lower",
                   "release", "retreat"],
}


def fold_yaw(yaw):
    """Cube yaw folded into (-pi/4, pi/4]: a square is symmetric under 90 deg."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


class ScriptedExpert:
    def __init__(self, env, speed=None, seed=0):
        self.env = env
        self.ik = IK(env.model)
        self.speed = np.array(speed if speed is not None else E.DELTA_SCALE, float)
        self.rng = np.random.default_rng(seed)
        self._home = scene.home_qpos(env.model)
        self.phases = PHASES[env.task.kind]
        self.reset()

    def reset(self):
        self.phase = 0
        self.timer = 0
        self.q_cmd = self.env.joint_target().copy()
        self._grasp_xy = None
        self._grasp_yaw = None
        self._branch = None
        self._corr = np.zeros(3)
        self._slewing = True

    @property
    def phase_name(self):
        return self.phases[min(self.phase, len(self.phases) - 1)]

    def _next(self):
        self.phase = min(self.phase + 1, len(self.phases) - 1)
        self.timer = 0
        self._corr[:] = 0.0

    # -- waypoint for the current phase ----------------------------------

    def _waypoint(self):
        """(pos, rot, jaw) for the current phase."""
        env = self.env
        cube = env.cube_pos()
        half = env.physics.cube_half
        ee = env.ee_pos()
        name = self.phase_name
        yaw = fold_yaw(env.cube_yaw()) if self._grasp_yaw is None else self._grasp_yaw
        rot = top_down_rotation(yaw)
        grasp_z = half + GRASP_DZ

        if name == "reach":
            return np.array([cube[0], cube[1], half + E.REACH_HOVER]), rot, scene.JAW_OPEN

        if name.startswith("p_"):
            tgt = env.target_pos()[:2]
            d = tgt - cube[:2]
            n = np.linalg.norm(d)
            u = d / n if n > 1e-6 else np.array([0.0, -1.0])
            push_rot = top_down_rotation(np.arctan2(u[1], u[0]))
            z = half + PUSH_DZ
            if name == "p_high":
                p = cube[:2] - PUSH_STANDOFF * u
                return np.array([p[0], p[1], z + 0.04]), push_rot, scene.JAW_CLOSED
            if name == "p_low":
                p = cube[:2] - PUSH_STANDOFF * u
                return np.array([p[0], p[1], z]), push_rot, scene.JAW_CLOSED
            # Aim the pad face PUSH_PRESS inside the cube's near face, so the PD
            # keeps pressing as the cube moves; the waypoint follows the cube.
            p = cube[:2] - (half + PAD_FACE - PUSH_PRESS) * u
            if n < E.PUSH_TOL * 0.8:
                p = ee[:2]
            return np.array([p[0], p[1], z]), push_rot, scene.JAW_CLOSED

        if name == "pregrasp":
            return np.array([cube[0], cube[1], grasp_z + PREGRASP_DZ]), rot, scene.JAW_OPEN
        if name == "hover":
            z = 2 * half + FINGER_BELOW_SITE + HOVER_CLEAR
            return np.array([cube[0], cube[1], z]), rot, scene.JAW_OPEN
        if name == "descend":
            return np.array([cube[0], cube[1], grasp_z]), rot, scene.JAW_OPEN
        if name == "close":
            jaw = scene.JAW_OPEN if self.timer < SETTLE else scene.JAW_CLOSED
            return np.array([cube[0], cube[1], grasp_z]), rot, jaw
        gx, gy = self._grasp_xy if self._grasp_xy is not None else cube[:2]
        if name == "lift":
            return np.array([gx, gy, grasp_z + LIFT_DZ]), rot, scene.JAW_CLOSED
        tgt = env.target_pos()
        if name == "carry":
            return np.array([tgt[0], tgt[1], grasp_z + CARRY_DZ]), rot, scene.JAW_CLOSED
        if name == "lower":
            return np.array([tgt[0], tgt[1], grasp_z + PLACE_DZ]), rot, scene.JAW_CLOSED
        if name == "release":
            return np.array([tgt[0], tgt[1], grasp_z + PLACE_DZ]), rot, scene.JAW_OPEN
        return np.array([tgt[0], tgt[1], grasp_z + CARRY_DZ]), rot, scene.JAW_OPEN   # retreat

    def _advance(self, pos):
        """Phase transitions, evaluated after the step's waypoint was issued."""
        env = self.env
        name = self.phase_name
        err = env.ee_pos() - pos
        near = np.linalg.norm(err) < WAYPOINT_TOL
        aligned = np.linalg.norm(err[:2]) < ALIGN_TOL and abs(err[2]) < WAYPOINT_TOL
        tight = np.linalg.norm(err) < DESCEND_TOL
        timeout = self.timer >= PHASE_TIMEOUT
        self.timer += 1
        last = self.phase >= len(self.phases) - 1
        if last or name == "p_push":
            return
        if name in ("p_high", "p_low", "pregrasp", "lift", "carry", "lower") and (near or timeout):
            self._next()
        elif name == "hover" and ((aligned and not self._slewing) or timeout):
            self._next()
        elif name == "descend" and (tight or timeout):
            self._grasp_xy = env.cube_pos()[:2].copy()
            self._grasp_yaw = fold_yaw(env.cube_yaw())
            self._next()
        elif name == "close" and ((self.timer > SETTLE + 2 and env.grasped())
                                  or self.timer >= CLOSE_TIMEOUT):
            self._next()
        elif name == "release" and self.timer >= RELEASE_WAIT:
            self._next()

    # -- IK with the grasp ambiguity resolved ----------------------------

    def _solve(self, pos, rot):
        """A parallel jaw closing along yaw theta is the same grasp as one
        closing along theta + pi, but only one of the two may be inside the
        wrist-roll limit from where the arm currently is. Both are solved
        from the current command; if neither converges, both are retried from
        the home pose. Once a branch is chosen it is kept unless the other is
        clearly better, so the wrist does not flip mid-approach."""
        flipped = rot.copy()
        flipped[:, 0] *= -1.0
        flipped[:, 2] *= -1.0
        cands = [(0, rot), (1, flipped)]
        if self._branch is not None:
            cands.sort(key=lambda c: c[0] != self._branch)
        best = None
        for start in (self.q_cmd[:5], self._home[:5]):
            for k, R in cands:
                q5, pe, re_, _ = self.ik.solve(start, pos, R)
                score = pe + 0.02 * re_
                if self._branch is not None and k != self._branch:
                    score *= 1.5
                if best is None or score < best[0]:
                    best = (score, k, q5)
            if best[0] < 0.004:
                break
        self._branch = best[1]
        return best[2]

    # -- action ----------------------------------------------------------

    def act(self, obs=None):
        pos, rot, jaw = self._waypoint()
        if not self._slewing:
            err = pos - self.env.ee_pos()
            self._corr = np.clip(self._corr + CORR_GAIN * err, -CORR_MAX, CORR_MAX)
        q5 = self._solve(pos + self._corr, rot)
        delta = q5 - self.q_cmd[:5]
        self._slewing = bool(np.any(np.abs(delta) > self.speed[:5]))
        self.q_cmd[:5] += np.clip(delta, -self.speed[:5], self.speed[:5])
        self.q_cmd[5] += np.clip(jaw - self.q_cmd[5], -self.speed[5], self.speed[5])
        lo, hi = self.env.joint_limits
        self.q_cmd = np.clip(self.q_cmd, lo, hi)
        self._advance(pos)
        if self.env.action_mode == "absolute":
            return self.q_cmd.copy()
        return np.clip((self.q_cmd - self.env.joint_target()) / E.DELTA_SCALE, -1.0, 1.0)

    def joint_targets(self):
        """The absolute targets behind the last action, whatever the env's mode."""
        return self.q_cmd.copy()
