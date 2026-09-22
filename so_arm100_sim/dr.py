"""Domain randomisation, and the held-out physics it is evaluated against.

Two things live here and they must stay separate:

  DRConfig      the ranges a *training* run samples from at every reset
  EVAL_PHYSICS  a small set of fixed physics shifts a *trained* policy is
                evaluated on, each of them OUTSIDE the DR range in at least
                one factor, so that "success under EVAL_PHYSICS" is a
                transfer claim rather than an interpolation claim.
                `test_dr.py` asserts the outside-the-range property.

Factors and how they are applied to the compiled model:

  cube_mass       body_mass and body_inertia of the active cube, x factor
  cube_friction   geom_friction[0] (sliding) of the active cube, x factor
  cube_size       geom_size of the cube, absolute half-extent in metres
                  (inertia rescaled as size^2, mass kept)
  kp              position-actuator gain on all six actuators: gainprm[0]
                  and biasprm[1] together, x factor. biasprm[2] (the
                  velocity feedback derived from dampratio at compile time)
                  is left alone, so a lower kp also means a more damped servo,
                  which is what a weaker motor does.
  damping         dof_damping of the six arm joints, x factor
  frictionloss    dof_frictionloss of the six arm joints, x factor
  latency         integer control steps the commanded action is delayed by;
                  handled in the environment's action buffer, not the model
  obs_noise_q     std of Gaussian noise added to observed joint angles, rad
  obs_noise_pos   std of noise on observed object / target positions, m

Nominal values are captured from the model once, at construction, so
`apply` is always relative to the vendored model and never compounds.
"""
from dataclasses import dataclass, asdict, field

import numpy as np

from . import scene


@dataclass
class DRConfig:
    # multiplicative ranges (lo, hi) on the nominal value
    cube_mass: tuple = (0.5, 2.0)
    cube_friction: tuple = (0.5, 1.5)
    kp: tuple = (0.6, 1.4)
    damping: tuple = (0.5, 2.0)
    frictionloss: tuple = (0.5, 2.0)
    # absolute ranges
    cube_size: tuple = (0.010, 0.015)       # m, half extent
    latency: tuple = (0, 2)                 # control steps, inclusive
    obs_noise_q: float = 0.005              # rad, applied with prob 1 when > 0
    obs_noise_pos: float = 0.003            # m

    @classmethod
    def none(cls):
        """Nominal physics: every range collapsed onto the vendored value."""
        return cls(cube_mass=(1.0, 1.0), cube_friction=(1.0, 1.0), kp=(1.0, 1.0),
                   damping=(1.0, 1.0), frictionloss=(1.0, 1.0),
                   cube_size=(scene.CUBE_HALF, scene.CUBE_HALF), latency=(0, 0),
                   obs_noise_q=0.0, obs_noise_pos=0.0)

    def sample(self, rng):
        """One draw of concrete factors, as a plain dict `Physics.apply` takes."""
        return {
            "cube_mass": float(rng.uniform(*self.cube_mass)),
            "cube_friction": float(rng.uniform(*self.cube_friction)),
            "kp": float(rng.uniform(*self.kp)),
            "damping": float(rng.uniform(*self.damping)),
            "frictionloss": float(rng.uniform(*self.frictionloss)),
            "cube_size": float(rng.uniform(*self.cube_size)),
            "latency": int(rng.integers(self.latency[0], self.latency[1] + 1)),
            "obs_noise_q": float(self.obs_noise_q),
            "obs_noise_pos": float(self.obs_noise_pos),
        }

    def to_dict(self):
        return asdict(self)


NOMINAL_FACTORS = {
    "cube_mass": 1.0, "cube_friction": 1.0, "kp": 1.0, "damping": 1.0,
    "frictionloss": 1.0, "cube_size": scene.CUBE_HALF, "latency": 0,
    "obs_noise_q": 0.0, "obs_noise_pos": 0.0,
}

# Held-out evaluation physics. Each entry overrides NOMINAL_FACTORS and sits
# outside DRConfig()'s default range in the named factor. "nominal" is the
# in-distribution reference every other row is compared against.
EVAL_PHYSICS = {
    "nominal":  {},
    "heavy":    {"cube_mass": 3.0},         # DR tops out at 2.0
    "slippery": {"cube_friction": 0.3},     # DR floors at 0.5
    "weak":     {"kp": 0.45},               # DR floors at 0.6
    "laggy":    {"latency": 3},             # DR tops out at 2
    "noisy":    {"obs_noise_q": 0.015, "obs_noise_pos": 0.009},   # 3x DR noise
    "small":    {"cube_size": 0.009},       # DR floors at 0.010
}


def resolve_physics(spec):
    """str name from EVAL_PHYSICS, or a dict of overrides, or None -> factors dict."""
    if spec is None:
        return dict(NOMINAL_FACTORS)
    if isinstance(spec, str):
        if spec not in EVAL_PHYSICS:
            raise KeyError(f"unknown physics {spec!r}; have {sorted(EVAL_PHYSICS)}")
        out = dict(NOMINAL_FACTORS)
        out.update(EVAL_PHYSICS[spec])
        return out
    out = dict(NOMINAL_FACTORS)
    unknown = set(spec) - set(out)
    if unknown:
        raise KeyError(f"unknown physics factors {sorted(unknown)}")
    out.update(spec)
    return out


class Physics:
    """Applies a factors dict to a compiled model, relative to captured nominals."""

    def __init__(self, model, color):
        self.model = model
        self.color = color
        self.body = scene.cube_body(model, color)
        self.geom = scene.cube_geom(model, color)
        self.dofs = scene.arm_qvel_index(model)
        self.nominal = {
            "body_mass": float(model.body_mass[self.body]),
            "body_inertia": model.body_inertia[self.body].copy(),
            "geom_friction": model.geom_friction[self.geom].copy(),
            "geom_size": model.geom_size[self.geom].copy(),
            "gainprm0": model.actuator_gainprm[:, 0].copy(),
            "biasprm1": model.actuator_biasprm[:, 1].copy(),
            "dof_damping": model.dof_damping[self.dofs].copy(),
            "dof_frictionloss": model.dof_frictionloss[self.dofs].copy(),
        }
        self.current = dict(NOMINAL_FACTORS)

    def apply(self, factors):
        f = dict(NOMINAL_FACTORS)
        f.update(factors)
        m, n = self.model, self.nominal
        size_ratio = f["cube_size"] / scene.CUBE_HALF
        m.body_mass[self.body] = n["body_mass"] * f["cube_mass"]
        m.body_inertia[self.body] = n["body_inertia"] * f["cube_mass"] * size_ratio ** 2
        m.geom_friction[self.geom] = n["geom_friction"]
        m.geom_friction[self.geom, 0] = n["geom_friction"][0] * f["cube_friction"]
        m.geom_size[self.geom] = f["cube_size"]
        m.actuator_gainprm[:, 0] = n["gainprm0"] * f["kp"]
        m.actuator_biasprm[:, 1] = n["biasprm1"] * f["kp"]
        m.dof_damping[self.dofs] = n["dof_damping"] * f["damping"]
        m.dof_frictionloss[self.dofs] = n["dof_frictionloss"] * f["frictionloss"]
        self.current = f
        return f

    def restore(self):
        return self.apply(NOMINAL_FACTORS)

    @property
    def latency(self):
        return int(self.current["latency"])

    @property
    def obs_noise_q(self):
        return float(self.current["obs_noise_q"])

    @property
    def obs_noise_pos(self):
        return float(self.current["obs_noise_pos"])

    @property
    def cube_half(self):
        return float(self.current["cube_size"])
