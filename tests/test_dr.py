"""Domain randomisation: applied exactly, restored exactly, and the held-out
physics really is outside the training range."""
import numpy as np
import pytest

from so_arm100_sim import scene
from so_arm100_sim.dr import (DRConfig, EVAL_PHYSICS, NOMINAL_FACTORS, Physics,
                              resolve_physics)
from so_arm100_sim.env import ArmEnv


def test_none_config_samples_nominal():
    f = DRConfig.none().sample(np.random.default_rng(0))
    assert f == NOMINAL_FACTORS


def test_samples_within_ranges():
    cfg = DRConfig()
    rng = np.random.default_rng(1)
    for _ in range(200):
        f = cfg.sample(rng)
        for k in ("cube_mass", "cube_friction", "kp", "damping", "frictionloss", "cube_size"):
            lo, hi = getattr(cfg, k)
            assert lo <= f[k] <= hi, k
        assert cfg.latency[0] <= f["latency"] <= cfg.latency[1]
        assert isinstance(f["latency"], int)


def test_eval_physics_is_held_out():
    """Every non-nominal entry overrides at least one factor to a value the
    default DRConfig can never sample. That is what makes it a transfer test."""
    cfg = DRConfig()
    for name, over in EVAL_PHYSICS.items():
        if name == "nominal":
            assert over == {}
            continue
        outside = False
        for k, v in over.items():
            if k in ("obs_noise_q", "obs_noise_pos"):
                outside |= v > getattr(cfg, k)
            else:
                lo, hi = getattr(cfg, k)
                outside |= not (lo <= v <= hi)
        assert outside, f"{name} is inside the DR range: {over}"


def test_resolve_physics():
    assert resolve_physics(None) == NOMINAL_FACTORS
    assert resolve_physics("heavy")["cube_mass"] == 3.0
    assert resolve_physics({"kp": 0.7})["kp"] == 0.7
    with pytest.raises(KeyError):
        resolve_physics("zero_g")
    with pytest.raises(KeyError):
        resolve_physics({"gravity": 0.0})


def test_apply_and_restore_are_exact():
    m = scene.build_model(3)
    snap = {k: getattr(m, k).copy() for k in
            ("body_mass", "body_inertia", "geom_friction", "geom_size",
             "actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_frictionloss")}
    ph = Physics(m, "red")
    f = DRConfig().sample(np.random.default_rng(3))
    ph.apply(f)
    body, geom = scene.cube_body(m, "red"), scene.cube_geom(m, "red")
    assert np.isclose(m.body_mass[body], scene.CUBE_MASS * f["cube_mass"])
    assert np.isclose(m.geom_friction[geom, 0], snap["geom_friction"][geom, 0] * f["cube_friction"])
    assert np.allclose(m.geom_size[geom], f["cube_size"])
    assert np.allclose(m.actuator_gainprm[:, 0], snap["actuator_gainprm"][:, 0] * f["kp"])
    assert np.allclose(m.actuator_biasprm[:, 1], snap["actuator_biasprm"][:, 1] * f["kp"])
    # the velocity feedback term is untouched on purpose
    assert np.allclose(m.actuator_biasprm[:, 2], snap["actuator_biasprm"][:, 2])
    # other cubes are untouched
    for c in ("green", "blue"):
        assert m.body_mass[scene.cube_body(m, c)] == snap["body_mass"][scene.cube_body(m, c)]
    ph.restore()
    for k, v in snap.items():
        assert np.array_equal(getattr(m, k), v), k


def test_apply_does_not_compound():
    m = scene.build_model(3)
    ph = Physics(m, "blue")
    for _ in range(5):
        ph.apply({"kp": 2.0})
    assert np.allclose(m.actuator_gainprm[:, 0], 100.0)


def test_env_with_dr_changes_physics_each_reset():
    env = ArmEnv("lift", seed=0, dr=DRConfig())
    seen = set()
    for _ in range(5):
        env.reset()
        seen.add(round(env.physics.current["kp"], 6))
    assert len(seen) == 5


def test_env_fixed_physics_is_constant():
    env = ArmEnv("lift", seed=0, physics="weak")
    for _ in range(3):
        env.reset()
        assert env.physics.current["kp"] == 0.45
        assert env.physics.latency == 0


def test_kp_affects_tracking():
    """A weaker servo tracks a step command more slowly. Direction check."""
    def settle(phys):
        e = ArmEnv("reach", action_mode="absolute", seed=0, physics=phys)
        e.reset(seed=1)
        q0 = e.arm_qpos()
        tgt = q0.copy()
        tgt[0] += 0.4
        e.step(tgt)
        return abs(e.arm_qpos()[0] - q0[0])
    assert settle("weak") < settle("nominal")
