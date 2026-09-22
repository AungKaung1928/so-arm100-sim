"""IK and the scripted expert. These are real success-rate gates, not smoke:
if the expert regresses, the demonstrations and every reference number in
the family regress with it. Thresholds are set below the measured rates in
`runs/expert_success.json` so the tests fail on a regression, not on noise."""
import numpy as np
import pytest

from so_arm100_sim import scene
from so_arm100_sim.env import ArmEnv
from so_arm100_sim.expert import ScriptedExpert, fold_yaw
from so_arm100_sim.ik import IK, top_down_rotation, rotation_error
from so_arm100_sim.evaluate import run_episode

N_EPISODES = 20
GATES = {"reach": 0.95, "lift": 0.75, "push": 0.65, "pick_place": 0.75}


def test_fold_yaw():
    for y in np.linspace(-3, 3, 61):
        f = fold_yaw(y)
        assert -np.pi / 4 - 1e-9 <= f <= np.pi / 4 + 1e-9
        assert abs(np.sin(4 * y) - np.sin(4 * f)) < 1e-9     # same square orientation


def test_top_down_rotation_is_orthonormal_and_points_down():
    for yaw in (0.0, 0.7, -1.2, 3.0):
        R = top_down_rotation(yaw)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) > 0
        assert np.allclose(-R[:, 1], [0, 0, -1])
        assert np.allclose(R[:, 0], [np.cos(yaw), np.sin(yaw), 0])


def test_rotation_error_zero_for_identity_and_small_for_small():
    R = top_down_rotation(0.3)
    assert np.allclose(rotation_error(R, R), 0.0)
    e = rotation_error(top_down_rotation(0.0), top_down_rotation(0.1))
    assert abs(np.linalg.norm(e) - 0.1) < 1e-6


def test_ik_reaches_random_points_in_the_spawn_region():
    model = scene.build_model(1)
    ik = IK(model)
    rng = np.random.default_rng(0)
    home = scene.home_qpos(model)[:5]
    fails = 0
    for _ in range(30):
        # heights the tasks actually use: grasp (0.03) up to the lift hold (0.09);
        # tools/reach_map.py shows top-down poses stop being solvable above that
        p = np.array([rng.uniform(*scene.SPAWN_X), rng.uniform(*scene.SPAWN_Y),
                      rng.uniform(0.03, 0.09)])
        R = top_down_rotation(rng.uniform(-np.pi / 4, np.pi / 4))
        q, pe, re_, _ = ik.solve(home, p, R)
        if pe > 3e-3 or re_ > 0.15:
            # the mirrored grasp is the same grasp; accept either branch
            Rf = R.copy()
            Rf[:, 0] *= -1
            Rf[:, 2] *= -1
            q, pe, re_, _ = ik.solve(home, p, Rf)
        if pe > 3e-3 or re_ > 0.15:
            fails += 1
    assert fails <= 2, f"{fails}/30 IK targets unreachable within tolerance"


def test_ik_result_respects_joint_limits():
    model = scene.build_model(1)
    ik = IK(model)
    lo, hi = scene.joint_limits(model)
    q, _, _, _ = ik.solve(scene.home_qpos(model)[:5], [0.1, -0.25, 0.05], top_down_rotation(0.2))
    assert np.all(q >= lo[:5] - 1e-9) and np.all(q <= hi[:5] + 1e-9)


@pytest.mark.parametrize("kind", ["reach", "lift", "push", "pick_place"])
def test_expert_success_gate(kind):
    env = ArmEnv(f"{kind}:red", action_mode="absolute", seed=0)
    ex = ScriptedExpert(env)
    res = [run_episode(env, ex, seed=5000 + i) for i in range(N_EPISODES)]
    rate = np.mean([r["success"] for r in res])
    assert rate >= GATES[kind], f"{kind}: expert success {rate:.2f} < gate {GATES[kind]}"


def test_expert_actions_are_smooth_and_in_range():
    env = ArmEnv("lift:blue", action_mode="absolute", seed=1)
    ex = ScriptedExpert(env)
    env.reset(seed=11)
    ex.reset()
    lo, hi = env.joint_limits
    prev = env.joint_target()
    for _ in range(60):
        a = ex.act()
        assert np.all(a >= lo - 1e-9) and np.all(a <= hi + 1e-9)
        assert np.all(np.abs(a - prev) <= ex.speed + 1e-9)
        prev = a
        _, _, done, _ = env.step(a)
        if done:
            break


def test_expert_in_delta_mode_matches_absolute_targets():
    env = ArmEnv("reach", action_mode="delta", seed=2)
    ex = ScriptedExpert(env)
    env.reset(seed=3)
    ex.reset()
    for _ in range(15):
        a = ex.act()
        assert np.all(np.abs(a) <= 1.0 + 1e-9)
        env.step(a)
        assert np.allclose(env.joint_target(), ex.joint_targets(), atol=1e-9)
