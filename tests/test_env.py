"""The environment contract. Every expected value is derived, not recorded."""
import numpy as np
import pytest

from so_arm100_sim import env as E
from so_arm100_sim import scene
from so_arm100_sim.env import ArmEnv, TaskSpec


def test_taskspec_parse():
    assert TaskSpec.parse("lift") == TaskSpec("lift", "red", False)
    assert TaskSpec.parse("push:blue") == TaskSpec("push", "blue", False)
    assert TaskSpec.parse("reach:green:distractors").distractors
    assert str(TaskSpec.parse("lift:blue:distractors")) == "lift:blue:distractors"
    with pytest.raises(KeyError):
        TaskSpec.parse("fly")
    with pytest.raises(KeyError):
        TaskSpec.parse("lift:purple")


@pytest.mark.parametrize("kind", E.TASKS)
def test_obs_layout_state(kind):
    env = ArmEnv(kind, obs_mode="state", seed=1)
    obs = env.reset()
    s = obs["state"]
    assert s.dtype == np.float32 and s.shape == (E.STATE_DIM,)
    assert set(obs) == {"state"}
    # rebuild from the declared sources
    q = env.arm_qpos()
    assert np.allclose(s[E.STATE_LAYOUT["qpos"]], q, atol=1e-6)
    assert np.allclose(s[E.STATE_LAYOUT["ee_pos"]], env.ee_pos(), atol=1e-6)
    assert np.allclose(s[E.STATE_LAYOUT["cube_pos"]], env.cube_pos(), atol=1e-6)
    assert np.allclose(s[E.STATE_LAYOUT["cube_rel"]], env.cube_pos() - env.ee_pos(), atol=1e-6)
    assert np.allclose(s[E.STATE_LAYOUT["target_pos"]], env.target_pos(), atol=1e-6)
    assert abs(s[E.STATE_LAYOUT["jaw_gap"]][0] - env.jaw_gap()) < 1e-6
    # every slice covers the vector exactly once
    covered = sorted((sl.start, sl.stop) for sl in E.STATE_LAYOUT.values())
    assert covered[0][0] == 0 and covered[-1][1] == E.STATE_DIM
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_obs_layout_proprio_has_no_object_state():
    env = ArmEnv("lift", obs_mode="proprio", seed=1)
    s = env.reset()["state"]
    assert s.shape == (E.PROPRIO_DIM,)
    cube = env.cube_pos()
    # no 3 consecutive entries equal the cube position
    for i in range(E.PROPRIO_DIM - 2):
        assert not np.allclose(s[i:i + 3], cube, atol=1e-4)


def test_seeded_determinism():
    a = ArmEnv("push", seed=3)
    b = ArmEnv("push", seed=3)
    oa, ob = a.reset(), b.reset()
    assert np.array_equal(oa["state"], ob["state"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        act = rng.uniform(-1, 1, 6)
        oa, ra, da, ia = a.step(act)
        ob, rb, db, ib = b.step(act)
        assert np.array_equal(oa["state"], ob["state"]) and ra == rb and da == db


def test_reset_seed_replaces_rng():
    env = ArmEnv("reach", seed=0)
    o1 = env.reset(seed=42)["state"]
    env.step(np.zeros(6))
    o2 = env.reset(seed=42)["state"]
    assert np.array_equal(o1, o2)


def test_cube_spawns_in_region_and_others_park():
    env = ArmEnv("lift:green", seed=5)
    for _ in range(10):
        env.reset()
        c = env.cube_pos("green")
        assert scene.SPAWN_X[0] <= c[0] <= scene.SPAWN_X[1]
        assert scene.SPAWN_Y[0] <= c[1] <= scene.SPAWN_Y[1]
        assert abs(c[2] - scene.CUBE_HALF) < 1e-3
        for other in ("red", "blue"):
            assert np.hypot(*env.cube_pos(other)[:2]) > 1.0


def test_distractors_on_table_and_separated():
    env = ArmEnv("lift:green:distractors", seed=6)
    for _ in range(10):
        env.reset()
        ps = [env.cube_pos(c)[:2] for c in scene.CUBE_COLORS]
        for p in ps:
            assert scene.SPAWN_X[0] <= p[0] <= scene.SPAWN_X[1]
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.linalg.norm(ps[i] - ps[j]) >= E.DISTRACTOR_MIN_SEP - 1e-9


def test_push_target_is_on_table_at_the_right_distance():
    env = ArmEnv("push", seed=7)
    for _ in range(10):
        env.reset()
        d = np.linalg.norm(env.target_pos()[:2] - env.cube_pos()[:2])
        assert E.PUSH_DIST[0] - 1e-6 <= d <= E.PUSH_DIST[1] + 1e-6
        assert env.target_pos()[2] == 0.0


def test_delta_action_moves_target_by_scale_and_clips():
    env = ArmEnv("reach", action_mode="delta", seed=0)
    env.reset()
    t0 = env.joint_target()
    env.step(np.ones(6))
    assert np.allclose(env.joint_target(), np.clip(t0 + E.DELTA_SCALE, *env.joint_limits))
    lo, hi = env.joint_limits
    for _ in range(200):
        env.step(np.full(6, 5.0))     # out of range, must clip to 1
    assert np.allclose(env.joint_target(), hi)


def test_absolute_action_is_radians_clipped():
    env = ArmEnv("reach", action_mode="absolute", seed=0)
    env.reset()
    lo, hi = env.joint_limits
    env.step(hi + 1.0)
    assert np.allclose(env.joint_target(), hi)
    env.step(lo - 1.0)
    assert np.allclose(env.joint_target(), lo)


def test_zero_delta_holds_still():
    env = ArmEnv("reach", seed=0)
    env.reset()
    for _ in range(20):
        env.step(np.zeros(6))
    q0 = env.arm_qpos()
    for _ in range(20):
        env.step(np.zeros(6))
    assert np.allclose(env.arm_qpos(), q0, atol=2e-3)


def test_latency_delays_the_command():
    lag = ArmEnv("reach", seed=0, physics={"latency": 2})
    now = ArmEnv("reach", seed=0)
    lag.reset(seed=1)
    now.reset(seed=1)
    a = np.ones(6) * 0.5
    lag.step(a)
    now.step(a)
    # with 2 steps of latency, the physics has not yet seen the command
    assert np.allclose(lag.data.ctrl, lag._home + 0 * a, atol=E.INIT_NOISE + 1e-9)
    assert not np.allclose(lag.data.ctrl, now.data.ctrl)
    lag.step(a)
    lag.step(a)
    assert np.allclose(lag.data.ctrl, now.data.ctrl)


def test_truncation_and_termination_flags():
    env = ArmEnv("push", seed=0, episode_steps=5)
    env.reset()
    for t in range(5):
        _, _, done, info = env.step(np.zeros(6))
        assert info["t"] == t + 1
    assert done and info["truncated"] and not info["terminated"]


def test_reach_success_terminates():
    """Teleport the arm so the end effector is at the hover point."""
    from so_arm100_sim.ik import IK, top_down_rotation
    env = ArmEnv("reach", action_mode="absolute", seed=0)
    env.reset()
    ik = IK(env.model)
    q5, pe, _, _ = ik.solve(env.arm_qpos()[:5], env.target_pos(), top_down_rotation(0.0))
    assert pe < 2e-3
    q = np.concatenate([q5, [scene.JAW_OPEN]])
    env.data.qpos[env._qpos_i] = q
    env.data.qvel[env._qvel_i] = 0.0
    _, r, done, info = env.step(q)
    assert info["success"] and done and info["terminated"] and not info["truncated"]
    assert info["terms"]["success"] == E.REWARD_WEIGHTS["success"]


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        ArmEnv("reach").step(np.zeros(6))


def test_reward_terms_sum_to_reward():
    env = ArmEnv("pick_place", seed=0)
    env.reset()
    _, r, _, info = env.step(np.full(6, 0.3))
    assert abs(sum(info["terms"].values()) - r) < 1e-9


def test_obs_noise_only_when_configured():
    clean = ArmEnv("lift", seed=0)
    noisy = ArmEnv("lift", seed=0, physics={"obs_noise_q": 0.01, "obs_noise_pos": 0.01})
    c, n = clean.reset(seed=2)["state"], noisy.reset(seed=2)["state"]
    assert not np.allclose(c[E.STATE_LAYOUT["qpos"]], n[E.STATE_LAYOUT["qpos"]])
    assert not np.allclose(c[E.STATE_LAYOUT["cube_pos"]], n[E.STATE_LAYOUT["cube_pos"]])
    assert np.allclose(c[E.STATE_LAYOUT["ee_pos"]], n[E.STATE_LAYOUT["ee_pos"]])  # kinematics, not sensed
