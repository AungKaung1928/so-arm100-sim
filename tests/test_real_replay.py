"""real_replay: a trajectory produced by the sim itself replays with zero error; other params do not."""
import numpy as np

from so_arm100_sim import real_replay as R

SIGN = np.ones(5)
OFFSET = np.array([0.0, -1.57, 1.57, 0.0, 0.0])   # inside every joint range


def _synthetic(p, T=60, fps=30):
    t = np.arange(T) / fps
    u = np.zeros((T, 6))
    u[:, :5] = 20.0 * np.sin(2 * np.pi * 0.5 * t)[:, None]   # degrees
    u[:, 5] = 30.0
    model = R.build(p)
    q_cmd = R.to_sim(u, SIGN, OFFSET)
    q_sim = R.replay(model, q_cmd, q_cmd[0], fps=fps, delay=p.delay)
    state = np.empty_like(u)
    state[:, :5] = np.rad2deg(q_sim[:, :5] - OFFSET)
    state[:, 5] = (q_sim[:, 5] - R.JAW_RANGE[0]) / (R.JAW_RANGE[1] - R.JAW_RANGE[0]) * 100
    return u, state


def test_to_sim_units():
    u = np.array([[90.0, 0, 0, 0, -45.0, 100.0]])
    q = R.to_sim(u, [-1, 1, 1, 1, 1], OFFSET)
    assert np.allclose(q[0, :5], [-np.pi / 2, -1.57, 1.57, 0.0, -np.pi / 4])
    assert np.isclose(q[0, 5], R.JAW_RANGE[1])


def test_build_sets_params():
    m = R.build(R.Params(kp=80.0, armature=0.05, frictionloss=0.2, damping=0.3))
    assert np.allclose(m.actuator_gainprm[:5, 0], 80.0) and np.allclose(m.actuator_biasprm[:5, 1], -80.0)
    assert (m.actuator_biasprm[:5, 2] < 0).all()      # dampratio resolved to a kv at compile
    assert np.allclose(m.dof_armature[:5], 0.05) and np.allclose(m.dof_frictionloss[:5], 0.2)
    assert np.allclose(m.dof_damping[:5], 0.3)


def test_self_replay_zero_error_and_param_sensitivity():
    p = R.Params(kp=50.0, delay=2)
    u, s = _synthetic(p)
    assert R.episode_error(p, u, s, SIGN, OFFSET).max() < 1e-6
    assert R.episode_error(R.Params(kp=20.0, delay=2), u, s, SIGN, OFFSET).max() > 0.1
    assert R.episode_error(R.Params(kp=50.0, delay=0), u, s, SIGN, OFFSET).max() > 0.1
