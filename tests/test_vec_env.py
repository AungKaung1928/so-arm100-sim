"""N workers reproduce N sequential envs, bit for bit."""
import numpy as np

from so_arm100_sim.env import ArmEnv
from so_arm100_sim.vec_env import VecEnv


def test_vec_env_matches_sequential():
    n, seed = 3, 40
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (12, n, 6))
    seq = [ArmEnv("push", seed=seed + i) for i in range(n)]
    seq_obs = [e.reset() for e in seq]
    with VecEnv(n=n, seed=seed, task="push") as vec:
        obs = vec.reset()
        assert obs["state"].shape == (n, 25)
        for i in range(n):
            assert np.array_equal(obs["state"][i], seq_obs[i]["state"])
        for t in range(12):
            vo, vr, vd, vi = vec.step(acts[t])
            for i, e in enumerate(seq):
                o, r, d, info = e.step(acts[t][i])
                assert np.array_equal(vo["state"][i], o["state"])
                assert vr[i] == r and vd[i] == d


def test_vec_env_autoreset_carries_terminal_obs():
    with VecEnv(n=2, seed=0, task="reach", episode_steps=3) as vec:
        for _ in range(2):
            _, _, done, infos = vec.step(np.zeros((2, 6)))
            assert not done.any()
        obs, _, done, infos = vec.step(np.zeros((2, 6)))
        assert done.all()
        for info in infos:
            assert "terminal_obs" in info and info["truncated"]
        # after autoreset the returned obs is a fresh episode (t back to 0 next step)
        _, _, _, infos = vec.step(np.zeros((2, 6)))
        assert all(i["t"] == 1 for i in infos)


def test_worker_error_is_named():
    import pytest
    with pytest.raises(RuntimeError, match="worker 0 raised"):
        VecEnv(n=1, seed=0, task="not_a_task")
