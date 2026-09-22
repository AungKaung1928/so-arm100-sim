"""N environments in N forked processes.

MuJoCo's step for this model is single threaded and small, so the only
parallelism available is process level. OMP_NUM_THREADS is pinned to 1
before numpy or mujoco load, so the workers do not each spawn a thread pool.

Deliberately not gymnasium's vector env: the API surface is five methods and
the consumers are in this project family. Observations are dicts; the vector
env stacks them per key. `terminal_obs` is carried across on autoreset so a
value function can bootstrap on the state the episode actually ended in.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp      # noqa: E402
import traceback                 # noqa: E402

import numpy as np               # noqa: E402

from . import env as _env        # noqa: E402

MAX_WORKERS = 8   # the thread budget the throughput numbers were measured under


def _worker(conn, seed, kwargs):
    try:
        e = _env.ArmEnv(seed=seed, **kwargs)
        conn.send(("ok", e.reset()))
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
        conn.close()
        return
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "step":
                obs, rew, done, info = e.step(payload)
                info.pop("physics", None)
                if done:
                    info["terminal_obs"] = obs
                    obs = e.reset()
                conn.send(("ok", (obs, rew, done, info)))
            elif cmd == "reset":
                conn.send(("ok", e.reset(seed=payload)))
            elif cmd == "close":
                break
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


def _stack(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


class VecEnv:
    """Env i is seeded `seed + i`, so N workers reproduce N sequential envs."""

    def __init__(self, n=4, seed=0, **env_kwargs):
        self.closed = False
        self._conns, self._procs = [], []
        if n > MAX_WORKERS:
            raise ValueError(f"{n} workers requested; MAX_WORKERS is {MAX_WORKERS}")
        ctx = mp.get_context("fork")
        self.n = n
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True, args=(child, seed + i, env_kwargs))
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)
        self._last_obs = _stack([self._recv(i) for i in range(n)])

    def _recv(self, i):
        try:
            tag, payload = self._conns[i].recv()
        except EOFError:
            raise RuntimeError(f"worker {i} exited without replying; see stderr above") from None
        if tag == "error":
            raise RuntimeError(f"worker {i} raised:\n{payload}")
        return payload

    def reset(self, seed=None):
        if seed is None:
            return {k: v.copy() for k, v in self._last_obs.items()}
        for i, c in enumerate(self._conns):
            c.send(("reset", seed + i))
        self._last_obs = _stack([self._recv(i) for i in range(self.n)])
        return {k: v.copy() for k, v in self._last_obs.items()}

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.n, _env.ACT_DIM)
        for c, a in zip(self._conns, actions):
            c.send(("step", a))
        out = [self._recv(i) for i in range(self.n)]
        obs = _stack([o[0] for o in out])
        rew = np.array([o[1] for o in out], dtype=np.float64)
        done = np.array([o[2] for o in out], dtype=bool)
        self._last_obs = obs
        return {k: v.copy() for k, v in obs.items()}, rew, done, [o[3] for o in out]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for c in self._conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for c in self._conns:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
