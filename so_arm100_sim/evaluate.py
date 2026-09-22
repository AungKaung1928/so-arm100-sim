"""The evaluation protocol. One function, used by every project in the family.

    result = evaluate(policy_factory, "lift:red", physics="heavy")

`policy_factory(env)` returns an object with `reset()` and `act(obs)`; it
gets the env so a scripted policy can read privileged state, and a learned
policy can ignore it. Success is `info["success_ever"]` at the end of the
episode: did the criterion in `env.py` hold at any step.

Default protocol: 100 episodes x 5 seeds = 500 episodes per (task, physics)
cell. The number reported is the mean over seeds of the per-seed success
rate, with the sample standard deviation over the 5 seeds -- that spread is
what makes two cells comparable. Episode j of seed s is reset with
`seed * 10_007 + j`, so any single episode can be replayed by hand.

Every result dict carries the package version, the vendored model commit and
the date, so a table in a README can be traced to the code that produced it.
"""
import datetime as _dt
import json
import os
import time

import numpy as np

from . import scene
from .env import ArmEnv, TaskSpec

DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def run_episode(env, policy, seed):
    obs = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total, steps, first_success = 0.0, 0, None
    while True:
        obs, r, done, info = env.step(policy.act(obs))
        total += r
        steps += 1
        if info["success"] and first_success is None:
            first_success = steps
        if done:
            return {"success": bool(info["success_ever"]), "steps": steps,
                    "return": total, "first_success": first_success,
                    "grasped": bool(info.get("grasped", False))}


def evaluate(policy_factory, task, n_episodes=100, seeds=DEFAULT_SEEDS, physics="nominal",
             dr=None, env_kwargs=None, progress=False):
    task = TaskSpec.parse(task)
    env_kwargs = dict(env_kwargs or {})
    t0 = time.perf_counter()
    per_seed, all_eps = [], []
    for s in seeds:
        env = ArmEnv(task, seed=int(s), physics=None if dr is not None else physics,
                     dr=dr, **env_kwargs)
        policy = policy_factory(env)
        eps = []
        for j in range(n_episodes):
            eps.append(run_episode(env, policy, int(s) * 10_007 + j))
            if progress and (j + 1) % 20 == 0:
                rate = np.mean([e["success"] for e in eps])
                print(f"  seed {s} ep {j + 1}/{n_episodes}  success {rate:.2f}", flush=True)
        env.close()
        per_seed.append(float(np.mean([e["success"] for e in eps])))
        all_eps.extend(eps)
    firsts = [e["first_success"] for e in all_eps if e["first_success"] is not None]
    return {
        "task": str(task), "physics": physics if dr is None else "dr",
        "dr": None if dr is None else dr.to_dict(),
        "n_episodes": int(n_episodes), "seeds": [int(s) for s in seeds],
        "success_rate": float(np.mean(per_seed)),
        "success_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed,
        "return_mean": float(np.mean([e["return"] for e in all_eps])),
        "steps_mean": float(np.mean([e["steps"] for e in all_eps])),
        "first_success_p50": float(np.percentile(firsts, 50)) if firsts else None,
        "first_success_p90": float(np.percentile(firsts, 90)) if firsts else None,
        "wall_s": time.perf_counter() - t0,
        **provenance(),
    }


def provenance():
    from . import __version__
    return {"so_arm100_sim": __version__, "model_commit": scene.VENDORED_COMMIT,
            "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def gap_table(results):
    """{name: result} -> rows of (name, success_rate, success_std, delta vs nominal)."""
    base = results["nominal"]["success_rate"] if "nominal" in results else None
    rows = []
    for name, r in results.items():
        delta = None if base is None else r["success_rate"] - base
        rows.append((name, r["success_rate"], r["success_std"], delta))
    return rows


def format_gap_table(results):
    lines = ["| physics | success | +- (5 seeds) | vs nominal |", "|---|---|---|---|"]
    for name, rate, std, delta in gap_table(results):
        d = "--" if delta is None else f"{delta:+.3f}"
        lines.append(f"| {name} | {rate:.3f} | {std:.3f} | {d} |")
    return "\n".join(lines)
