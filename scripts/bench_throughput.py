"""Environment throughput at 1, 2, 4 and 8 processes, with a drift guard.

One env step is one 20 Hz control decision = 25 physics steps + one
`mj_forward` + the observation and reward. That is the unit RL budgets are
quoted in, so it is the unit measured here.

The guard: a single-process reference is measured before and after every
row. If the reference moves more than `--drift` (default 10%), or trends
monotonically across the sweep, the JSON is written with `stable: false`
and the table is not to be quoted. A laptop that is throttling, charging
from a weak supply, or running something else looks exactly like a scaling
result otherwise. Rows are short bursts; a sustained run (`--sustained N`)
shows where the burst rate settles.

    nice -n 10 python scripts/bench_throughput.py --seconds 15
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                       # noqa: E402

from so_arm100_sim.evaluate import provenance, save_json   # noqa: E402
from so_arm100_sim.vec_env import VecEnv                  # noqa: E402


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def measure(n, seconds, task, seed=0):
    with VecEnv(n=n, seed=seed, task=task) as vec:
        rng = np.random.default_rng(seed)
        vec.step(rng.uniform(-1, 1, (n, 6)))          # warm
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            vec.step(rng.uniform(-1, 1, (n, 6)))
            steps += n
        return steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lift")
    ap.add_argument("--procs", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--ref-seconds", type=float, default=6.0)
    ap.add_argument("--drift", type=float, default=0.10)
    ap.add_argument("--sustained", type=int, default=0,
                    help="instead of the sweep: run N procs in back-to-back windows")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    warnings = []
    if load1() > 2.0:
        warnings.append(f"1-min load average {load1():.2f} at start; something else is running")
        print("WARNING", warnings[-1])

    if a.sustained:
        rates = []
        print(f"sustained: {a.sustained} procs, {a.windows} x {a.seconds:.0f} s, no pauses")
        for w in range(a.windows):
            r = measure(a.sustained, a.seconds, a.task)
            rates.append(r)
            print(f"  window {w + 1:2d}  {r:8,.0f} env-steps/s  ({100 * (r / rates[0] - 1):+.0f}% vs first)")
        tail = rates[len(rates) // 2:]
        out = {"mode": "sustained", "task": a.task, "procs": a.sustained, "seconds": a.seconds,
               "rates": rates, "plateau_mean": float(np.mean(tail)),
               "plateau_band": [float(min(tail)), float(max(tail))],
               "decay_vs_first": float(tail[-1] / rates[0] - 1), "warnings": warnings,
               **provenance()}
        path = save_json(f"runs/sustained{('_' + a.tag) if a.tag else ''}.json", out)
        print(f"plateau (second half) {out['plateau_mean']:,.0f}, band {out['plateau_band']}")
        print(f"wrote {path}")
        return

    refs = [measure(1, a.ref_seconds, a.task)]
    rows = []
    print(f"sweep on {a.task!r}: {a.seconds:.0f} s per row, {a.ref_seconds:.0f} s references\n")
    print(f"  reference  {refs[0]:8,.0f} env-steps/s (1 proc)")
    for n in a.procs:
        r = measure(n, a.seconds, a.task)
        refs.append(measure(1, a.ref_seconds, a.task))
        rows.append({"procs": n, "env_steps_per_s": r, "per_proc": r / n,
                     "speedup": r / refs[0], "efficiency": r / refs[0] / n,
                     "ref_after": refs[-1], "ref_drift": refs[-1] / refs[0] - 1})
        print(f"  {n} proc     {r:8,.0f} env-steps/s  {100 * rows[-1]['efficiency']:3.0f}% "
              f"efficient   ref after {refs[-1]:,.0f} ({100 * rows[-1]['ref_drift']:+.1f}%)")
    drifts = np.array(refs) / refs[0] - 1
    worst = float(np.abs(drifts).max())
    diffs = np.diff(refs)
    trend = bool(len(diffs) >= 3 and (np.all(diffs > 0) or np.all(diffs < 0)))
    stable = worst <= a.drift and not trend and not warnings
    print(f"\n  worst reference drift {100 * worst:+.1f}%  monotonic trend: {trend}  "
          f"-> {'STABLE' if stable else 'NOT STABLE, do not quote these rows'}")
    out = {"mode": "sweep", "task": a.task, "seconds": a.seconds, "ref_seconds": a.ref_seconds,
           "references": refs, "rows": rows, "worst_ref_drift": worst, "trend": trend,
           "stable": stable, "warnings": warnings, "cores": os.cpu_count(), **provenance()}
    path = save_json(f"runs/throughput{('_' + a.tag) if a.tag else ''}.json", out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
