"""Refit kp, dampratio, armature, frictionloss on single episodes (offsets and damping fixed from
runs/real2sim_fit.json) and report the per-episode spread, the input for DR ranges.

    python scripts/real2sim_spread.py PATH/file-000.parquet --out runs/real2sim_spread.json
    # second dataset: its own offsets (fit_real2sim.py --dynamics-from), pick-place dynamics as the start
    python scripts/real2sim_spread.py STACKING.parquet --fit runs/real2sim_transfer.json --out runs/real2sim_spread_stacking.json
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize

from so_arm100_sim import real_replay as R

N = ["kp", "dampratio", "armature", "frictionloss"]

ap = argparse.ArgumentParser()
ap.add_argument("parquet")
ap.add_argument("--fit", default="runs/real2sim_fit.json")
ap.add_argument("--episodes", default="10:30")
ap.add_argument("--maxfev", type=int, default=120)
ap.add_argument("--out", default="runs/real2sim_spread.json")
a = ap.parse_args()

eps = R.load_episodes(a.parquet)
lo, hi = map(int, a.episodes.split(":"))
ks = list(eps)[lo:hi]
fit = json.load(open(a.fit))
off, sign = np.deg2rad(fit["offset_deg"]), np.ones(5)
x0 = np.log([fit["fitted"][k] for k in N])
simplex = np.vstack([x0, x0 + np.log(1.5) * np.eye(len(x0))])   # see fit_real2sim.py


def f(x, k):
    if (np.abs(x - x0) > np.log(10)).any():
        return 1e3
    p = R.Params(**dict(zip(N, np.exp(x))), damping=fit["fitted"]["damping"])
    return R.episode_error(p, *eps[k], sign, off).mean()


per = np.array([np.exp(minimize(f, x0, args=(k,), method="Nelder-Mead",
                                options=dict(xatol=1e-2, fatol=1e-4, maxfev=a.maxfev,
                                             initial_simplex=simplex)).x) for k in ks])
res = {"data": a.parquet.split("/")[-1], "fit": a.fit, "maxfev": a.maxfev,   # inputs, so the file says how to rerun it
       "episodes": ks, "params": N, "per_episode": per.round(4).tolist(),
       "median": dict(zip(N, np.median(per, 0).round(4).tolist())),
       "p10": dict(zip(N, np.percentile(per, 10, 0).round(4).tolist())),
       "p90": dict(zip(N, np.percentile(per, 90, 0).round(4).tolist()))}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ["median", "p10", "p90"]}))
