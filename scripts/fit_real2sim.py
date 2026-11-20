"""Fit sim arm parameters to real SO-100 data (two Nelder-Mead stages), report held-out error before/after.

Stage 1: zero offsets of the gravity joints (Pitch, Elbow, Wrist_Pitch), menagerie defaults, offsets kept so the
real range stays inside the sim joint range. Stage 2: kp, dampratio, armature, frictionloss, damping (log space,
within x20 of the start) with stage-1 offsets fixed. Train = first N episodes, held-out = last M episodes.

Data: meta + data parquet of lerobot/svla_so100_pickplace (data/chunk-000/file-000.parquet, 476 KB).
Single process; about 2.5 min on one core at the defaults.

    python scripts/fit_real2sim.py PATH/file-000.parquet --out runs/real2sim_fit.json
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

from so_arm100_sim import real_replay as R
from so_arm100_sim.real_replay import Params

LO = np.array([-1.92, -3.32, -0.174, -1.66, -2.79])    # so_arm100.xml joint ranges (rad), 5 arm joints
HI = np.array([1.92, 0.174, 3.14, 1.66, 2.79])
GRAVITY_JOINTS = [1, 2, 3]
NAMES = ["kp", "dampratio", "armature", "frictionloss", "damping"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--heldout", type=int, default=10)
    ap.add_argument("--maxfev", type=int, default=250)
    ap.add_argument("--out", default="runs/real2sim_fit.json")
    a = ap.parse_args()

    eps = R.load_episodes(a.parquet)
    keys = list(eps)
    tr, te = keys[:a.train], keys[-a.heldout:]
    assert not set(tr) & set(te), "train and held-out overlap"
    S = np.concatenate([s for _, s in eps.values()])
    rlo, rhi = np.deg2rad(S.min(0)[:5]), np.deg2rad(S.max(0)[:5])
    sign = np.ones(5)
    olo, ohi = LO - rlo, HI - rhi
    off0 = (LO + HI) / 2 - (rlo + rhi) / 2

    def err(p, off, ks):
        return np.mean([R.episode_error(p, *eps[k], sign, off) for k in ks], 0)

    t0 = time.time()
    default = Params()

    def f_off(x):
        off = off0.copy()
        off[GRAVITY_JOINTS] = x
        if (off < olo).any() or (off > ohi).any():
            return 1e3
        return err(default, off, tr).mean()

    r1 = minimize(f_off, off0[GRAVITY_JOINTS], method="Nelder-Mead",
                  options=dict(xatol=1e-3, fatol=1e-4, maxfev=a.maxfev))
    off = off0.copy()
    off[GRAVITY_JOINTS] = r1.x

    x0 = np.log([default.kp, default.dampratio, default.armature, default.frictionloss, 0.01])

    def to_params(x):
        return Params(**dict(zip(NAMES, np.exp(x))))

    def f_dyn(x):
        if (np.abs(x - x0) > np.log(20)).any():
            return 1e3
        return err(to_params(x), off, tr).mean()

    r2 = minimize(f_dyn, x0, method="Nelder-Mead", options=dict(xatol=1e-2, fatol=1e-4, maxfev=a.maxfev))
    fitted = to_params(r2.x)

    res = {
        "data": a.parquet.split("/")[-1], "train_episodes": tr, "heldout_episodes": te,
        "offset_deg": np.rad2deg(off).round(3).tolist(),
        "offset_at_bound": [bool(min(abs(off[j] - olo[j]), abs(off[j] - ohi[j])) < 1e-2) for j in range(5)],
        "fitted": {k: round(float(getattr(fitted, k)), 4) for k in NAMES},
        "stage_nfev": [int(r1.nfev), int(r2.nfev)], "stage_converged": [bool(r1.success), bool(r2.success)],
        "heldout_err_deg_midrange_offsets_defaults": err(default, off0, te).round(3).tolist(),
        "heldout_err_deg_fitted_offsets_defaults": err(default, off, te).round(3).tolist(),
        "heldout_err_deg_fitted_all": err(fitted, off, te).round(3).tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
