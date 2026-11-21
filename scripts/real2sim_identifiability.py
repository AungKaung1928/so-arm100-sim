"""Which of the fitted servo parameters does the real SO-100 data actually pin down?

Starts from a fit json (scripts/fit_real2sim.py) and answers three questions on its train episodes:

1. Convergence: restart Nelder-Mead from the fit until it reports success. How far do the parameters and the
   replay error move?
2. Profile per parameter: hold one parameter at m x its optimum (m on a log grid), refit the others, and compare
   the per-episode replay error with the optimum, paired by episode. A value is "distinguishable" when the mean
   increase is above the one-sided 95 % t bound of its own per-episode standard error. The reported interval is
   the contiguous run of grid values around m = 1 that are not distinguishable; "open" means it reaches the end
   of the grid. The inner refits are capped (--profile-maxfev), so a profile error can only be too high: a
   parameter reported as not identified stays so with a better inner fit. The slice (others fixed) is saved too.
3. Offsets: refit the three gravity-joint zero offsets at the converged dynamics (stage 1 fitted them at the
   menagerie defaults), then slice the error along each one across its allowed range.

--slice-only skips 1-3 and only slices the dynamics around the json's values: the cheap cross-check on a second
dataset (does its error also bottom out at m = 1?).

One process, one core. About 55 min at the defaults, 1 min with --slice-only.

    python scripts/real2sim_identifiability.py PICKPLACE.parquet --out runs/real2sim_identifiability.json
    python scripts/real2sim_identifiability.py STACKING.parquet --fit runs/real2sim_transfer.json --slice-only \
        --out runs/real2sim_identifiability_stacking.json
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

from so_arm100_sim import real_replay as R

LO = np.array([-1.92, -3.32, -0.174, -1.66, -2.79])    # so_arm100.xml joint ranges (rad), as in fit_real2sim.py
HI = np.array([1.92, 0.174, 3.14, 1.66, 2.79])
GRAVITY_JOINTS = [1, 2, 3]
JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
NAMES = ["kp", "dampratio", "armature", "frictionloss", "damping"]
GRID = [0.125, 0.25, 0.5, 0.71, 0.84, 1.0, 1.19, 1.41, 2.0, 4.0, 8.0]


def paired(c, c_opt):
    """Mean per-episode increase over the optimum, its standard error, and whether it clears the t bound."""
    d = c - c_opt
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(se), bool(d.mean() > student_t.ppf(0.95, len(d) - 1) * se)


def interval(points):
    """Contiguous run of not-distinguishable multipliers around 1; None on a side = open at the grid end."""
    ms = [p["m"] for p in points]
    i = j = ms.index(1.0)
    while i > 0 and not points[i - 1]["distinguishable"]:
        i -= 1
    while j < len(ms) - 1 and not points[j + 1]["distinguishable"]:
        j += 1
    return [None if i == 0 else ms[i], None if j == len(ms) - 1 else ms[j]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--fit", default="runs/real2sim_fit.json")
    ap.add_argument("--slice-only", action="store_true")
    ap.add_argument("--restart-maxfev", type=int, default=600)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--profile-maxfev", type=int, default=100)
    ap.add_argument("--out", default="runs/real2sim_identifiability.json")
    a = ap.parse_args()

    with open(a.fit) as fh:
        fit = json.load(fh)
    eps = R.load_episodes(a.parquet)
    tr, te = fit["train_episodes"], fit["heldout_episodes"]
    sign = np.ones(5)
    off_fit = np.deg2rad(fit["offset_deg"])
    x_fit = np.log([fit["fitted"][k] for k in NAMES])
    t0 = time.time()

    def costs(x, off, ks):
        p = R.Params(**dict(zip(NAMES, np.exp(x))))
        return np.array([R.episode_error(p, *eps[k], sign, off).mean() for k in ks])

    def nm(f, x0, step, maxfev):
        simplex = np.vstack([x0, x0 + step * np.eye(len(x0))])
        return minimize(f, x0, method="Nelder-Mead",
                        options=dict(xatol=1e-2, fatol=1e-4, maxfev=maxfev, initial_simplex=simplex))

    res = {"data": a.parquet.split("/")[-2:], "fit": a.fit, "train_episodes": tr, "grid": GRID}

    # 1. convergence
    x_opt, off = x_fit.copy(), off_fit
    if not a.slice_only:
        runs = []
        for _ in range(a.restarts):
            r = nm(lambda x: costs(x, off, tr).mean(), x_opt, np.log(1.2), a.restart_maxfev)
            x_opt = r.x
            runs.append({"nfev": int(r.nfev), "converged": bool(r.success), "train_err_deg": round(float(r.fun), 4)})
            print("restart", runs[-1], flush=True)
            if r.success:
                break
        res["convergence"] = {
            "restarts": runs,
            "train_err_deg_fit": round(float(costs(x_fit, off, tr).mean()), 4),
            "heldout_err_deg_fit": round(float(costs(x_fit, off, te).mean()), 4),
            "heldout_err_deg_converged": round(float(costs(x_opt, off, te).mean()), 4),
            "fit": dict(zip(NAMES, np.exp(x_fit).round(4).tolist())),
            "converged": dict(zip(NAMES, np.exp(x_opt).round(4).tolist())),
        }

    # 2. slice and profile per parameter
    c_opt = costs(x_opt, off, tr)
    res["train_err_deg_at_optimum"] = round(float(c_opt.mean()), 4)
    res["params"] = {}
    for i, name in enumerate(NAMES):
        free = [j for j in range(len(NAMES)) if j != i]
        pts = {"slice": [], "profile": []}
        for m in GRID:
            x = x_opt.copy()
            x[i] += np.log(m)
            c = costs(x, off, tr)
            inc, se, dist = paired(c, c_opt)
            pts["slice"].append({"m": m, "train_err_deg": round(float(c.mean()), 4), "increase_deg": round(inc, 4),
                                 "se_deg": round(se, 4), "distinguishable": dist})
            if a.slice_only:
                continue
            if m != 1.0:
                def f(y, x=x, free=free):
                    z = x.copy()
                    z[free] = y
                    return costs(z, off, tr).mean()
                x[free] = nm(f, x[free], np.log(1.3), a.profile_maxfev).x
                c = costs(x, off, tr)
                inc, se, dist = paired(c, c_opt)
            pts["profile"].append({"m": m, "train_err_deg": round(float(c.mean()), 4), "increase_deg": round(inc, 4),
                                   "se_deg": round(se, 4), "distinguishable": dist,
                                   "heldout_err_deg": round(float(costs(x, off, te).mean()), 4),
                                   "refit": dict(zip(NAMES, np.exp(x).round(4).tolist()))})
        out = {"optimum": round(float(np.exp(x_opt[i])), 4), "slice_interval_x": interval(pts["slice"]),
               "slice": pts["slice"]}
        if not a.slice_only:
            out["profile_interval_x"] = interval(pts["profile"])
            out["profile"] = pts["profile"]
        res["params"][name] = out
        print(name, {k: v for k, v in out.items() if k.endswith("_x")}, flush=True)

    # 3. offsets at the converged dynamics
    if not a.slice_only:
        S = np.concatenate([s for _, s in eps.values()])
        olo, ohi = LO - np.deg2rad(S.min(0)[:5]), HI - np.deg2rad(S.max(0)[:5])

        def f_off(y):
            o = off_fit.copy()
            o[GRAVITY_JOINTS] = y
            if (o < olo).any() or (o > ohi).any():
                return 1e3
            return costs(x_opt, o, tr).mean()

        r = nm(f_off, off_fit[GRAVITY_JOINTS], -np.deg2rad(2.0) * np.sign(off_fit[GRAVITY_JOINTS] - (olo + ohi)[GRAVITY_JOINTS] / 2), 200)
        off_new = off_fit.copy()
        off_new[GRAVITY_JOINTS] = r.x
        res["offsets"] = {
            "allowed_deg": {JOINTS[j]: [round(float(np.rad2deg(olo[j])), 2), round(float(np.rad2deg(ohi[j])), 2)]
                            for j in GRAVITY_JOINTS},
            "fit_deg": {JOINTS[j]: round(float(np.rad2deg(off_fit[j])), 2) for j in GRAVITY_JOINTS},
            "refit_deg": {JOINTS[j]: round(float(np.rad2deg(off_new[j])), 2) for j in GRAVITY_JOINTS},
            "refit_at_bound": {JOINTS[j]: bool(min(abs(off_new[j] - olo[j]), abs(off_new[j] - ohi[j])) < 1e-2)
                               for j in GRAVITY_JOINTS},
            "refit_nfev": int(r.nfev), "refit_converged": bool(r.success),
            "train_err_deg_refit": round(float(r.fun), 4),
            "heldout_err_deg_refit": round(float(costs(x_opt, off_new, te).mean()), 4),
            "slice": {},
        }
        for j in GRAVITY_JOINTS:
            rows = []
            for o_j in np.linspace(olo[j], ohi[j], 9):
                o = off_fit.copy()
                o[j] = o_j
                c = costs(x_opt, o, tr)
                rows.append({"offset_deg": round(float(np.rad2deg(o_j)), 2), "train_err_deg": round(float(c.mean()), 4)})
            res["offsets"]["slice"][JOINTS[j]] = rows
        print("offsets", res["offsets"]["refit_deg"], res["offsets"]["refit_at_bound"], flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", a.out, res["seconds"], "s")


if __name__ == "__main__":
    main()
