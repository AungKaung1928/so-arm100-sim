"""Export one real episode and its sim replays (command delay 0..5 frames) as JSON for a browser plot.

Uses the offsets and fitted dynamics of runs/real2sim_fit.json. Angles in degrees, sim mapped back to
the real units (deg = (q - offset) / sign), so real and sim share one axis.

    python scripts/export_replay_json.py PATH/file-000.parquet --episode 40 --out replay_ep40.json
"""
import argparse
import json

import numpy as np

from so_arm100_sim import real_replay as R
from so_arm100_sim.scene import ARM_JOINTS

ap = argparse.ArgumentParser()
ap.add_argument("parquet")
ap.add_argument("--episode", type=int, default=40)
ap.add_argument("--fit", default="runs/real2sim_fit.json")
ap.add_argument("--max-delay", type=int, default=5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

fit = json.load(open(a.fit))
action, state = R.load_episodes(a.parquet)[a.episode]
sign, off = np.ones(5), np.deg2rad(fit["offset_deg"])
model = R.build(R.Params(**fit["fitted"]))
q_cmd, q_real = R.to_sim(action, sign, off), R.to_sim(state, sign, off)
sim = {}
for d in range(a.max_delay + 1):
    q = R.replay(model, q_cmd, q_real[0], fps=30, delay=d)
    deg = np.rad2deg((q[:, :5] - off) / sign)
    sim[str(d)] = {"deg": deg.round(3).tolist(), "err_deg": R.tracking_error(q, q_real).round(4).tolist()}
out = {"episode": a.episode, "fps": 30, "joints": list(ARM_JOINTS), "fitted": fit["fitted"],
       "command_deg": action[:, :5].round(3).tolist(), "real_deg": state[:, :5].round(3).tolist(), "sim_by_delay": sim}
json.dump(out, open(a.out, "w"), separators=(",", ":"))
print(a.out, len(action), "frames;", {d: round(float(np.mean(v["err_deg"])), 3) for d, v in sim.items()})
