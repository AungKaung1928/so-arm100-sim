"""Where is a TOP-DOWN grasp pose IK-reachable? Prints a grid per height.

The arm can put its fingertip almost anywhere in a 0.4 m radius; it cannot
point straight down everywhere in that radius, because pointing down spends
the wrist pitch. This map is what set `scene.SPAWN_X` / `SPAWN_Y` and the lift
height: the spawn box must be inside the '##' region at the height the lift
task has to hold the cube (about z = 0.09 m).

    python tools/reach_map.py --z 0.03 0.09
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np                                     # noqa: E402

from so_arm100_sim import scene                        # noqa: E402
from so_arm100_sim.ik import IK, top_down_rotation     # noqa: E402


def reachable(ik, home, x, y, z, pos_tol=3e-3, rot_tol=0.1):
    for yaw in (0.0, 0.7, -0.7):
        for flip in (False, True):
            R = top_down_rotation(yaw)
            if flip:
                R = R.copy()
                R[:, 0] *= -1
                R[:, 2] *= -1
            _, pe, re_, _ = ik.solve(home, [x, y, z], R)
            if pe < pos_tol and re_ < rot_tol:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, nargs="+", default=[0.03, 0.06, 0.09, 0.11])
    ap.add_argument("--step", type=float, default=0.02)
    a = ap.parse_args()
    m = scene.build_model(1)
    ik = IK(m)
    home = scene.home_qpos(m)[:5]
    xs = np.arange(-0.16, 0.161, a.step)
    ys = np.arange(-0.34, -0.119, a.step)
    for z in a.z:
        print(f"\nz = {z:.2f} m   ## = top-down pose solvable to 3 mm / 0.1 rad")
        print("       " + " ".join(f"{x:+.2f}" for x in xs))
        for y in ys:
            cells = []
            for x in xs:
                inbox = scene.SPAWN_X[0] <= x <= scene.SPAWN_X[1] and scene.SPAWN_Y[0] <= y <= scene.SPAWN_Y[1]
                ok = reachable(ik, home, x, y, z)
                cells.append((" [#] " if inbox else " ##  ") if ok else (" [.] " if inbox else "  .  "))
            print(f"{y:+.2f} " + "".join(cells))
        print("       [ ] marks the spawn box")


if __name__ == "__main__":
    main()
