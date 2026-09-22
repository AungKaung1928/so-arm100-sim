"""Print the measured facts `scene.py` quotes about the vendored arm.

Pinch point at home, pad gap versus jaw angle, table-level reach. Run it
after any upstream model update and compare with the module docstring.

    python tools/probe_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mujoco                    # noqa: E402
import numpy as np               # noqa: E402

from so_arm100_sim import scene  # noqa: E402


def main():
    m = scene.build_model(1)
    d = mujoco.MjData(m)
    qi = scene.arm_qpos_index(m)
    a = scene.name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_3")
    b = scene.name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_3")
    sid = scene.ee_site_id(m)

    mujoco.mj_resetDataKeyframe(m, d, scene.name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    print("home: ee_site", d.site_xpos[sid].round(4), " pad-3 midpoint",
          (0.5 * (d.geom_xpos[a] + d.geom_xpos[b])).round(4))
    print("site axes (columns = local x,y,z in world):\n", d.site_xmat[sid].reshape(3, 3).round(3))

    print("\njaw angle -> pad gap")
    for jaw in (-0.174, 0.0, 0.5, 1.0, 1.75):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[qi[scene.JAW]] = jaw
        mujoco.mj_forward(m, d)
        print(f"  {jaw:+.3f} rad  {np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) * 1e3:5.1f} mm")

    rng = np.random.default_rng(0)
    lo, hi = scene.joint_limits(m)
    pts = []
    for _ in range(20000):
        q = rng.uniform(lo, hi)
        q[scene.JAW] = 0.5
        d.qpos[qi] = q
        mujoco.mj_forward(m, d)
        pts.append(d.site_xpos[sid].copy())
    pts = np.array(pts)
    near = pts[np.abs(pts[:, 2]) < 0.03]
    r = np.hypot(near[:, 0], near[:, 1])
    print(f"\ntable-level reach (|z| < 3 cm, {len(near)} of 20000 samples):")
    print("  radius 5/50/95 pct:", np.percentile(r, [5, 50, 95]).round(3))
    print("  y range:", near[:, 1].min().round(3), near[:, 1].max().round(3),
          " (negative y is in front of the robot)")
    print("  spawn box:", scene.SPAWN_X, scene.SPAWN_Y)


if __name__ == "__main__":
    main()
