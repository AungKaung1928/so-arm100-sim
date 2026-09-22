"""Render each camera once, and an expert lift episode as a filmstrip.

    python scripts/view_scene.py --out out
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np                                   # noqa: E402

from so_arm100_sim import scene                      # noqa: E402
from so_arm100_sim.env import ArmEnv                 # noqa: E402
from so_arm100_sim.expert import ScriptedExpert      # noqa: E402


def save(path, img):
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--task", default="pick_place:red:distractors")
    ap.add_argument("--size", type=int, nargs=2, default=(240, 320))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h, w = a.size

    env = ArmEnv(a.task, action_mode="absolute", seed=0)
    env.reset(seed=0)
    for cam in scene.CAMERAS:
        save(os.path.join(a.out, f"camera_{cam}.png"), env.render(cam, h, w))
        print(f"wrote {a.out}/camera_{cam}.png")

    ex = ScriptedExpert(env)
    env.reset(seed=1)
    ex.reset()
    frames = [env.render("front", h, w)]
    while True:
        _, _, done, info = env.step(ex.act())
        frames.append(env.render("front", h, w))
        if done:
            break
    # eight frames spread over the episode that actually happened, plus the last
    idx = np.unique(np.linspace(0, len(frames) - 1, 8).astype(int))
    strip = np.concatenate([frames[i] for i in idx], axis=1)
    save(os.path.join(a.out, "expert_filmstrip.png"), strip)
    print(f"wrote {a.out}/expert_filmstrip.png  ({len(idx)} of {len(frames)} frames, "
          f"success={info['success_ever']})")


if __name__ == "__main__":
    main()
