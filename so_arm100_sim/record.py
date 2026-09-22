"""Record scripted-expert demonstrations as a LeRobotDataset v3.

Only successful episodes are kept (the expert's failures are not
demonstrations). Each frame stores the proprio state, one image per
requested camera, the expert's absolute joint targets as the action, and a
language instruction sampled from the training templates for that
(task, colour). The held-out templates and combinations in
`instructions.py` are never written here.

    python -m so_arm100_sim.record --root data/lift_red --repo-id local/so_arm100_lift_red \
        --tasks lift:red --episodes 50 --cameras front

Requires the `lerobot` extra (Python >= 3.12) and a GL context. Lives in the
package (not only under scripts/) so a consumer that installed this from git
can call `python -m so_arm100_sim.record` or import `record_episode`.
"""
import argparse
import json
import os
import time

import numpy as np

from . import instructions as I
from .datasets import LeRobotRecorder
from .dr import DRConfig
from .env import ArmEnv, TaskSpec, CONTROL_HZ
from .expert import ScriptedExpert


def record_episode(env, expert, cameras, instruction, seed):
    obs = env.reset(seed=seed)
    expert.reset()
    frames = []
    while True:
        a = expert.act()
        frames.append({"state": obs["state"], "action": a, "task": instruction,
                       "images": {c: (obs["image"] if c == env.image_cfg["camera"]
                                      else env.render(c, env.image_cfg["height"],
                                                      env.image_cfg["width"]))
                                  for c in cameras}})
        obs, _, done, info = env.step(a)
        if done:
            return frames, bool(info["success_ever"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--tasks", nargs="+", default=["lift:red"],
                    help="TaskSpec strings, e.g. lift:red push:green:distractors")
    ap.add_argument("--episodes", type=int, default=50, help="successful episodes per task")
    ap.add_argument("--cameras", nargs="+", default=["front"])
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dr", action="store_true", help="randomise physics per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=3, help="x episodes, before giving up")
    a = ap.parse_args(argv)

    rec = LeRobotRecorder(a.root, a.repo_id, fps=CONTROL_HZ, cameras=a.cameras,
                          height=a.height, width=a.width)
    rng = np.random.default_rng(a.seed)
    log = {"tasks": {}, "cameras": a.cameras, "size": [a.height, a.width], "dr": a.dr}
    t0 = time.perf_counter()
    for spec in a.tasks:
        task = TaskSpec.parse(spec)
        env = ArmEnv(task, obs_mode="proprio", action_mode="absolute", seed=a.seed,
                     dr=DRConfig() if a.dr else None,
                     image={"camera": a.cameras[0], "height": a.height, "width": a.width})
        ex = ScriptedExpert(env)
        kept, attempts, frames_total = 0, 0, 0
        while kept < a.episodes and attempts < a.max_attempts * a.episodes:
            instr = I.sample_instruction(task.kind, task.color, rng, "train")
            frames, ok = record_episode(env, ex, a.cameras, instr, seed=a.seed * 100_000 + attempts)
            attempts += 1
            if ok:
                rec.add_episode(frames)
                kept += 1
                frames_total += len(frames)
                if kept % 10 == 0:
                    print(f"  {spec}: {kept}/{a.episodes} kept, {attempts} attempts, "
                          f"{frames_total} frames, {time.perf_counter() - t0:.0f} s", flush=True)
        env.close()
        log["tasks"][spec] = {"kept": kept, "attempts": attempts, "frames": frames_total,
                              "expert_success_during_recording": kept / max(attempts, 1)}
        print(f"{spec}: kept {kept} of {attempts} attempts ({kept / max(attempts, 1):.2f})")
    rec.finalize()
    log["episodes"] = rec.n_episodes
    log["frames"] = rec.n_frames
    log["wall_s"] = time.perf_counter() - t0
    os.makedirs(a.root, exist_ok=True)
    with open(os.path.join(a.root, "recording_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"dataset: {rec.n_episodes} episodes, {rec.n_frames} frames -> {a.root}")


if __name__ == "__main__":
    main()
