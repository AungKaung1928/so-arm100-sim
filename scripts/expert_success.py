"""Measure the scripted expert on every task under nominal and held-out physics.

This is the bench's own headline table: the ceiling a learned policy is
compared against, and the expert's own transfer gap. Single core. Writes
runs/expert_success<tag>.json and prints the markdown table the README uses.

    python scripts/expert_success.py --episodes 100 --seeds 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from so_arm100_sim.dr import EVAL_PHYSICS                    # noqa: E402
from so_arm100_sim.env import TASKS                          # noqa: E402
from so_arm100_sim.evaluate import evaluate, save_json, provenance   # noqa: E402
from so_arm100_sim.expert import ScriptedExpert              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--physics", nargs="*", default=list(EVAL_PHYSICS))
    ap.add_argument("--color", default="red")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    out = {"episodes": a.episodes, "seeds": a.seeds, "cells": {}, **provenance()}
    print(f"expert success, {a.episodes} episodes x {a.seeds} seeds per cell\n")
    print("| task | " + " | ".join(a.physics) + " |")
    print("|---|" + "---|" * len(a.physics))
    for task in a.tasks:
        row = []
        for phys in a.physics:
            r = evaluate(ScriptedExpert, f"{task}:{a.color}", n_episodes=a.episodes,
                         seeds=tuple(range(a.seeds)), physics=phys,
                         env_kwargs={"action_mode": "absolute"})
            out["cells"][f"{task}/{phys}"] = r
            row.append(f"{r['success_rate']:.2f} +- {r['success_std']:.2f}")
        print(f"| {task} | " + " | ".join(row) + " |", flush=True)
    path = save_json(f"runs/expert_success{('_' + a.tag) if a.tag else ''}.json", out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
