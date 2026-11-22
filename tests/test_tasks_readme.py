"""The README task table (success tolerances, episode lengths, physics step) is what env.py and scene.py hold."""
import os

from so_arm100_sim import env, scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = open(os.path.join(ROOT, "docs", "DETAILS.md")).read().splitlines()


def cm(x):
    return f"{100 * x:g} cm"


def test_task_table():
    s = env.EPISODE_STEPS
    rows = [
        f"| `reach` | end effector within {cm(env.REACH_TOL)} of a point {cm(env.REACH_HOVER)} above the cube | {s['reach']} |",
        f"| `push` | cube centre within {cm(env.PUSH_TOL)} (xy) of the marker | {s['push']} |",
        f"| `lift` | cube {cm(env.LIFT_HEIGHT)} above its resting height **and** both jaws touching it | {s['lift']} |",
        f"| `pick_place` | cube within {cm(env.PLACE_TOL)} of the marker, on the table, released | {s['pick_place']} |",
    ]
    # whole lines: the expert-success table repeats these cells before its own columns
    for row in rows:
        assert row in LINES, row
        assert any(l.startswith(row + " ") for l in LINES), "success table: " + row


def test_physics_step():
    assert f"{1000 * scene.PHYSICS_DT:g} ms physics step." in LINES
