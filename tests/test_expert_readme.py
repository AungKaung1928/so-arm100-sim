"""README expert-success table must match runs/expert_success*.json, cell by cell."""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = (ROOT / "docs" / "DETAILS.md").read_text()
CELLS = {**json.loads((ROOT / "runs/expert_success.json").read_text())["cells"],
         **json.loads((ROOT / "runs/expert_success_realfit.json").read_text())["cells"]}
HEADER = "| task | nominal | heavy | slippery | weak | laggy | noisy | small | real_fit |"


def test_expert_table_matches_json():
    physics = [c.strip() for c in HEADER.strip("|").split("|")][1:]
    rows = README.split(HEADER)[1].splitlines()[2:]
    seen = 0
    for line in rows:
        if not line.startswith("|"):
            break
        task, *vals = [c.strip() for c in line.strip().strip("|").split("|")]
        for p, v in zip(physics, vals):
            r = CELLS[f"{task}/{p}"]
            assert v == f"{r['success_rate']:.2f} ± {r['success_std']:.2f}", (task, p, v)
        seen += 1
    assert seen == 4


def _pm(r):
    return f"{r['success_rate']:.2f} ± {r['success_std']:.2f}"


def test_summary_rows_match_json():
    held_out = ["heavy", "slippery", "weak", "laggy", "noisy", "small", "real_fit"]
    for task in ("push", "lift", "pick_place"):
        worst = min(held_out, key=lambda p: CELLS[f"{task}/{p}"]["success_rate"])
        row = f"| {_pm(CELLS[f'{task}/nominal'])} | {_pm(CELLS[f'{task}/{worst}'])} (`{worst}`) |"
        assert row in README, row
    for task, p, cond in (("lift", "nominal", "nominal, 100 ep x 5 seeds"), ("lift", "small", "`small` cube (9 mm"),
                          ("pick_place", "nominal", "nominal |"), ("pick_place", "small", "`small` cube |")):
        assert f"| scripted expert | success, `{task}` | {_pm(CELLS[f'{task}/{p}'])} | {cond}" in README, (task, p)
    push = [CELLS[f"push/{p}"] for p in ["nominal"] + held_out]
    lo, hi = min(r["success_rate"] for r in push), max(r["success_rate"] for r in push)
    slo, shi = min(r["success_std"] for r in push), max(r["success_std"] for r in push)
    assert f"| {lo:.2f} – {hi:.2f} | every column, ± {slo:.2f} – {shi:.2f} seed spread |" in README
