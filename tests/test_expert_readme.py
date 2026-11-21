"""README expert-success table must match runs/expert_success*.json, cell by cell."""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text()
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
