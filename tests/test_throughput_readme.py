"""README throughput numbers must match runs/throughput_s{1,2,3}.json and runs/sustained.json."""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = (ROOT / "docs" / "DETAILS.md").read_text()
SWEEPS = [json.loads((ROOT / f"runs/throughput_s{i}.json").read_text())["rows"] for i in (1, 2, 3)]
SUS = json.loads((ROOT / "runs/sustained.json").read_text())


def _rates():
    by = {}
    for rows in SWEEPS:
        for r in rows:
            by.setdefault(r["procs"], []).append(r["env_steps_per_s"])
    return by


def test_scaling_table():
    by = _rates()
    one = st.mean(by[1])
    for p, v in by.items():
        row = f"| {p} | {st.mean(v):,.0f} ± {st.stdev(v):,.0f} | {st.mean(v) / (p * one):.0%} |"
        assert row in README, row


def test_summary_and_sustained():
    by = _rates()
    # anchored on the row label: the bare "| mean ± sd |" also occurs in the scaling table
    for p, label in ((1, "1 process"), (8, "8 processes")):
        assert f"| `VecEnv`, {label} | env-steps/s | {st.mean(by[p]):,.0f} ± {st.stdev(by[p]):,.0f} |" in README, p
    eff = st.mean(by[8]) / (8 * st.mean(by[1]))
    assert f"| {st.mean(by[8]):,.0f} ± {st.stdev(by[8]):,.0f} | 15 s bursts, {eff:.0%} efficiency |" in README
    lo, hi = SUS["plateau_band"]
    assert f"| {SUS['plateau_mean']:,.0f} ({lo:,.0f} – {hi:,.0f}) | {len(SUS['rates'])} x 15 s" in README
    drop = f"{1 - SUS['plateau_mean'] / SUS['rates'][0]:.0%}"
    assert f"about {drop} below its first window" in README
    assert f"gave {SUS['rates'][0]:,.0f} in the first window" in README
    assert f"about {drop} lower" in README
