"""README real2sim numbers must match the saved fit and spread results."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text()
FIT = json.loads((ROOT / "runs/real2sim_fit.json").read_text())
SPREAD = json.loads((ROOT / "runs/real2sim_spread.json").read_text())


def _mean3(xs):
    return f"{sum(xs) / len(xs):.3f}"


def test_heldout_table_matches_fit():
    rows = {
        "menagerie defaults, mid-range zero offsets": "heldout_err_deg_midrange_offsets_defaults",
        "fitted offsets (Pitch, Elbow, Wrist_Pitch)": "heldout_err_deg_fitted_offsets_defaults",
        "fitted offsets + kp": "heldout_err_deg_fitted_all",
    }
    for label, key in rows.items():
        m = re.search(r"^\| " + re.escape(label) + r"[^|]*\| ([0-9.]+) \|$", README, re.M)
        assert m, label
        assert m.group(1) == _mean3(FIT[key]), (label, m.group(1), FIT[key])


def test_fitted_params_in_table():
    f = FIT["fitted"]
    row = (f"kp {f['kp']:.1f}, dampratio {f['dampratio']:.2f}, "
           f"armature {f['armature']:.3f}, frictionloss {f['frictionloss']:.2f}")
    assert row in README, row


def test_spread_sentence_matches_json():
    med, p10, p90 = SPREAD["median"], SPREAD["p10"], SPREAD["p90"]
    kp = f"kp median {med['kp']:.1f} (p10–p90\n{p10['kp']:.1f}–{p90['kp']:.1f})"
    fl = f"frictionloss median {med['frictionloss']:.2f} ({p10['frictionloss']:.2f}–{p90['frictionloss']:.2f})"
    assert kp in README, kp
    assert fl in README, fl
