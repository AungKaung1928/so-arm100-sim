"""README identifiability numbers must match runs/real2sim_identifiability*.json."""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = " ".join((ROOT / "README.md").read_text().split())
ID = json.loads((ROOT / "runs/real2sim_identifiability.json").read_text())
STACK = json.loads((ROOT / "runs/real2sim_identifiability_stacking.json").read_text())
SPREAD = json.loads((ROOT / "runs/real2sim_spread.json").read_text())


def _inc(params, name, m, kind="profile"):
    return next(p["increase_deg"] for p in params[name][kind] if p["m"] == m)


def test_profile_table_matches_json():
    for name in ["kp", "frictionloss", "armature", "dampratio", "damping"]:
        row = f"| {name} | +{_inc(ID['params'], name, 0.5):.3f} | +{_inc(ID['params'], name, 2.0):.3f} |"
        assert row in README, row


def test_convergence_sentence_matches_json():
    c = ID["convergence"]
    assert f"converges in {c['restarts'][-1]['nfev']}" in README
    assert c["restarts"][-1]["converged"]
    assert f"train error {c['train_err_deg_fit']:.3f} → {ID['train_err_deg_at_optimum']:.3f} deg" in README
    assert f"held-out {c['heldout_err_deg_fit']:.3f} → {c['heldout_err_deg_converged']:.3f} deg" in README
    f, v = c["fit"], c["converged"]
    assert f"kp moves {f['kp']:.1f} → {v['kp']:.1f}" in README
    assert f"dampratio drops {f['dampratio']:.2f} → {v['dampratio']:.2f}" in README
    assert f"damping rises {f['damping']:.2f} → {v['damping']:.2f}" in README


def test_claims_match_json():
    p = ID["params"]
    assert f"0.25× costs +{_inc(p, 'kp', 0.25):.3f} deg" in README
    assert f"damping 4× costs +{_inc(p, 'damping', 4.0):.3f} deg" in README
    # "resolved within ±19 %": the not-distinguishable run around m=1 is no wider than 0.84-1.19
    for name in ["frictionloss", "armature"]:
        assert p[name]["profile_interval_x"] == [1.0, 1.0], name
    assert p["dampratio"]["profile_interval_x"][0] < 1.0     # the one that is not resolved
    assert max(_inc(p, n, m) for n in ["frictionloss", "armature"] for m in (0.5, 2.0)) <= 0.040
    o = ID["offsets"]
    assert not any(o["refit_at_bound"].values())
    assert f"Elbow {o['refit_deg']['Elbow']:.2f} deg, bound {o['allowed_deg']['Elbow'][0]:.2f}" in README
    assert f"held-out {o['heldout_err_deg_refit']:.3f} deg)" in README


def test_stacking_damping_flat():
    assert max(abs(_inc(STACK["params"], "damping", m, "slice")) for m in (0.5, 0.71, 0.84, 1.19, 1.41, 2.0)) < 0.006


def test_stacking_full_paragraph_matches_json():
    S = json.loads((ROOT / "runs/real2sim_identifiability_stacking_full.json").read_text())
    c, p = S["convergence"], S["params"]
    v = c["converged"]
    assert c["restarts"][-1]["converged"]
    assert f"converges in {c['restarts'][-1]['nfev']} evaluations to kp {v['kp']:.1f}," in README
    assert f"frictionloss {v['frictionloss']:.2f}, armature {v['armature']:.3f}; held-out" in README
    assert f"held-out {c['heldout_err_deg_fit']:.3f} → {c['heldout_err_deg_converged']:.3f} deg" in README
    assert f"(0.5× costs +{_inc(p, 'kp', 0.5):.3f} deg)" in README
    pk = ID["convergence"]["converged"]["kp"]
    assert f"sits {100 * (1 - v['kp'] / pk):.0f} % below the pick-place value ({pk:.1f})" in README
    lo, hi = SPREAD["p10"]["kp"], SPREAD["p90"]["kp"]
    assert f"p10–p90 band ({lo:.1f}–{hi:.1f})" in README and v["kp"] < lo and v["kp"] < 0.6 * 50
    assert max(abs(_inc(p, "damping", m)) for m in (0.25, 0.5, 0.71, 0.84, 1.19, 1.41, 2.0, 4.0)) < 0.001
    assert S["offsets"]["refit_at_bound"]["Elbow"]


def test_stacking_spread_sentence_matches_json():
    S2 = json.loads((ROOT / "runs/real2sim_spread_stacking.json").read_text())
    med, lo, hi = S2["median"], S2["p10"], S2["p90"]
    assert f"kp at median {med['kp']:.1f}, p10–p90 {lo['kp']:.1f}–{hi['kp']:.1f}, no overlap" in README
    assert hi["kp"] < SPREAD["p10"]["kp"]                          # "no overlap with pick-place"
    assert f"frictionloss overlaps ({lo['frictionloss']:.2f}–{hi['frictionloss']:.2f})" in README
    assert lo["frictionloss"] <= SPREAD["p90"]["frictionloss"] and SPREAD["p10"]["frictionloss"] <= hi["frictionloss"]
