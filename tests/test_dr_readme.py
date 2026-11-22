"""The README DR table (ranges and held-out cells) is what DRConfig and EVAL_PHYSICS hold."""
import os

import pytest

from so_arm100_sim.dr import DRConfig, EVAL_PHYSICS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md")).read()


def r(t):
    return f"{t[0]} – {t[1]}"


def test_dr_table_matches_code():
    d, e = DRConfig(), EVAL_PHYSICS
    assert d.damping == d.frictionloss                          # one README row for both
    assert (e["noisy"]["obs_noise_q"], e["noisy"]["obs_noise_pos"]) == pytest.approx((3 * d.obs_noise_q, 3 * d.obs_noise_pos))
    lines = [
        f"| cube mass | {r(d.cube_mass)} | `heavy` {e['heavy']['cube_mass']} |",
        f"| cube sliding friction | {r(d.cube_friction)} (only > 1.0 acts, see below) | `slippery` {e['slippery']['cube_friction']} (inert, see below) |",
        f"| servo gain kp | {r(d.kp)} | `weak` {e['weak']['kp']} |",
        f"| joint damping / frictionloss | {r(d.damping)} | `real_fit`: frictionloss {e['real_fit']['frictionloss']} with kp {e['real_fit']['kp']} (fitted to real data, see Real2sim) |",
        f"| cube half-extent | {round(d.cube_size[0] * 1000)} – {round(d.cube_size[1] * 1000)} mm | `small` {round(e['small']['cube_size'] * 1000)} mm |",
        f"| action latency | {r(d.latency)} steps | `laggy` {e['laggy']['latency']} |",
        f"| observation noise (rad, m) | {d.obs_noise_q}, {d.obs_noise_pos} | `noisy` 3x |",
    ]
    for line in lines:
        assert line in README, line
