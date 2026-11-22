"""interval() and paired() decide every 'identified' claim in the docs/DETAILS.md identifiability paragraph."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")     # the script imports it at module level: extra "real2sim"

_spec = importlib.util.spec_from_file_location(
    "ident", Path(__file__).parents[1] / "scripts/real2sim_identifiability.py")
ident = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ident)


def _pts(flags):
    return [{"m": m, "distinguishable": f} for m, f in zip(ident.GRID, flags)]


def test_interval_tight_open_and_gap():
    n, i1 = len(ident.GRID), ident.GRID.index(1.0)
    flags = [True] * n
    flags[i1] = False
    assert ident.interval(_pts(flags)) == [1.0, 1.0]
    flat = [False] * n
    assert ident.interval(_pts(flat)) == [None, None]
    # contiguous run only: a not-distinguishable point beyond a distinguishable one does not widen it
    flags[i1 - 1] = False
    flags[0] = False
    assert ident.interval(_pts(flags)) == [ident.GRID[i1 - 1], 1.0]


def test_paired_needs_mean_above_t_bound():
    c_opt = np.zeros(10)
    assert ident.paired(c_opt + np.linspace(0.009, 0.011, 10), c_opt)[2]
    noisy = np.array([0.05, -0.05] * 5) + 0.001
    assert not ident.paired(noisy, c_opt)[2]
