"""Regression gate on a real log: replay one held-out real SO-100 episode with the committed fit.

If a change to the model, the replay or MuJoCo moves the sim away from the real arm, this fails.
Measured 2026-11-20 (mujoco 3.13): fitted 0.345 deg, menagerie defaults 0.423 deg (same offsets).
"""
import json
import os

import numpy as np

from so_arm100_sim import real_replay as R

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = json.load(open(os.path.join(HERE, "..", "runs", "real2sim_fit.json")))
LOG = np.load(os.path.join(HERE, "data", "svla_so100_pickplace_ep40.npz"))
SIGN = np.ones(5)
OFF = np.deg2rad(FIT["offset_deg"])


def _err(params):
    return R.episode_error(params, LOG["action"], LOG["state"], SIGN, OFF)


def test_episode_is_held_out():
    assert 40 not in FIT["train_episodes"]


def test_fitted_servo_tracks_the_real_arm():
    e = _err(R.Params(**FIT["fitted"]))
    assert e.mean() < 0.37, e


def test_fit_still_beats_the_defaults():
    assert _err(R.Params()).mean() - _err(R.Params(**FIT["fitted"])).mean() > 0.05
