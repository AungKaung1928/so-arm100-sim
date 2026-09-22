import json
import os

import numpy as np

from so_arm100_sim.evaluate import evaluate, format_gap_table, save_json
from so_arm100_sim.expert import ScriptedExpert


class Still:
    def reset(self):
        pass

    def act(self, obs):
        return np.zeros(6)


def test_protocol_shape_and_reproducibility():
    r1 = evaluate(lambda env: Still(), "reach", n_episodes=3, seeds=(0, 1))
    r2 = evaluate(lambda env: Still(), "reach", n_episodes=3, seeds=(0, 1))
    for k in ("task", "physics", "n_episodes", "seeds", "success_rate", "success_std",
              "per_seed", "return_mean", "steps_mean", "so_arm100_sim", "model_commit", "date"):
        assert k in r1
    assert r1["per_seed"] == r2["per_seed"] and r1["return_mean"] == r2["return_mean"]
    assert r1["success_rate"] == 0.0 and len(r1["per_seed"]) == 2


def test_expert_scores_on_reach_and_table_formats(tmp_path):
    r = evaluate(ScriptedExpert, "reach", n_episodes=4, seeds=(0,),
                 env_kwargs={"action_mode": "absolute"})
    assert r["success_rate"] == 1.0
    assert r["first_success_p50"] is not None
    p = save_json(os.path.join(tmp_path, "x", "r.json"), r)
    assert json.load(open(p))["task"] == "reach:red"
    table = format_gap_table({"nominal": r, "heavy": r})
    assert "| heavy |" in table and "+0.000" in table
