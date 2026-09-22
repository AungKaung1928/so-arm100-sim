# Issues to open on GitHub

Each block is one issue: title, then body. They are the known gaps at the
first release, so that the repository's own tracker says what is and is not
done rather than the README having to.

---

**Expert success on `small` cube physics is the weakest cell**

The scripted expert's success rate drops most on `EVAL_PHYSICS["small"]`
(a 9 mm half-extent cube). The grasp closes to a 16 mm minimum pad gap, so an
18 mm cube is held by less than a millimetre of squeeze per side. Options:
lower `JAW_CLOSED` toward the joint limit, or accept and document that
`small` is the hardest held-out cell. Numbers: `runs/expert_success.json`.

---

**Push expert has no recovery when the cube rotates off the target line**

The pusher re-aims every step (waypoint follows the cube), but when the
closed pads catch a cube corner the cube yaws and slides sideways faster
than the arm re-aims. A two-sided cage (open jaws either side of the cube,
pushing with the jaw body) would be more robust. Not started.

---

**Wrist camera**

Only fixed cameras exist (`front`, `top`, `side`). A wrist camera on
`Fixed_Jaw` is one `add_camera` call in `scene.py`; nothing consumes it yet.

---

**Throughput sweep is not in CI**

`scripts/bench_throughput.py` needs an idle machine and writes
`runs/throughput.json`; CI runs the tests only. Decide whether a 1-process
smoke row belongs in CI as a regression guard.

---

**gymnasium adapter is untested against `check_env`**

`gym_adapter.make_gym_env` follows the 1.x API but is not run through
`gymnasium.utils.env_checker.check_env` in the tests.
