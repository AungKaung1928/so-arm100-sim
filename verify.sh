#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tiered. Tier 1 is the test suite: model assumptions, the environment
# contract, IK, the scripted expert's success gates, DR, the vector env. It
# needs mujoco and numpy only and runs in about a minute on one core. Tier 2
# renders the cameras (needs a GL context). Tier 3 measures the expert's
# success table and tier 4 the throughput sweep; both write runs/*.json, and
# tier 4 is the only one that loads the machine.
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import mujoco, numpy' 2>/dev/null; then
  echo "mujoco/numpy are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install -e '.[dev]'" >&2
  exit 1
fi
if ! "$PY" -c 'import so_arm100_sim' 2>/dev/null; then
  echo "so_arm100_sim is not importable. From the repo root:  pip install -e '.[dev]'" >&2
  exit 1
fi

# Defaults are what every number in the README was measured with. Both
# honour whatever is already exported.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

hr() { printf '\n=== %s ===\n' "$1"; }

hr "1/4  tests -- model, contract, IK, expert gates, DR, vector env"
"$PY" -m pytest tests || exit 1

hr "2/4  cameras -- one PNG per camera, and an expert filmstrip"
if "$PY" -c 'from so_arm100_sim import render; import sys; sys.exit(0 if render.available() else 1)'; then
  "$PY" scripts/view_scene.py --out out || exit 1
else
  echo "skipped: no offscreen GL context (set MUJOCO_GL=glfw with a display, or egl/osmesa)"
fi

hr "3/4  scripted expert success, all tasks x held-out physics"
cat <<'MSG'
Single core, a few minutes. This is the reference table in the README:

    python scripts/expert_success.py --episodes 100 --seeds 5

A quick version (20 episodes, 1 seed, ~1 min):

    python scripts/expert_success.py --episodes 20 --seeds 1 --tag quick
MSG

hr "4/4  throughput -- env-steps/s at 1/2/4/8 processes"
cat <<'MSG'
Loads the machine for a few minutes. Close other work first. The sweep
brackets every row with a single-process reference and refuses to certify
the table if that reference drifts more than 10%.

    nice -n 10 python scripts/bench_throughput.py --seconds 15
MSG
if [ -f runs/throughput.json ]; then
  "$PY" - <<'PY'
import json
b = json.load(open("runs/throughput.json"))
print(f"\nlast recorded sweep ({b['task']}): stable={b['stable']}")
for r in b["rows"]:
    print(f"  {r['procs']} proc  {r['env_steps_per_s']:>8,.0f} env-steps/s  {100*r['efficiency']:3.0f}%")
PY
fi
