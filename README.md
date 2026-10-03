# so-arm100-sim

MuJoCo tabletop tasks, a scripted expert, domain randomisation and one
evaluation protocol for the [Standard Open Arm 100](https://github.com/TheRobotStudio/SO-ARM100),
the 5-DOF + gripper arm behind Hugging Face LeRobot. CPU only, no framework.

**Walkthrough:** https://aungkaung1928.github.io/projects/so-arm100.html — the bench and the three policy projects built on it, explained end to end.

## At a glance

The scripted expert placing the red cube on the marker with the other two cubes as distractors (`out/expert_filmstrip.png`):

![expert filmstrip](out/expert_filmstrip.png)

Four tasks, one success rule each, and what the scripted expert scores on them (100 episodes x 5 seeds, mean ± sd across seeds):

| task | success rule | steps | expert, nominal | expert, worst held-out cell |
|---|---|---|---|---|
| `reach` | end effector within 2 cm of a point 3 cm above the cube | 100 | 1.00 ± 0.00 | 1.00 ± 0.00 (all cells) |
| `push` | cube centre within 2.5 cm (xy) of the marker | 150 | 0.76 ± 0.07 | 0.73 ± 0.06 (`small`) |
| `lift` | cube 5 cm above its resting height **and** both jaws touching it | 150 | 0.93 ± 0.02 | 0.73 ± 0.05 (`small`) |
| `pick_place` | cube within 2.5 cm of the marker, on the table, released | 250 | 0.93 ± 0.01 | 0.72 ± 0.05 (`small`) |

This repository is the shared floor under three policy projects. Everything a learned policy is later measured against is defined once, here: the success criteria, the held-out physics, the seeds, the expert's own numbers.

```mermaid
flowchart LR
    B["so-arm100-sim<br/>ArmEnv · VecEnv · DRConfig · EVAL_PHYSICS<br/>ScriptedExpert · evaluate · instructions<br/>63 tests"]
    RL["so-arm100-rl<br/>PPO, reach → push → lift curriculum<br/>with and without DR"]
    IL["so-arm100-il<br/>BC → DAgger → ACT<br/>from scripted demonstrations"]
    VLA["so-arm100-vla<br/>language-conditioned<br/>multitask policy"]
    B -- "ArmEnv, VecEnv, DRConfig,<br/>EVAL_PHYSICS, evaluate" --> RL
    B -- "ScriptedExpert, LeRobotRecorder,<br/>evaluate" --> IL
    B -- "all of the above<br/>+ instructions" --> VLA
```

### Results

| method | metric | value | condition |
|---|---|---|---|
| scripted expert | success, `lift` | 0.93 ± 0.02 | nominal, 100 ep x 5 seeds |
| scripted expert | success, `lift` | 0.73 ± 0.05 | `small` cube (9 mm half-extent) |
| scripted expert | success, `pick_place` | 0.93 ± 0.01 | nominal |
| scripted expert | success, `pick_place` | 0.72 ± 0.05 | `small` cube |
| scripted expert | success, `push` | 0.73 – 0.79 | every column, ± 0.05 – 0.07 seed spread |
| `VecEnv`, 1 process | env-steps/s | 1,954 ± 49 | lift task, 15 s bursts, 3 sweeps |
| `VecEnv`, 8 processes | env-steps/s | 7,302 ± 569 | 15 s bursts, 47% efficiency |
| `VecEnv`, 8 processes | env-steps/s, sustained | 5,361 (5,257 – 5,465) | 12 x 15 s, about 28% below the burst figure |
| Docker image | tests inside | 59 passed, 4 skipped | 423 MB, built 2026-09-23 |

### Key points

- **The small object is the expert's real failure.** Lift and pick_place drop from 0.93 to 0.73 / 0.72 when the cube half-extent shrinks to 9 mm; every other perturbation moves them by at most 0.08, and the 30-episode single-seed development readings had overstated these cells at 0.87 / 0.80.
- **Push is the weakest task everywhere.** It sits at 0.73–0.79 in every column and the seed spread (± 0.05–0.07) covers every column's difference from nominal, so the limit is the scripted push itself, not the physics.
- **Every held-out cell is outside the DR range.** `heavy` 3.0x mass, `weak` kp 0.45, `small` 9 mm, `laggy` 3 steps, `noisy` 3x each exceed the training range in at least one factor and `test_dr.py` asserts it, which is what makes a success rate under `EVAL_PHYSICS` a transfer claim.
- **Budget training from 5,361 env-steps/s, not 7,302.** Eight processes burst at 7,302 ± 569 but settle at 5,361 (band 5,257–5,465) over 12 x 15 s; the first three throughput attempts drifted 25–58% until a 10 s warm-up fixed the low starting clock.
- **The `slippery` column is inert, and so is half the friction range.** Nominal and slippery are bit-identical in every table: MuJoCo gives a contact the larger friction of its two geoms and the fingers and floor stay at 1.0, so 0.3x never reaches the contact and DR 0.5–1.5 only acts above 1.0.

<details><summary><b>What is in the box</b></summary>

| project | trains | consumes from here |
|---|---|---|
| [`so-arm100-rl`](https://github.com/AungKaung1928/so-arm100-rl) | PPO, reach → push → lift curriculum, with and without DR | `ArmEnv`, `VecEnv`, `DRConfig`, `EVAL_PHYSICS`, `evaluate` |
| [`so-arm100-il`](https://github.com/AungKaung1928/so-arm100-il) | BC → DAgger → ACT from scripted demonstrations | `ScriptedExpert`, `LeRobotRecorder`, `evaluate` |
| [`so-arm100-vla`](https://github.com/AungKaung1928/so-arm100-vla) | a language-conditioned multitask policy | all of the above plus `instructions` |

**Status: code complete, tests green, reference numbers measured.**
Every number in this README comes from the named script's JSON in `runs/`, never
typed in by hand.

<p>
<img src="out/camera_front.png" width="320" alt="front camera"> <img src="out/camera_top.png" width="320" alt="top camera">
</p>

```
so_arm100_sim/
  assets/trs_so_arm100/   the Menagerie arm, vendored byte-for-byte (Apache-2.0)
  scene.py                the tabletop: floor, 3 cubes, target marker, 3 cameras, ee site
  env.py                  ArmEnv: 4 tasks, 2 observation modes, 2 action modes
  dr.py                   DRConfig (training ranges) and EVAL_PHYSICS (held-out shifts)
  ik.py                   damped least-squares IK on the end-effector site
  expert.py               ScriptedExpert: IK waypoints for every task
  vec_env.py              N envs in N forked processes
  evaluate.py             the protocol: 100 episodes x 5 seeds per cell, JSON with provenance
  instructions.py         language templates with train / held-out splits
  datasets.py, record.py  LeRobotDataset v3 writer and the demo recorder (optional extra)
  render.py, gym_adapter.py
scripts/                  expert_success, bench_throughput, record_dataset, view_scene
tools/                    probe_model (measured frames), reach_map (where top-down IK works)
tests/                    63 checks; the expert's success gates are among them
```

The arm description is not edited. Everything the tasks need on top of it is
attached in Python through `mujoco.MjSpec`, and `test_scene.py` hashes the
vendored XML so an accidental edit fails CI.

</details>

<details><summary><b>The tasks: success rules, observations, actions</b></summary>

Four tasks, one cube colour each (`red`, `green`, `blue`), optionally with the
other two cubes on the table as distractors. Control at 20 Hz over the model's
2 ms physics step.

| task | success | steps |
|---|---|---|
| `reach` | end effector within 2 cm of a point 3 cm above the cube | 100 |
| `push` | cube centre within 2.5 cm (xy) of the marker | 150 |
| `lift` | cube 5 cm above its resting height **and** both jaws touching it | 150 |
| `pick_place` | cube within 2.5 cm of the marker, on the table, released | 250 |

"Both jaws touching" is there because a cube flung upward otherwise counts
as lifted. Episodes terminate on success; `info` carries `terminated` and
`truncated` separately because a value bootstrap has to tell them apart.

Two observation layouts, chosen at construction:

- `state` (25-d, privileged): joint angles and velocities, end-effector
  position, active cube position, cube minus end effector, target position,
  jaw gap. For state-based RL.
- `proprio` (15-d): joint angles and velocities, end-effector position. What
  the real arm knows about itself. Objects come from the image. For image
  policies.

Two action conventions, both 6-d: `delta` in [-1, 1] added to the previous
joint target (a zero action holds still), or `absolute` joint targets in
radians, which is what a recorded dataset stores and what the real arm takes.

</details>

<details><summary><b>Domain randomisation and the held-out physics</b></summary>

`DRConfig` is what a training run samples at every reset. `EVAL_PHYSICS` is
a fixed set of shifts a trained policy is evaluated on, and every one of
them is **outside** the DR range in at least one factor -- `test_dr.py`
asserts it. That is what makes a success rate under `EVAL_PHYSICS` a transfer
claim and not an interpolation claim.

| factor | DR range (x nominal) | held-out cell |
|---|---|---|
| cube mass | 0.5 – 2.0 | `heavy` 3.0 |
| cube sliding friction | 0.5 – 1.5 (only > 1.0 acts, see below) | `slippery` 0.3 (inert, see below) |
| servo gain kp | 0.6 – 1.4 | `weak` 0.45 |
| joint damping / frictionloss | 0.5 – 2.0 | -- |
| cube half-extent | 10 – 15 mm | `small` 9 mm |
| action latency | 0 – 2 steps | `laggy` 3 |
| observation noise (rad, m) | 0.005, 0.003 | `noisy` 3x |

`kp` scales the position actuator's gain and position bias together and
leaves the velocity bias alone, so a weaker motor is also a more damped one,
which is what a weaker motor does.

</details>

<details><summary><b>The scripted expert and its measured success</b></summary>

IK waypoints per task, rate limited to 0.05 rad per step so the actions are
smooth enough to learn from. It reads the cube's true pose from the
simulator -- that is the privileged information a learned policy has to
recover from the image, and it is why the expert is a demonstrator and a
reference, not a controller a real robot would run.

Two things it had to do that a first draft did not:

- **Resolve the 180-degree grasp ambiguity.** A parallel jaw closing along
  yaw θ is the same grasp as θ + π, but only one branch may be inside the
  wrist-roll limit from where the arm is. Both are solved, the better one
  wins, and the branch is held for the episode so the wrist does not flip
  mid-approach.
- **Close the loop on the end effector.** The position actuators are plain
  PD with no gravity compensation; the arm settles 1–2 cm from the pose IK
  asked for, more under a weakened `kp`. The expert integrates the Cartesian
  error near the waypoint and shifts its IK target by it, with anti-windup.
  Without this the descending open jaw landed on the cube and pushed it away
  instead of straddling it.

One defect the tests caught before any policy did: the action-latency
buffer held one entry too many, so every command reached the servos a step
late even at zero latency. `test_kp_affects_tracking` failed because a step
command produced no motion inside the step that issued it. The expert had
been succeeding through it.

Where a top-down grasp is reachable at all is narrower than where the
fingertip can go (`tools/reach_map.py`), and that map, not the arm's reach,
is what set the spawn box and the lift height.

### Expert success, 100 episodes x 5 seeds per cell

`python scripts/expert_success.py` → `runs/expert_success.json`

| task | nominal | heavy | slippery | weak | laggy | noisy | small |
|---|---|---|---|---|---|---|---|
| reach | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| push | 0.76 ± 0.07 | 0.79 ± 0.07 | 0.76 ± 0.07 | 0.74 ± 0.05 | 0.74 ± 0.05 | 0.76 ± 0.07 | 0.73 ± 0.06 |
| lift | 0.93 ± 0.02 | 0.90 ± 0.02 | 0.93 ± 0.02 | 0.97 ± 0.02 | 0.92 ± 0.03 | 0.93 ± 0.02 | 0.73 ± 0.05 |
| pick_place | 0.93 ± 0.01 | 0.85 ± 0.02 | 0.93 ± 0.01 | 0.96 ± 0.01 | 0.90 ± 0.04 | 0.93 ± 0.01 | 0.72 ± 0.05 |

Mean ± standard deviation of the per-seed success rate across the 5 seeds.

What the table says:

- **The small object is the expert's real failure.** Lift and pick_place drop
  from 0.93 to 0.73 / 0.72 when the cube shrinks. Every other perturbation
  moves them by at most 0.08.
- **Push is the weakest task everywhere.** It sits at 0.73–0.79 in every
  column, and the seed spread (± 0.05–0.07) covers every column's difference
  from nominal. The limit is the scripted push itself, not the physics.
- **Heavy costs pick_place 0.08 but lift only 0.03.** The extra mass matters
  during the carry, not the grasp.
- **Weak actuators score slightly higher** on lift and pick_place (0.97 /
  0.96). The slower approach knocks the cube less.
- **The `slippery` column is inert, and so is half the friction range.**
  Nominal and slippery are bit-identical here and in every downstream table
  (RL and imitation). `dr.py` scales the cube geom's sliding friction, but
  MuJoCo gives a contact the larger friction of its two geoms, and the
  fingers and floor stay at 1.0. So 0.3x never reaches the contact, and the
  DR range 0.5–1.5 only acts above 1.0. Found while writing up the
  imitation results; the column is kept as measured. Fixing it (scale the
  finger and floor friction too, or give the cube a higher contact
  `priority`) changes the bench and would be versioned as a new experiment.

The development readings (30 episodes, one seed) set the test gates in
`tests/test_expert.py`. They overstated the small object: 0.87 lift and
0.80 pick_place, against the measured 0.73 and 0.72. The seed spread above
is why one seed at 30 episodes is not a reference. The gates only run on the
nominal task and still sit below the measured values.

</details>

<details><summary><b>Throughput</b></summary>

`nice -n 10 python scripts/bench_throughput.py --tag sN` → `runs/throughput_sN.json`

One env step = 25 physics steps + `mj_forward` + observation and reward.
Bare `mj_step` on this model measured 18 µs single-process during
development, i.e. about 2,200 env-steps/s per process before any Python
overhead.

Lift task, laptop CPU under WSL2 capped at 8 threads, on mains power, Windows
"Balanced" power plan, a browser open. Mean ± sd over three stable sweeps
(`runs/throughput_s1/s2/s3.json`):

| processes | env-steps/s | efficiency |
|---|---|---|
| 1 | 1,954 ± 49 | 100% |
| 2 | 3,691 ± 34 | 94% |
| 4 | 5,918 ± 358 | 76% |
| 8 | 7,302 ± 569 | 47% |

Each sweep brackets every row with a single-process reference and marks the
JSON `stable: false` if the reference drifts more than 10% or trends
monotonically. The kept sweeps drifted at most +6.9%, +3.5% and +7.6%.

**Warm-up protocol.** The first three attempts drifted 25–58% and are kept
as `runs/discarded_throughput_*_unstable.json`. The cause was the power plan:
on "Balanced" the CPU starts at a low clock, so the first reference read low
and every later row looked faster. A 10 s, 8-process warm-up before the sweep,
started once the load average was below 0.5, fixed it. One later attempt still
drifted −15.9% and was discarded as well.

**Sustained rate is lower.** The sweep rows are 15 s bursts. An 8-process run
held for 12 × 15 s (`runs/sustained.json`) gave 7,432 in the first window and
then settled at **5,361** (band 5,257–5,465), about 28% lower. WSL2 cannot read the CPU
temperature, so thermal or power throttling is the likely cause but is not
measured. Budget training from 5,361, not 7,302.

</details>

## Reproducing

```
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # mujoco + numpy + pytest
./verify.sh                        # tests, camera PNGs, and the two measurement commands
```

Optional extras: `.[lerobot]` for the dataset writer (Python ≥ 3.12),
`.[gym]` for the gymnasium adapter, `.[image]` for PNG output from the
scripts. Rendering needs a GL context: `MUJOCO_GL=glfw` with a display, or
`egl` / `osmesa` headless. Tests that render skip cleanly without one.

```
docker build -t so-arm100-sim . && docker run --rm so-arm100-sim
```

Image built on 2026-09-23 and its default command passed inside it (59 tests passed, 4 skipped), image size 423 MB.

<details><summary><b>Recording demonstrations, and the limits</b></summary>

### Recording demonstrations

```
python -m so_arm100_sim.record --root data/lift_red --repo-id local/so_arm100_lift_red \
    --tasks lift:red --episodes 50 --cameras front
```

Only successful expert episodes are kept. Each frame stores the 15-d proprio
state, one 96x128 image per camera, the expert's absolute joint targets as
the action, and an instruction sampled from the **training** templates in
`instructions.py`. The held-out paraphrases and the held-out (task, colour)
combinations are never written, so "unseen" means unseen.

### Limits, stated

- Simulation only. No number here is a claim about the physical SO-ARM100.
- Fixed cameras only; no wrist camera yet.
- The expert is IK-scripted and fails in ways a human demonstrator would not
  (it re-aims a push but cannot recover a cube that has yawed off the line).
- `kp` randomisation changes the damping ratio as a side effect; that is a
  choice, documented above, not an oversight.
- Rendering on the development machine is software GL at roughly 20 ms per
  96x128 frame, which bounds image-based data collection, not the physics.

</details>

## Licence

MIT for the code in this repository. The robot description under
`so_arm100_sim/assets/trs_so_arm100/` is from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache-2.0, commit `822c2d8`, unchanged.
