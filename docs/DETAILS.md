# so-arm100-sim: full write-up

The short version is the [README](../README.md). The tests in `tests/test_*_readme.py` check the numbers here against the code and `runs/*.json`.

## At a glance

The scripted expert placing the red cube on the marker with the other two cubes as distractors (`out/expert_filmstrip.png`):

![expert filmstrip](../out/expert_filmstrip.png)

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
| `VecEnv`, 8 processes | env-steps/s, sustained | 5,361 (5,257 – 5,465) | 12 x 15 s, about 28% below its first window |
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
  smolvla_export.py       same demos in the smolvla_base layout: cameras camera1-3, SO-100 units (zero-shot only)
  render.py, gym_adapter.py
scripts/                  expert_success, bench_throughput, record_dataset, view_scene
tools/                    probe_model (measured frames), reach_map (where top-down IK works)
tests/                    68 checks; the expert's success gates are among them
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
| joint damping / frictionloss | 0.5 – 2.0 | `real_fit`: frictionloss 2.5 with kp 0.78 (fitted to real data, see Real2sim) |
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

`python scripts/expert_success.py` → `runs/expert_success.json`; the `real_fit` column:
`--physics real_fit --tag realfit` → `runs/expert_success_realfit.json` (added 2026-11-20, same env and expert code)

| task | nominal | heavy | slippery | weak | laggy | noisy | small | real_fit |
|---|---|---|---|---|---|---|---|---|
| reach | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| push | 0.76 ± 0.07 | 0.79 ± 0.07 | 0.76 ± 0.07 | 0.74 ± 0.05 | 0.74 ± 0.05 | 0.76 ± 0.07 | 0.73 ± 0.06 | 0.75 ± 0.05 |
| lift | 0.93 ± 0.02 | 0.90 ± 0.02 | 0.93 ± 0.02 | 0.97 ± 0.02 | 0.92 ± 0.03 | 0.93 ± 0.02 | 0.73 ± 0.05 | 0.97 ± 0.01 |
| pick_place | 0.93 ± 0.01 | 0.85 ± 0.02 | 0.93 ± 0.01 | 0.96 ± 0.01 | 0.90 ± 0.04 | 0.93 ± 0.01 | 0.72 ± 0.05 | 0.97 ± 0.01 |

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

<details><summary><b>Real2sim: fitting the sim arm to a real SO-100</b></summary>

Data: the public real dataset `lerobot/svla_so100_pickplace` (50 episodes, 30 fps).
`action` is the leader-arm command, `observation.state` the follower position, both in
degrees. `so_arm100_sim/real_replay.py` feeds each command to the MuJoCo arm for one frame
and scores the mean |sim − real| per joint. A trajectory made by the sim itself replays with
zero error (`tests/test_real_replay.py`).

Measured in the data before any fit: the follower matches the command best 3 frames (100 ms)
later, and when the command is still the shoulder sits 0.46 ± 0.09 deg below it (gravity on
a position loop).

`scripts/fit_real2sim.py` (`pip install -e ".[real2sim]"`, about 2.5 min on one core) fits in
two Nelder-Mead stages on episodes 0–9 and scores episodes 40–49 (`runs/real2sim_fit.json`):

| held-out, mean over 5 arm joints | error (deg) |
|---|---|
| menagerie defaults, mid-range zero offsets | 0.370 |
| fitted offsets (Pitch, Elbow, Wrist_Pitch) | 0.349 |
| fitted offsets + kp 38.8, dampratio 0.76, armature 0.103, frictionloss 0.25 | 0.294 |

Limits. LeRobot puts degree zero at the middle of each operator's calibrated range, so the
zero offsets are fitted, not known; the Elbow offset ends on its range bound. Stage 2 stopped
at its evaluation cap (not converged), and passive damping came out 0.20, 0.018 and 0.11 in
three runs: the data does not pin it down.

What the data can identify (`scripts/real2sim_identifiability.py`, 55 min on one core →
`runs/real2sim_identifiability.json`). Restarting Nelder-Mead from the fit converges in 537
evaluations: train error 0.264 → 0.254 deg, held-out 0.294 → 0.291 deg. kp moves 38.8 → 36.6,
frictionloss 0.25 → 0.23, armature 0.103 → 0.076, while dampratio drops 0.76 → 0.32 and passive
damping rises 0.11 → 1.97: the two velocity terms trade against each other. Profile per parameter
(hold it at 0.5× or 2× the optimum, refit the other four, train episodes 0–9), increase in train error:

| parameter | 0.5× (deg) | 2× (deg) |
|---|---|---|
| kp | +0.127 | +0.050 |
| frictionloss | +0.019 | +0.040 |
| armature | +0.007 | +0.019 |
| dampratio | +0.004 | +0.007 |
| damping | +0.006 | +0.279 |

For scale, fitting the dynamics bought 0.055 deg held-out (0.349 → 0.294). kp is identified
(0.25× costs +0.720 deg). Frictionloss and armature are statistically resolved to within ±19 %
(paired t over episodes), but a factor-2 error in either costs at most 0.040 deg. The data bounds
total velocity damping from above (damping 4× costs +2.744 deg) and not from below, and does not
separate actuator damping (dampratio) from passive joint damping. Refitting the zero offsets at
the converged dynamics puts none on its bound (Elbow 37.73 deg, bound 36.52; held-out 0.288 deg).
On the stacking dataset (slice around `runs/real2sim_transfer.json`, others fixed,
`runs/real2sim_identifiability_stacking.json`) damping moves the error by under 0.006 deg over 0.5–2×.
Not identifiable from this data at all: anything that needs a torque or current signal.

Refitting kp, dampratio, armature and frictionloss on each of episodes 10–29 alone (offsets
fixed; `scripts/real2sim_spread.py` → `runs/real2sim_spread.json`): kp median 35.7 (p10–p90
32.6–38.7), frictionloss median 0.25 (0.22–0.28). kp sits inside the DR range
(0.6–1.4 × 50); frictionloss is 2.2–2.8 × the nominal 0.1, outside the DR range (0.5–2.0 ×),
so a policy trained with the current DR never sees friction this high. `EVAL_PHYSICS["real_fit"]`
(kp 0.78, frictionloss 2.5) makes it a held-out cell. The scripted expert is not hurt by it
(lift and pick_place 0.97, push 0.75, nominal 0.93 / 0.93 / 0.76). The two PPO lift policies in
[so-arm100-rl](https://github.com/AungKaung1928/so-arm100-rl) score 0.996 / 0.992 there (100 × 5).

`DRConfig.real_informed()` is the default config with frictionloss widened to 0.5–3.5 ×, so it
covers every per-episode fit (kp 0.64–0.89 ×, frictionloss 2.1–3.0 ×; `tests/test_dr.py` checks
this against `runs/real2sim_spread.json`). For a policy trained with it, `real_fit` is in range:
an interpolation cell, not a transfer cell. No policy has been trained with it yet.

`DRConfig.real_centred()` makes the opposite trade: kp and frictionloss narrowed onto the p10–p90
band of the same fits (kp 0.65–0.77 ×, frictionloss 2.2–2.8 ×; `tests/test_dr.py` derives the band
from `runs/real2sim_spread.json`), so `nominal` is outside it. The paired A/B in
[so-arm100-rl](https://github.com/AungKaung1928/so-arm100-rl) (3 training seeds, numbers and test
there) finds no measurable difference against the default DR on `nominal` or `real_fit`, and a lead
on `laggy` only.

Transfer to a second real dataset, `lerobot/svla_so100_stacking` (`--dynamics-from`, `runs/real2sim_transfer.json`):
offsets refit on its episodes 0–9, dynamics taken unchanged from the pick-place fit, scored on
episodes 46–55. Mean error 0.463 (defaults, mid-range offsets) → 0.439 (own offsets) → 0.342 deg
(pick-place dynamics). The stacking offsets differ from the pick-place ones (Elbow 77 vs 37 deg)
and two of them end on a range bound, so the zero convention is not shared between the two
recordings, or the offsets are poorly constrained by these motions.

Refitting the dynamics on the stacking data itself (`scripts/real2sim_identifiability.py` with
`--fit runs/real2sim_transfer.json`, its episodes 0–9, 33 min on one core →
`runs/real2sim_identifiability_stacking_full.json`) converges in 116 evaluations to kp 22.0,
frictionloss 0.18, armature 0.057; held-out 0.342 → 0.319 deg. Within this dataset kp is again
sharply identified (0.5× costs +0.232 deg), yet it sits 40 % below the pick-place value (36.6),
outside that dataset's per-episode p10–p90 band (32.6–38.7) and below the DR floor (0.6 × 50).
Per-episode refits on stacking episodes 10–29 (`scripts/real2sim_spread.py`, same start as
pick-place → `runs/real2sim_spread_stacking.json`) put kp at median 21.5, p10–p90 18.9–25.0, no
overlap with pick-place, so the gap is in the data, not one soft fit; frictionloss overlaps
(0.17–0.25). Passive damping is flat: 0.25×–4× moves the error by under 0.001 deg. The Elbow offset refits onto
its range bound, and gravity sag trades kp against a zero offset, so the data cannot say whether
the gap is a different arm, a different calibration, or offset error. Either way one dataset's
kp is not the SO-100's kp: `real_centred()` (kp 0.65–0.77×) is built from pick-place only and
does not cover this fit.

Related: mujoco_menagerie issue #368 (opened 2026-11-17) fits STS3215 servo actuators for the
SO-101 model against 12 real arms and 10 public LeRobot SO-101 datasets, with tip-space error.
This section uses the SO-100 model and joint-space error, so the numbers are not comparable.

CI gate on a real log: `tests/test_real_regression.py` replays held-out episode 40 (committed as
`tests/data/svla_so100_pickplace_ep40.npz`, 9 KB, Apache-2.0) with the committed fit and fails above
0.37 deg (measured 0.345; defaults 0.423). Injected regressions it catches: gain 20 → 0.682, gain 80 →
0.611, 2-frame command delay → 0.758 deg. One arm, one dataset, 10 training episodes. The commit
message of 9bb9e13 gives the baseline as 0.331; the correct value is 0.370 (table above).

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
