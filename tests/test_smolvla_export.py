"""SmolVLA-base layout: units, camera names, feature shapes. No lerobot needed."""
import numpy as np
import pytest

from so_arm100_sim import scene, smolvla_export as sx


def test_jaw_range_matches_the_model():
    m = scene.build_model()
    jid = m.joint("Jaw").id
    assert tuple(np.round(m.jnt_range[jid], 3)) == sx.JAW_RANGE


def test_units_round_trip_and_jaw_ends():
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, (100, 6))
    q[:, scene.JAW] = rng.uniform(*sx.JAW_RANGE, 100)
    np.testing.assert_allclose(sx.from_so100_units(sx.to_so100_units(q)), q, atol=1e-5)
    ends = np.zeros((2, 6))
    ends[:, scene.JAW] = sx.JAW_RANGE
    u = sx.to_so100_units(ends)
    assert u[0, scene.JAW] == pytest.approx(0.0, abs=1e-4)
    assert u[1, scene.JAW] == pytest.approx(100.0, abs=1e-4)
    assert sx.to_so100_units(np.full(6, np.pi / 2))[0] == pytest.approx(90.0)


def test_features_match_smolvla_base_inputs():
    # names and shapes from lerobot/smolvla_base config.json: state [6], action [6],
    # camera1..3 at 3x256x256 (stored HWC here)
    f = sx.features()
    assert f["observation.state"]["shape"] == (6,) and f["action"]["shape"] == (6,)
    cams = sorted(k for k in f if k.startswith("observation.images."))
    assert cams == [f"observation.images.camera{i}" for i in (1, 2, 3)]
    assert all(f[k]["shape"] == (256, 256, 3) for k in cams)
    assert set(sx.CAMERA_MAP.values()) <= set(scene.CAMERAS)


def test_frame_converts_state_slice_and_maps_cameras():
    imgs = {c: np.full((256, 256, 3), i, np.uint8) for i, c in enumerate(scene.CAMERAS)}
    state = np.concatenate([np.full(6, np.pi), np.ones(9)])  # 15-d proprio
    row = sx.frame(state, np.zeros(6), "lift the blue cube", imgs)
    assert row["observation.state"].shape == (6,)
    assert row["observation.state"][0] == pytest.approx(180.0, abs=1e-3)
    assert row["observation.images.camera1"][0, 0, 0] == list(scene.CAMERAS).index("top")
    assert row["task"] == "lift the blue cube"


def test_recorder_round_trip(tmp_path):
    from so_arm100_sim import datasets as D, render
    from so_arm100_sim.env import ArmEnv
    from so_arm100_sim.expert import ScriptedExpert
    if not D.lerobot_available():
        pytest.skip("lerobot not installed")
    if not render.available():
        pytest.skip("no offscreen GL context")
    env = ArmEnv("reach", obs_mode="proprio", action_mode="absolute", seed=0)
    ex = ScriptedExpert(env)
    rec = sx.SmolVLARecorder(tmp_path / "ds", "test/so_arm100_smolvla", height=32, width=48)
    obs = env.reset(seed=0)
    ex.reset()
    frames, q = [], []
    for _ in range(4):
        a = ex.act()
        frames.append({"state": obs["state"], "action": a, "task": "go to the red cube",
                       "images": {c: env.render(c, 32, 48) for c in sx.CAMERA_MAP.values()}})
        q.append(np.asarray(obs["state"][:6]))
        obs, _, _, _ = env.step(a)
    rec.add_episode(frames)
    ds = rec.finalize()
    ds = D.load(tmp_path / "ds", "test/so_arm100_smolvla")
    assert len(ds) == 4
    item = ds[2]
    np.testing.assert_allclose(item["observation.state"].numpy(), sx.to_so100_units(q[2]), atol=1e-4)
    assert item["observation.images.camera3"].shape == (3, 32, 48)
    env.close()
