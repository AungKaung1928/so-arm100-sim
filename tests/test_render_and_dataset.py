"""Rendering and the LeRobot writer. Both skip cleanly where they cannot run:
no GL context on a headless runner, no lerobot in the core install."""
import numpy as np
import pytest

from so_arm100_sim import render, scene
from so_arm100_sim.env import ArmEnv

needs_gl = pytest.mark.skipif(not render.available(), reason="no offscreen GL context")


@needs_gl
def test_image_observation_shape_and_content():
    env = ArmEnv("lift:blue", obs_mode="proprio", seed=0,
                 image={"camera": "front", "height": 48, "width": 64})
    obs = env.reset()
    img = obs["image"]
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    assert img.std() > 5.0, "a flat image means the camera is not looking at the scene"
    env.close()


@needs_gl
def test_active_cube_is_visible_and_parked_ones_are_not():
    """Bluish pixels exist and reddish ones do not when only the blue cube is
    on the table. Relative thresholds: the lit top face of a 25 mm cube seen
    from 0.8 m is a dozen pixels of pale blue, not saturated blue."""
    env = ArmEnv("lift:blue", seed=1, image={"camera": "top", "height": 96, "width": 128})
    img = env.reset()["image"].astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    bluish = ((b > r + 40) & (b > g + 20)).sum()
    reddish = ((r > b + 40) & (r > g + 20)).sum()
    assert bluish >= 8 and reddish == 0, (bluish, reddish)
    env.close()


@needs_gl
def test_all_cameras_render():
    env = ArmEnv("push", seed=0)
    for cam in scene.CAMERAS:
        img = env.render(cam, 32, 32)
        assert img.shape == (32, 32, 3)
    env.close()


def test_lerobot_writer_round_trip(tmp_path):
    from so_arm100_sim import datasets as D
    if not D.lerobot_available():
        pytest.skip("lerobot not installed")
    if not render.available():
        pytest.skip("no offscreen GL context")
    from so_arm100_sim.expert import ScriptedExpert
    env = ArmEnv("reach", obs_mode="proprio", action_mode="absolute", seed=0,
                 image={"camera": "front", "height": 32, "width": 48})
    ex = ScriptedExpert(env)
    rec = D.LeRobotRecorder(tmp_path / "ds", "test/so_arm100_reach", fps=20,
                            cameras=("front",), height=32, width=48)
    n = 0
    for ep in range(2):
        obs = env.reset(seed=ep)
        ex.reset()
        frames = []
        for _ in range(6):
            a = ex.act()
            frames.append({"state": obs["state"], "action": a, "task": "go to the red cube",
                           "images": {"front": obs["image"]}})
            obs, _, done, _ = env.step(a)
            if done:
                break
        rec.add_episode(frames)
        n += len(frames)
    ds = rec.finalize()
    assert rec.n_episodes == 2 and rec.n_frames == n
    ds2 = D.load(tmp_path / "ds", "test/so_arm100_reach")
    assert len(ds2) == n
    item = ds2[0]
    assert tuple(item["action"].shape) == (6,)
    assert tuple(item["observation.state"].shape) == (15,)
    assert item["observation.images.front"].shape[-2:] == (32, 48)
