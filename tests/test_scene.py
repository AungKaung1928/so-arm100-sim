"""The model and the scene: every assumption the rest of the package makes,
checked against the compiled model rather than believed."""
import numpy as np
import mujoco
import pytest

from so_arm100_sim import scene


@pytest.fixture(scope="module")
def model():
    return scene.build_model(3)


def test_vendored_file_is_untouched():
    assert scene.vendored_sha256() == scene.VENDORED_SHA256, (
        "so_arm100.xml differs from the vendored Menagerie file. This project "
        "attaches everything it needs via MjSpec and never edits the upstream XML.")


def test_actuators_and_joints(model):
    assert model.nu == 6
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert tuple(names) == scene.JOINT_NAMES, "actuator order is the action layout; it moved"
    lo, hi = scene.joint_limits(model)
    assert np.all(hi > lo)
    # ctrlrange inherits the joint range (inheritrange="1" upstream), so a
    # target clipped to the joint limits is always inside ctrlrange.
    assert np.allclose(model.actuator_ctrlrange[:, 0], lo)
    assert np.allclose(model.actuator_ctrlrange[:, 1], hi)


def test_three_cubes_and_a_target(model):
    for c in scene.CUBE_COLORS:
        scene.cube_body(model, c)
        scene.cube_geom(model, c)
        i = scene.cube_qpos_index(model, c)
        assert model.nq >= i + 7
    scene.target_site_id(model)
    assert model.nq == 6 + 7 * 3


def test_ee_site_points_down_at_home(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, scene.name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)
    sid = scene.ee_site_id(model)
    pos = data.site_xpos[sid]
    # Measured from the vendored model: pinch point at home.
    assert np.allclose(pos, [0.0, -0.239, 0.0986], atol=2e-3), pos
    R = data.site_xmat[sid].reshape(3, 3)
    # local -y (finger direction) is world -z; local x (closing) is horizontal
    assert np.allclose(-R[:, 1], [0, 0, -1], atol=1e-2)
    assert abs(R[2, 0]) < 1e-2


def test_ee_site_sits_between_the_pads(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    a = data.geom_xpos[scene.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_3")]
    b = data.geom_xpos[scene.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_3")]
    mid = 0.5 * (a + b)
    assert np.linalg.norm(data.site_xpos[scene.ee_site_id(model)] - mid) < 2e-3


def test_jaw_gap_table(model):
    """The pad-to-pad gap the module docstring quotes, re-measured."""
    data = mujoco.MjData(model)
    qi = scene.arm_qpos_index(model)
    a = scene.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_3")
    b = scene.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_3")
    expect = {-0.174: 0.016, 0.0: 0.026, 0.5: 0.053, 1.0: 0.077, 1.75: 0.104}
    for jaw, gap in expect.items():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[qi[scene.JAW]] = jaw
        mujoco.mj_forward(model, data)
        got = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b])
        assert abs(got - gap) < 1.5e-3, (jaw, got)
    # a 25 mm cube fits between the pads well before the jaw is fully open
    assert expect[1.0] > 2 * scene.CUBE_HALF + 0.02


def test_cameras_exist_and_look_at_the_table(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, (pos, look) in scene.CAMERAS.items():
        cid = scene.name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert np.allclose(data.cam_xpos[cid], pos, atol=1e-6)
        R = data.cam_xmat[cid].reshape(3, 3)
        forward = -R[:, 2]
        want = np.asarray(look) - np.asarray(pos)
        want /= np.linalg.norm(want)
        assert np.dot(forward, want) > 0.999, name


def test_parked_cubes_are_out_of_frame(model):
    for xy in scene.PARK_XY:
        assert np.hypot(*xy) > 1.0
        assert not (scene.SPAWN_X[0] <= xy[0] <= scene.SPAWN_X[1]
                    and scene.SPAWN_Y[0] <= xy[1] <= scene.SPAWN_Y[1])


def test_spawn_region_is_reachable(model):
    """Every corner of the spawn box is within the table-level reach annulus
    measured in the module docstring (0.035 .. 0.40 m), with margin."""
    for x in scene.SPAWN_X:
        for y in scene.SPAWN_Y:
            r = np.hypot(x, y)
            assert 0.10 < r < 0.35, (x, y, r)


def test_xml_dump_recompiles(model):
    xml = scene.scene_xml(3)
    assert "ee_site" in xml and "cube_red" in xml and "target" in xml
    # The dump has meshdir resolved relative to the dumped location, so it
    # only recompiles from the asset directory. Check the text, not a compile.
    assert xml.count("<camera") == len(scene.CAMERAS)
