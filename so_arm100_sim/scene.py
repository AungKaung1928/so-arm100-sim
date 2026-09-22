"""The tabletop scene, built from the vendored Menagerie arm with MjSpec.

The arm description under `assets/trs_so_arm100/` is the MuJoCo Menagerie
model of TheRobotStudio's Standard Open Arm 100, Apache-2.0, pinned to the
commit in `VENDORED_COMMIT` and left byte-for-byte untouched -- `test_scene.py`
hashes it. Everything this project needs on top of it (an end-effector site,
a floor that acts as the table, three coloured cubes, a target marker, three
cameras, a light) is attached here in Python through `mujoco.MjSpec`, so that
the upstream file never has to be edited and an upstream update is a file copy.

Frames, measured from the vendored model at its `home` keyframe rather than
assumed (`tools/probe_model.py` prints them):

  - The arm base sits at the world origin. The gripper reaches out along -y,
    so "in front of the robot" is negative y. At `home` the pinch point is at
    (0, -0.239, 0.099) m.
  - `ee_site` is placed on `Fixed_Jaw` at the midpoint between the third pad
    of each jaw, (0, -0.078, 0) in that body's frame. Its local -y axis is
    the finger direction (world -z at `home`, i.e. pointing down) and its
    local x axis is the direction the jaws close along.
  - Pad-to-pad gap versus the `Jaw` joint angle: 16 mm at -0.174 rad (limit),
    26 mm at 0, 53 mm at 0.5, 104 mm at 1.75 (limit). A 25 mm cube is held
    at roughly Jaw = 0.0 and the actuator has 3.5 N of force range to
    squeeze with.
  - Table-level reach: 20k random joint configurations put the pinch point
    within 3 cm of the table at radii between 0.035 m (5th pct) and 0.40 m
    (95th pct), almost all of it at negative y. That is reach with any
    orientation; the top-down grasp pose the tasks need is far more limited
    and is what actually bounds `SPAWN_X` / `SPAWN_Y` (see there).
"""
import hashlib
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "assets", "trs_so_arm100")
ARM_XML = os.path.join(ASSET_DIR, "so_arm100.xml")

# google-deepmind/mujoco_menagerie, trs_so_arm100 v1.3, 2025-06-09 joint limits.
VENDORED_COMMIT = "822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7"
# sha256 of so_arm100.xml as vendored. A failing hash means the upstream file
# was edited by hand, which this project promises never to do.
VENDORED_SHA256 = "1901fe7e86d2458e2011390bf52014c1f6f4fcea79abef1788b3108ea9781955"

JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
ARM_JOINTS = JOINT_NAMES[:5]          # the 5 that move the end effector
N_JOINTS = 6
JAW = 5                               # index of the gripper joint / actuator

EE_SITE = "ee_site"
EE_SITE_POS = (0.0, -0.078, 0.0)      # in the Fixed_Jaw frame, see module doc

JAW_OPEN = 1.0                        # rad; 77 mm pad gap
JAW_CLOSED = -0.10                    # rad; commands past a 25 mm cube so the PD squeezes

CUBE_COLORS = ("red", "green", "blue")
CUBE_RGBA = {
    "red":   (0.85, 0.12, 0.10, 1.0),
    "green": (0.10, 0.65, 0.20, 1.0),
    "blue":  (0.12, 0.25, 0.85, 1.0),
}
CUBE_HALF = 0.0125                    # m, a 25 mm cube
CUBE_MASS = 0.020                     # kg
# Where cubes that are not part of the current episode wait: on the floor,
# well outside every camera frustum and the arm's reach.
PARK_XY = ((1.0, 1.0), (2.0, 1.0), (3.0, 1.0))

TARGET_SITE = "target"
TARGET_RGBA = (0.95, 0.80, 0.10, 0.9)
TARGET_RADIUS = 0.02

# Object / target spawn region on the table, world frame. Bounded by where a
# TOP-DOWN grasp pose is reachable, not by where the arm can reach at all:
# `tools/reach_map.py` shows the pointing-down pose is solvable to 3 mm at
# z = 0.09 m only for y >= -0.26 and |x| <= 0.12, and the lift task has to
# hold the cube there. The box below keeps 2 cm of margin to that boundary.
SPAWN_X = (-0.10, 0.10)
SPAWN_Y = (-0.26, -0.16)

PHYSICS_DT = 0.002                    # the model's own default, stated

# name -> (position, look-at point). Chosen so the whole spawn region and the
# arm are in every frame at 45 deg fovy.
# Framed on the spawn box, not on the arm: at 96x128 a 25 mm cube seen from
# 0.85 m was four pixels wide, which is too little for an image policy.
CAMERAS = {
    "front": ((0.0, -0.62, 0.34), (0.0, -0.21, 0.03)),
    "top":   ((0.0, -0.21, 0.55), (0.0, -0.2101, 0.0)),
    "side":  ((0.50, -0.21, 0.26), (0.0, -0.21, 0.03)),
}
CAMERA_FOVY = 45.0


def vendored_sha256():
    with open(ARM_XML, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def lookat_quat(pos, look, up=(0.0, 0.0, 1.0)):
    """Quaternion (w,x,y,z) of a camera at `pos` looking at `look`.

    MuJoCo cameras look along their local -z with local +y up, so the frame is
    built from z = -(look - pos), x = up x z, y = z x x.
    """
    pos, look = np.asarray(pos, float), np.asarray(look, float)
    f = look - pos
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def build_spec(n_cubes=3):
    """The scene as an MjSpec. Compile with `.compile()`; dump with `.to_xml()`."""
    if not 1 <= n_cubes <= len(CUBE_COLORS):
        raise ValueError(f"n_cubes must be 1..{len(CUBE_COLORS)}, got {n_cubes}")
    spec = mujoco.MjSpec.from_file(ARM_XML)
    spec.modelname = "so_arm100_tabletop"
    spec.option.timestep = PHYSICS_DT

    # End-effector site on the fixed jaw. Small sphere, visual group 4 so it
    # never shows in a rendered camera image.
    site = spec.body("Fixed_Jaw").add_site(name=EE_SITE, pos=list(EE_SITE_POS),
                                           size=[0.003, 0.003, 0.003],
                                           rgba=[1.0, 0.0, 0.0, 0.5])
    site.group = 4

    w = spec.worldbody
    # Lighting is deliberately modest: MuJoCo's default headlight plus a
    # bright directional light saturated the floor to pure white in the top
    # camera (pixel mean 251/255), which throws away every shading cue an
    # image policy could use.
    spec.visual.headlight.ambient = [0.15, 0.15, 0.15]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    w.add_light(name="sun", pos=[0.3, -0.5, 1.2], dir=[-0.25, 0.25, -1.0],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                diffuse=[0.55, 0.55, 0.55], ambient=[0.1, 0.1, 0.1],
                specular=[0.1, 0.1, 0.1], castshadow=True)
    # The floor is the table. Plane, so nothing can fall off the edge; the
    # spawn region is what bounds the task.
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[0.0, 0.0, 0.05], rgba=[0.48, 0.46, 0.42, 1.0],
               friction=[1.0, 0.005, 0.0001])

    for i in range(n_cubes):
        color = CUBE_COLORS[i]
        b = w.add_body(name=f"cube_{color}",
                       pos=[PARK_XY[i][0], PARK_XY[i][1], CUBE_HALF])
        b.add_freejoint(name=f"cube_{color}_free")
        b.add_geom(name=f"cube_{color}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[CUBE_HALF] * 3, rgba=list(CUBE_RGBA[color]),
                   mass=CUBE_MASS, friction=[1.0, 0.005, 0.0001])

    # Target marker: a flat disc, visual only (no contype), on the table.
    t = w.add_site(name=TARGET_SITE, pos=[PARK_XY[0][0], PARK_XY[0][1] + 0.5, 0.0005],
                   type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[TARGET_RADIUS, TARGET_RADIUS, 0.0005],
                   rgba=list(TARGET_RGBA))
    t.group = 0

    for name, (pos, look) in CAMERAS.items():
        c = w.add_camera(name=name, pos=list(pos))
        c.quat = lookat_quat(pos, look)
        c.fovy = CAMERA_FOVY
    return spec


def build_model(n_cubes=3):
    return build_spec(n_cubes).compile()


def scene_xml(n_cubes=3):
    return build_spec(n_cubes).to_xml()


# -- id helpers, resolved by name so a reordering upstream fails loudly ------

def name2id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise KeyError(f"no {kind} named {name!r}")
    return i


def joint_ids(model):
    return np.array([name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])


def arm_qpos_index(model):
    return np.array([model.jnt_qposadr[j] for j in joint_ids(model)])


def arm_qvel_index(model):
    return np.array([model.jnt_dofadr[j] for j in joint_ids(model)])


def joint_limits(model):
    j = joint_ids(model)
    return model.jnt_range[j, 0].copy(), model.jnt_range[j, 1].copy()


def cube_qpos_index(model, color):
    """qpos address of a cube's free joint: 7 numbers, xyz + wxyz quaternion."""
    j = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{color}_free")
    return int(model.jnt_qposadr[j])


def cube_qvel_index(model, color):
    j = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{color}_free")
    return int(model.jnt_dofadr[j])


def cube_body(model, color):
    return name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{color}")


def cube_geom(model, color):
    return name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"cube_{color}_geom")


def pad_geoms(model):
    """(fixed_jaw_pads, moving_jaw_pads) geom ids. Both jaws carry four."""
    fixed = [name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"fixed_jaw_pad_{i}") for i in range(1, 5)]
    moving = [name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"moving_jaw_pad_{i}") for i in range(1, 5)]
    return np.array(fixed), np.array(moving)


def ee_site_id(model):
    return name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)


def target_site_id(model):
    return name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE)


def home_qpos(model):
    """The 6 joint angles of the vendored `home` keyframe."""
    k = name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_qpos[k][arm_qpos_index(model)].copy()
