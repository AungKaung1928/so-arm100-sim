"""Demonstrations in the layout `lerobot/smolvla_base` was pretrained on.

`datasets.py` stores the simulator's own units (radians, 15-d proprio, sim
camera names). SmolVLA-base was pretrained on community SO-100/101 data in
this layout:

    observation.state            float32 [6]  5 body joints in degrees + jaw 0..100
    action                       float32 [6]  absolute joint targets, same units
    observation.images.camera1   top view     (pretraining order: top, wrist, side)
    observation.images.camera2   front view   (the sim has no wrist camera)
    observation.images.camera3   side view

Units follow LeRobot's SO-100 follower: body joints `MotorNormMode.DEGREES`,
gripper `RANGE_0_100` (lerobot robots/so_follower). The conversion fixes scale
only: the sim's zero pose is not a real arm's calibration zero, so absolute
offsets still differ from any real SO-100 dataset. At the sim home pose the
Pitch joint is -90 deg, 4.0 std below the `so100` action mean in the
smolvla_base normalizer stats (119.94 +- 52.41 deg): a sign or zero
convention gap that only a real arm's calibration would settle.

What the units change, and what they do not: a fine-tune with `lerobot-train`
replaces the normalizer stats with this dataset's own (MEAN_STD), and a
per-joint linear map such as rad -> deg cancels out of a z-score. So for
fine-tuning, only the camera names and order matter; the units matter when
running the base checkpoint with its own stats (zero-shot).

fps stays 20 (the sim's control rate): SmolVLA indexes action chunks by step,
not by seconds, so a 50-step chunk covers 2.5 s here.
"""
import numpy as np

from .env import ACT_DIM
from .scene import JAW, JOINT_NAMES

JAW_RANGE = (-0.174, 1.75)  # rad, Jaw joint range in assets/trs_so_arm100/so_arm100.xml
CAMERA_MAP = {"camera1": "top", "camera2": "front", "camera3": "side"}
NAMES = [f"{j}.pos" for j in JOINT_NAMES]


def to_so100_units(q_rad):
    """[..., 6] radians -> [..., 6] (degrees for body joints, 0..100 for the jaw)."""
    q = np.asarray(q_rad, np.float64)
    out = np.degrees(q)
    lo, hi = JAW_RANGE
    out[..., JAW] = (q[..., JAW] - lo) / (hi - lo) * 100.0
    return out.astype(np.float32)


def from_so100_units(u):
    """Inverse of `to_so100_units`: policy actions back to sim joint targets."""
    u = np.asarray(u, np.float64)
    q = np.radians(u)
    lo, hi = JAW_RANGE
    q[..., JAW] = lo + u[..., JAW] / 100.0 * (hi - lo)
    return q


def features(height=256, width=256):
    f = {
        "observation.state": {"dtype": "float32", "shape": (ACT_DIM,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (ACT_DIM,), "names": NAMES},
    }
    for key in CAMERA_MAP:
        f[f"observation.images.{key}"] = {"dtype": "image", "shape": (height, width, 3),
                                          "names": ["height", "width", "channels"]}
    return f


def frame(state, action, task, images):
    """One recorder frame. state: sim proprio (first 6 = joint angles, rad);
    action: 6 joint targets (rad); images: {sim camera name: HxWx3 uint8}."""
    row = {
        "observation.state": to_so100_units(np.asarray(state)[:ACT_DIM]),
        "action": to_so100_units(action),
        "task": str(task),
    }
    for key, cam in CAMERA_MAP.items():
        row[f"observation.images.{key}"] = np.asarray(images[cam], np.uint8)
    return row


class SmolVLARecorder:
    """Same contract as `datasets.LeRobotRecorder`, SmolVLA-base layout."""

    def __init__(self, root, repo_id, fps=20, height=256, width=256):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.ds = LeRobotDataset.create(
            repo_id=repo_id, fps=int(fps), root=str(root), robot_type="so100_follower",
            features=features(height, width), use_videos=False)
        self.n_frames = 0
        self.n_episodes = 0

    def add_episode(self, frames):
        """frames: list of dict(state, action, task, images={cam: HxWx3 uint8})."""
        for fr in frames:
            self.ds.add_frame(frame(fr["state"], fr["action"], fr["task"], fr["images"]))
            self.n_frames += 1
        self.ds.save_episode()
        self.n_episodes += 1

    def finalize(self):
        self.ds.finalize()
        return self.ds
