"""Writing demonstrations as a LeRobotDataset (format v3).

Optional: `lerobot` is an extra, not a core dependency, and everything else
in this package works without it. The writer stores

    observation.state           float32 [15]   the proprio vector
    observation.images.<cam>    uint8 image    one entry per requested camera
    action                      float32 [6]    absolute joint targets, radians
    task                        str            the language instruction

with images as PNG-in-parquet (`use_videos=False`) rather than encoded video:
frames here are 96x128 and a few hundred episodes, so the video path's
ffmpeg dependency and its compression artefacts buy nothing.

`finalize()` is mandatory in v3 -- without it the parquet shards are left
unconsolidated and the dataset does not load.
"""
import numpy as np

from . import scene
from .env import PROPRIO_DIM, ACT_DIM

STATE_NAMES = ([f"{j}.pos" for j in scene.JOINT_NAMES]
               + [f"{j}.vel" for j in scene.JOINT_NAMES] + ["ee.x", "ee.y", "ee.z"])
ACTION_NAMES = [f"{j}.pos" for j in scene.JOINT_NAMES]


def lerobot_available():
    try:
        import lerobot  # noqa: F401
        return True
    except ImportError:
        return False


def features(cameras=("front",), height=96, width=128):
    f = {
        "observation.state": {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (ACT_DIM,), "names": ACTION_NAMES},
    }
    for cam in cameras:
        f[f"observation.images.{cam}"] = {"dtype": "image", "shape": (height, width, 3),
                                          "names": ["height", "width", "channels"]}
    return f


class LeRobotRecorder:
    def __init__(self, root, repo_id, fps=20, cameras=("front",), height=96, width=128):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.cameras = tuple(cameras)
        self.ds = LeRobotDataset.create(
            repo_id=repo_id, fps=int(fps), root=str(root),
            robot_type="so_arm100_sim", features=features(cameras, height, width),
            use_videos=False)
        self.n_frames = 0
        self.n_episodes = 0

    def add_episode(self, frames):
        """frames: list of dict(state, action, task, images={cam: HxWx3 uint8})."""
        for fr in frames:
            row = {
                "observation.state": np.asarray(fr["state"], np.float32),
                "action": np.asarray(fr["action"], np.float32),
                "task": str(fr["task"]),
            }
            for cam in self.cameras:
                row[f"observation.images.{cam}"] = np.asarray(fr["images"][cam], np.uint8)
            self.ds.add_frame(row)
            self.n_frames += 1
        self.ds.save_episode()
        self.n_episodes += 1

    def finalize(self):
        self.ds.finalize()
        return self.ds


def load(root, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, root=str(root))
