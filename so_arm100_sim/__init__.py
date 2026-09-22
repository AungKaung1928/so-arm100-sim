"""so_arm100_sim -- MuJoCo tasks, scripted expert and evaluation protocol for
the Standard Open Arm 100.

Consumed by the RL, imitation-learning and language-conditioned policy
repositories. The environment surface is deliberately small and framework
free; a gymnasium adapter is in `gym_adapter` for tools that want one.
"""
from .scene import CUBE_COLORS, JOINT_NAMES, N_JOINTS        # noqa: F401
from .env import ArmEnv, TaskSpec, TASKS, OBS_MODES, ACTION_MODES   # noqa: F401
from .dr import DRConfig, EVAL_PHYSICS                        # noqa: F401
from .expert import ScriptedExpert                            # noqa: F401
from .evaluate import evaluate as evaluate_policy   # noqa: F401

__version__ = "0.1.0"
