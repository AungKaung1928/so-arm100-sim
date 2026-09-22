"""gymnasium.Env wrapper, for tools that expect one. Optional dependency."""
import numpy as np

from . import env as E
from . import scene


def make_gym_env(**kwargs):
    import gymnasium as gym
    from gymnasium import spaces

    class SoArm100Gym(gym.Env):
        metadata = {"render_modes": ["rgb_array"], "render_fps": E.CONTROL_HZ}

        def __init__(self, **kw):
            super().__init__()
            self.env = E.ArmEnv(**kw)
            d = self.env.state_dim
            obs = {"state": spaces.Box(-np.inf, np.inf, (d,), np.float32)}
            if self.env.image:
                c = self.env.image_cfg
                obs["image"] = spaces.Box(0, 255, (c["height"], c["width"], 3), np.uint8)
            self.observation_space = spaces.Dict(obs)
            if self.env.action_mode == "delta":
                self.action_space = spaces.Box(-1.0, 1.0, (E.ACT_DIM,), np.float32)
            else:
                lo, hi = scene.joint_limits(self.env.model)
                self.action_space = spaces.Box(lo.astype(np.float32), hi.astype(np.float32))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.env.reset(seed=seed), {}

        def step(self, action):
            obs, r, done, info = self.env.step(action)
            return obs, r, info["terminated"], info["truncated"], info

        def render(self):
            return self.env.render(**self.env.image_cfg)

        def close(self):
            self.env.close()

    return SoArm100Gym(**kwargs)
