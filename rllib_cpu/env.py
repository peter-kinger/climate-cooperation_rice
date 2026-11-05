"""Gymnasium adapter; original Rice dynamics are unchanged."""
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from rice import Rice


class RiceEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.env = Rice(**dict(config or {}))
        self.agents = list(range(self.env.num_regions))
        self.possible_agents = self.agents.copy()
        self._agent_ids = set(self.agents)
        sample = self.env.reset()[0]
        self.observation_spaces = {
            i: gym.spaces.Dict({
                "features": gym.spaces.Box(-np.inf, np.inf, sample["features"].shape, np.float32),
                "action_mask": gym.spaces.Box(0, 1, sample["action_mask"].shape, np.float32),
            }) for i in self.agents
        }
        self.action_spaces = self.env.action_space
        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)

    @staticmethod
    def convert(obs):
        return {i: {k: np.asarray(v, dtype=np.float32) for k, v in o.items()} for i, o in obs.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.convert(self.env.reset()), {}

    def step(self, actions):
        obs, rewards, done, _ = self.env.step(actions)
        terminal = bool(done["__all__"])
        return (self.convert(obs), {i: float(r) for i, r in rewards.items()},
                {**{i: terminal for i in self.agents}, "__all__": terminal},
                {**{i: False for i in self.agents}, "__all__": False}, {})
