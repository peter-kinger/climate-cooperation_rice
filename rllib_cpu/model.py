"""Shared actor/critic with the simulator's mitigation action mask."""
import numpy as np
import torch
from torch import nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class MaskedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        space = getattr(obs_space, "original_space", obs_space)
        size = int(np.prod(space["features"].shape))
        self.body = nn.Sequential(nn.Linear(size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, num_outputs)
        self.critic = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        hidden = self.body(obs["features"].float())
        self._value = self.critic(hidden).squeeze(-1)
        mask = obs["action_mask"].float()
        return self.actor(hidden).masked_fill(mask < 0.5, -1e9), state

    def value_function(self):
        return self._value.reshape(-1)


ModelCatalog.register_custom_model("rice_masked", MaskedModel)
