import numpy as np
import pytest
from rllib_cpu.env import RiceEnv


@pytest.mark.parametrize("negotiation", [False, True])
def test_complete_episode_and_action_masks(negotiation):
    env = RiceEnv({"negotiation_on": negotiation})
    obs, _ = env.reset(seed=42)
    rng = np.random.default_rng(42)
    total = 0
    for t in range(env.env.episode_length):
        actions = {}
        for i, item in obs.items():
            assert env.observation_spaces[i].contains(item)
            segments = np.split(item["action_mask"], np.cumsum(env.action_spaces[i].nvec)[:-1])
            actions[i] = np.array([rng.choice(np.flatnonzero(mask)) for mask in segments])
        obs, rewards, done, truncated, _ = env.step(actions)
        assert all(np.isfinite(v) for v in rewards.values())
        assert not truncated["__all__"]
        total += 1
        if done["__all__"]:
            break
    assert total == (60 if negotiation else 20)
    assert env.env.current_year == 2115
    assert done["__all__"]


def test_masked_policy_never_selects_forbidden_actions():
    import torch
    from rllib_cpu.model import MaskedModel
    env = RiceEnv({"negotiation_on": True})
    obs, _ = env.reset()
    space = env.observation_spaces[0]
    model = MaskedModel(space, env.action_spaces[0], int(sum(env.action_spaces[0].nvec)), {}, "test")
    features = torch.tensor(obs[0]["features"]).unsqueeze(0)
    mask = torch.tensor(obs[0]["action_mask"]).unsqueeze(0)
    mask[0, 0] = 0
    logits, _ = model({"obs": {"features": features, "action_mask": mask}}, [], None)
    assert logits[0, 0].item() < -1e8
    assert torch.isfinite(logits).all()
    assert model.value_function().shape == (1,)


def test_talk_steps_do_not_advance_economic_year():
    env = RiceEnv({"negotiation_on": True})
    obs, _ = env.reset()
    year = env.env.current_year
    for step in range(3):
        actions = {i: np.zeros_like(env.action_spaces[i].nvec) for i in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        if step < 2:
            assert env.env.current_year == year
            assert all(value == 0 for value in rewards.values())
        else:
            assert env.env.current_year == year + 5
