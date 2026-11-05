"""Regression test: eval/success_mean is the per-EPISODE success rate (the
`is_success` reported at each episode's final step), not an average of
`is_success` over every timestep."""

from __future__ import annotations

import types

import torch
from test_envlayer_helpers import SuccessAtEndEnv

from srl.cli.train import _evaluate_agent
from srl.envs.gymnasium_wrapper import GymnasiumWrapper


class _FakeAgent:
    device = "cpu"

    def __init__(self) -> None:
        self.model = types.SimpleNamespace(encoders={"state": None}, encoder_input_names=None)

    def predict(self, obs, deterministic=False):
        return torch.zeros(1, 1), None, None, None


def test_success_mean_uses_final_step_of_each_episode() -> None:
    env = GymnasiumWrapper(SuccessAtEndEnv(ep_len=4))
    metrics = _evaluate_agent(_FakeAgent(), env, env_type="flat", seed=0, episodes=3, render=False)
    # Every episode ends in success (is_success=1 on its last step only):
    # the per-step average this used to report would be 0.25.
    assert metrics["eval/success_mean"] == 1.0
    assert metrics["eval/episodes"] == 3.0


def test_success_mean_absent_without_success_info() -> None:
    from test_envlayer_helpers import CounterEnv

    metrics = _evaluate_agent(
        _FakeAgent(),
        GymnasiumWrapper(CounterEnv()),
        env_type="flat",
        seed=0,
        episodes=1,
        render=False,
    )
    assert "eval/success_mean" not in metrics
