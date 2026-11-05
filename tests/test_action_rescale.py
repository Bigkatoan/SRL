"""Regression tests: policy/buffer act in [-1, 1]; envs see their real bounds."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from test_envlayer_helpers import (
    CounterEnv,
    mlp_config,
    record_buffer_adds,
    register_test_envs,
    run_cli,
)

from srl.envs.action_rescale import RescaleActionWrapper, maybe_rescale_action
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.training.envs import make_env as _make_cli_env
from srl.utils.spaces import sample_action_space, sample_vector_actions


def test_rescale_maps_unit_interval_onto_env_bounds() -> None:
    inner = CounterEnv(low=-3.0, high=5.0)
    env = RescaleActionWrapper(GymnasiumWrapper(inner))
    assert np.all(env.act_space.low == -1.0) and np.all(env.act_space.high == 1.0)
    env.reset(seed=0)
    for a, expected in [(-1.0, -3.0), (1.0, 5.0), (0.0, 1.0), (2.0, 5.0), (-7.0, -3.0)]:
        env.step(np.array([a], dtype=np.float32))
        assert inner.last_action[0] == expected
    np.testing.assert_allclose(env.unscale_action(np.array([5.0])), [1.0])
    # SRL attributes forwarded through the wrapper.
    assert env.num_envs == 1 and env.obs_key == "state"
    assert env.obs_space.shape == (2,)


def test_infinite_bounds_pass_through_and_noop_cases() -> None:
    base = GymnasiumWrapper(CounterEnv())
    assert maybe_rescale_action(base) is base  # already [-1, 1]

    class _Mixed(CounterEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(
                np.array([-np.inf, 0.0], dtype=np.float32),
                np.array([np.inf, 10.0], dtype=np.float32),
            )

    inner = _Mixed()
    env = maybe_rescale_action(GymnasiumWrapper(inner))
    assert isinstance(env, RescaleActionWrapper)
    assert env.act_space.low[0] == -np.inf and env.act_space.high[1] == 1.0
    env.reset()
    env.step(np.array([7.5, 0.0], dtype=np.float32))
    np.testing.assert_allclose(inner.last_action, [7.5, 5.0])

    unbounded = GymnasiumWrapper(gym.make("CartPole-v1"))
    assert maybe_rescale_action(unbounded) is unbounded  # discrete


def test_cli_env_pendulum_gets_full_torque_range() -> None:
    env = _make_cli_env("Pendulum-v1", "cpu", 1)
    assert np.all(env.act_space.high == 1.0)
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))
    assert env.unwrapped.last_u == 2.0
    env.close()


def test_warmup_samples_are_uniform_in_unit_box_for_unbounded_space() -> None:
    space = gym.spaces.Box(-np.inf, np.inf, shape=(1000,), dtype=np.float32)
    space.seed(0)
    a = sample_action_space(space)
    assert a.dtype == np.float32 and a.min() >= -1.0 and a.max() <= 1.0
    assert a.std() > 0.4  # uniform(-1, 1) has std ~0.577, not a point mass


def test_sample_vector_actions_shape() -> None:
    class _Vec:
        num_envs = 3
        act_space = gym.spaces.Box(-1, 1, shape=(2,))
        single_act_space = act_space

    assert sample_vector_actions(_Vec()).shape == (3, 2)


def test_off_policy_warmup_and_buffer_live_in_normalised_domain(monkeypatch, tmp_path) -> None:
    register_test_envs()
    adds: list[dict] = []
    cfg = mlp_config("sac", total_steps=40, start_steps=40, update_after=1000)
    run_cli(
        monkeypatch,
        tmp_path,
        cfg,
        env_id="SRLCounterWide-v0",
        on_agent=record_buffer_adds(adds),
    )
    actions = np.array([np.asarray(k["action"]).reshape(-1)[0] for k in adds])
    assert len(actions) == 40
    # Warmup uses the env's [-3, 5] space before this fix -> out of [-1, 1].
    assert actions.min() >= -1.0 and actions.max() <= 1.0
