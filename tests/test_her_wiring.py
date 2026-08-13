"""Tests for the HER (Hindsight Experience Replay) YAML/CLI wiring.

`HERReplayBuffer`'s own storage/relabelling unit tests live in
test_buffers.py. What is covered here is everything that connects it to a real
run: that `use_her: true` actually swaps the buffer SAC builds, that the
sampled batch has the dict shape SAC's `update()` requires, that goals really
get relabelled, and that the CLI's off-policy collection loop routes whole
episodes through `add_transition()` rather than `ReplayBuffer.add()`.
"""

from __future__ import annotations

import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from srl.core.config import SACConfig
from srl.core.her_replay_buffer import HERReplayBuffer
from srl.core.replay_buffer import ReplayBuffer
from srl.registry.builder import ModelBuilder

OBSERVATION_DIM = 4
GOAL_DIM = 2
ACTION_DIM = 2
# GoalEnvWrapper flattens to [observation | achieved_goal | desired_goal] ...
FLAT_OBS_DIM = OBSERVATION_DIM + GOAL_DIM + GOAL_DIM
# ... and HER stores [observation | achieved_goal], re-appending the (possibly
# relabelled) desired goal at sample time to reproduce that same layout.
HER_OBS_DIM = OBSERVATION_DIM + GOAL_DIM


def _sparse_reward_fn(achieved, desired, info):
    """FetchReach-style sparse reward: 0 when the goal is reached, else -1."""
    return 0.0 if float(np.linalg.norm(np.asarray(achieved) - np.asarray(desired))) < 0.05 else -1.0


class _FakeGoalEnv(gym.Env):
    """Minimal gymnasium-robotics-shaped GoalEnv (no MuJoCo needed).

    Truncates at `episode_len` and never terminates -- the same shape as every
    Fetch task, which is what makes episode-commit-on-truncation load-bearing.
    """

    def __init__(self, episode_len: int = 5) -> None:
        self.episode_len = episode_len
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32),
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (GOAL_DIM,), np.float32),
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, (GOAL_DIM,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self._t = 0
        self._episode = 0

    def compute_reward(self, achieved_goal, desired_goal, info):
        return _sparse_reward_fn(achieved_goal, desired_goal, info)

    def _obs(self):
        # Achieved goal walks with t, desired goal is fixed per episode and
        # deliberately far away -- so a relabelled goal is always distinguishable
        # from the original one.
        return {
            "observation": np.full(OBSERVATION_DIM, self._t, dtype=np.float32),
            "achieved_goal": np.full(GOAL_DIM, self._t, dtype=np.float32),
            "desired_goal": np.full(GOAL_DIM, 100.0 + self._episode, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._episode += 1
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        obs = self._obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        truncated = self._t >= self.episode_len
        return obs, reward, False, truncated, {}


def _make_goal_env(episode_len: int = 5):
    from srl.envs.goal_env_wrapper import GoalEnvWrapper

    return GoalEnvWrapper(_FakeGoalEnv(episode_len=episode_len))


def _build_sac(cfg: SACConfig):
    import copy

    from srl.algorithms.sac import SAC

    model_dict = {
        "encoders": [
            {
                "name": "actor_state_enc",
                "type": "mlp",
                "input_dim": FLAT_OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
            },
            {
                "name": "critic_state_enc",
                "type": "mlp",
                "input_dim": FLAT_OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
            },
        ],
        "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
        "actor": {
            "name": "actor",
            "type": "squashed_gaussian",
            "action_dim": ACTION_DIM,
            "log_std_min": -5.0,
            "log_std_max": 2.0,
            "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
        },
        "critic": {
            "name": "critic",
            "type": "twin_q",
            "action_dim": ACTION_DIM,
            "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
        },
    }
    model = ModelBuilder.from_dict(model_dict)
    return SAC(model, copy.deepcopy(model), config=cfg, device="cpu")


def _her_cfg(**overrides) -> SACConfig:
    kwargs = dict(
        action_dim=ACTION_DIM,
        batch_size=8,
        use_her=True,
        her_obs_dim=HER_OBS_DIM,
        her_goal_dim=GOAL_DIM,
        her_reward_fn=_sparse_reward_fn,
        her_max_episode_len=5,
        buffer_size=1000,
        learning_starts=0,
    )
    kwargs.update(overrides)
    return SACConfig(**kwargs)


def _fill_episodes(buf: HERReplayBuffer, n_episodes: int, episode_len: int = 5) -> None:
    for ep in range(n_episodes):
        for t in range(episode_len):
            buf.add_transition(
                obs=np.full(HER_OBS_DIM, t, dtype=np.float32),
                achieved_goal=np.full(GOAL_DIM, t, dtype=np.float32),
                desired_goal=np.full(GOAL_DIM, 100.0 + ep, dtype=np.float32),
                action=np.zeros(ACTION_DIM, dtype=np.float32),
                next_obs=np.full(HER_OBS_DIM, t + 1, dtype=np.float32),
                next_achieved_goal=np.full(GOAL_DIM, t + 1, dtype=np.float32),
                done=False,
                truncated=(t == episode_len - 1),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Buffer selection
# ──────────────────────────────────────────────────────────────────────────────


def test_sac_defaults_to_plain_replay_buffer() -> None:
    agent = _build_sac(SACConfig(action_dim=ACTION_DIM, batch_size=8))
    assert isinstance(agent.buffer, ReplayBuffer)
    assert not isinstance(agent.buffer, HERReplayBuffer)


def test_sac_use_her_builds_her_replay_buffer() -> None:
    agent = _build_sac(_her_cfg(her_ratio=0.7, her_strategy="final"))
    assert isinstance(agent.buffer, HERReplayBuffer)
    assert agent.buffer.her_ratio == 0.7
    assert agent.buffer.strategy == "final"
    assert agent.buffer.max_episode_len == 5
    assert agent.buffer.obs_dim == HER_OBS_DIM
    assert agent.buffer.goal_dim == GOAL_DIM


def test_sac_use_her_without_env_derived_fields_raises() -> None:
    """Silently falling back to a plain buffer is the exact bug this fixes."""
    with pytest.raises(ValueError, match="her_obs_dim"):
        _build_sac(SACConfig(action_dim=ACTION_DIM, use_her=True))


@pytest.mark.parametrize("strategy", ["future", "final", "episode", "random"])
def test_all_documented_strategies_are_accepted_and_relabel(strategy: str) -> None:
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        strategy=strategy,
        her_ratio=1.0,
        max_episode_len=5,
    )
    _fill_episodes(buf, n_episodes=3)
    batch = buf.sample(32)
    sampled_dg = batch.obs["state"][:, HER_OBS_DIM:].numpy()
    # Stored desired goals are all >= 100; every achieved goal is < 100, so a
    # relabelled goal is unambiguously distinguishable from the original.
    assert (sampled_dg < 100.0).all(), f"{strategy}: goals were not relabelled"


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown HER strategy"):
        HERReplayBuffer(
            capacity=10,
            obs_dim=HER_OBS_DIM,
            goal_dim=GOAL_DIM,
            action_dim=ACTION_DIM,
            reward_fn=_sparse_reward_fn,
            strategy="nearest",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Batch shape / relabelling
# ──────────────────────────────────────────────────────────────────────────────


def test_sample_returns_dict_observations_matching_model_input() -> None:
    """Regression test: sample() used to return a bare concatenated tensor,
    which SAC.update() cannot consume (it iterates `batch.obs.items()`)."""
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        max_episode_len=5,
    )
    _fill_episodes(buf, n_episodes=3)
    batch = buf.sample(16)

    assert isinstance(batch.obs, dict)
    assert isinstance(batch.next_obs, dict)
    assert set(batch.obs) == {"state"}
    assert batch.obs["state"].shape == (16, FLAT_OBS_DIM)
    assert batch.next_obs["state"].shape == (16, FLAT_OBS_DIM)
    assert torch.isfinite(batch.obs["state"]).all()


def test_relabelled_and_non_relabelled_goals_differ_within_a_batch() -> None:
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        her_ratio=0.5,
        max_episode_len=5,
    )
    _fill_episodes(buf, n_episodes=4)

    np.random.seed(0)
    batch = buf.sample(400)
    sampled_dg = batch.obs["state"][:, HER_OBS_DIM:].numpy()
    relabelled = (sampled_dg < 100.0).all(axis=-1)

    assert relabelled.any(), "no transition was relabelled at her_ratio=0.5"
    assert (~relabelled).any(), "every transition was relabelled at her_ratio=0.5"
    # ~50% relabelled; generous bounds so this can't flake.
    assert 0.3 < relabelled.mean() < 0.7


def test_her_ratio_zero_never_relabels() -> None:
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        her_ratio=0.0,
        max_episode_len=5,
    )
    _fill_episodes(buf, n_episodes=4)
    batch = buf.sample(100)
    sampled_dg = batch.obs["state"][:, HER_OBS_DIM:].numpy()
    assert (sampled_dg >= 100.0).all()


def test_episode_commits_on_truncation_without_fabricating_done() -> None:
    """Time-limited goal envs never set `terminated`, so the episode must be
    committed on truncation -- but the stored per-timestep `done` must stay 0,
    or the bootstrap target for the non-relabelled fraction gets biased."""
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        her_ratio=0.0,
        max_episode_len=50,  # deliberately larger than the episode
    )
    _fill_episodes(buf, n_episodes=2, episode_len=5)

    assert len(buf) == 2, "episodes did not commit on truncation"
    assert buf.num_transitions == 10
    batch = buf.sample(100)
    assert float(batch.dones.sum()) == 0.0, "truncation was stored as a real terminal"


def test_can_sample_uses_transitions_not_episodes() -> None:
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        max_episode_len=5,
    )
    assert not buf.can_sample(8)
    _fill_episodes(buf, n_episodes=2)  # 2 episodes, 10 transitions
    assert len(buf) == 2
    assert buf.can_sample(8)


def test_state_dict_round_trip() -> None:
    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        max_episode_len=5,
    )
    _fill_episodes(buf, n_episodes=3)

    restored = HERReplayBuffer(
        capacity=100,
        obs_dim=HER_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        reward_fn=_sparse_reward_fn,
        max_episode_len=5,
    )
    restored.load_state_dict(buf.state_dict())
    assert len(restored) == 3
    assert restored.num_transitions == buf.num_transitions
    assert restored.sample(8).obs["state"].shape == (8, FLAT_OBS_DIM)


def test_sac_checkpoint_round_trip_with_her() -> None:
    """SAC.checkpoint_payload() calls buffer.state_dict(); HER must support it."""
    agent = _build_sac(_her_cfg())
    _fill_episodes(agent.buffer, n_episodes=3)
    payload = agent.checkpoint_payload()

    restored = _build_sac(_her_cfg())
    restored.load_checkpoint_payload(payload)
    assert len(restored.buffer) == 3
    assert restored.buffer.num_transitions == agent.buffer.num_transitions


# ──────────────────────────────────────────────────────────────────────────────
# CLI wiring
# ──────────────────────────────────────────────────────────────────────────────


def test_configure_her_from_env_fills_dims_and_reward_fn() -> None:
    from srl.cli.train import _configure_her_from_env

    cfg = SACConfig(action_dim=ACTION_DIM, use_her=True)
    env = _make_goal_env()
    _configure_her_from_env(cfg, env, "goal")

    assert cfg.her_obs_dim == HER_OBS_DIM
    assert cfg.her_goal_dim == GOAL_DIM
    assert cfg.her_reward_fn is not None
    assert cfg.her_reward_fn(np.zeros(GOAL_DIM), np.zeros(GOAL_DIM), {}) == 0.0


def test_configure_her_from_env_is_noop_when_disabled() -> None:
    from srl.cli.train import _configure_her_from_env

    cfg = SACConfig(action_dim=ACTION_DIM)
    _configure_her_from_env(cfg, _make_goal_env(), "goal")
    assert cfg.her_obs_dim == 0
    assert cfg.her_reward_fn is None


def test_configure_her_from_env_rejects_non_goal_env() -> None:
    from srl.cli.train import _configure_her_from_env

    cfg = SACConfig(action_dim=ACTION_DIM, use_her=True)
    with pytest.raises(SystemExit, match="env_type"):
        _configure_her_from_env(cfg, _make_goal_env(), "flat")


def test_her_goal_obs_parts_matches_wrapper_flat_layout() -> None:
    """The stored obs + the appended goal must reproduce the wrapper's flat
    observation exactly, or the sampled batch won't match the encoder."""
    from srl.cli.train import _her_goal_obs_parts

    env = _make_goal_env()
    flat_obs, info = env.reset(seed=0)
    obs_vec, ag, dg = _her_goal_obs_parts(info["goal_obs"])

    assert obs_vec.shape == (HER_OBS_DIM,)
    assert np.allclose(np.concatenate([obs_vec, dg]), flat_obs["state"])
    assert np.allclose(obs_vec[OBSERVATION_DIM:], ag)


class _StubLogger:
    """Just enough Logger surface for _run_off_policy."""

    def __init__(self) -> None:
        self.metrics: list[dict] = []

    def update_episodes(self, reward, done, truncated=None, *, step, info=None) -> None:
        pass

    def set_step(self, step) -> None:
        pass

    def record_metrics(self, metrics, *, step, total_steps, prefix=None, console=True) -> None:
        self.metrics.append(dict(metrics))


def test_run_off_policy_collects_her_episodes_end_to_end() -> None:
    """The real CLI off-policy loop, against a goal env, with HER enabled:
    episodes must accumulate in the HER buffer and updates must actually run."""
    from srl.cli.train import _configure_her_from_env, _run_off_policy

    env = _make_goal_env(episode_len=5)
    cfg = SACConfig(
        action_dim=ACTION_DIM,
        batch_size=8,
        use_her=True,
        her_max_episode_len=5,
        buffer_size=1000,
        learning_starts=10,
        train_freq=5,
        gradient_steps=1,
    )
    _configure_her_from_env(cfg, env, "goal")
    agent = _build_sac(cfg)
    assert isinstance(agent.buffer, HERReplayBuffer)

    args = types.SimpleNamespace(
        seed=0,
        steps=60,
        env="FakeGoal-v0",
        env_type="goal",
        eval_freq=0,
        log_interval=1000,
    )
    logger = _StubLogger()
    _run_off_policy(agent, env, args, [], logger, start_step=0, device="cpu")

    # 60 steps of 5-step episodes, collected through add_transition().
    assert len(agent.buffer) == 12, f"expected 12 committed episodes, got {len(agent.buffer)}"
    assert agent.buffer.num_transitions == 60

    her_metrics = [m for m in logger.metrics if "her/episodes" in m]
    assert her_metrics, "no HER metrics were logged -- did update() ever run?"
    assert her_metrics[-1]["her/episodes"] > 0
    losses = [m["sac/critic_loss"] for m in logger.metrics if "sac/critic_loss" in m]
    assert losses, "SAC never performed an update on HER batches"
    assert all(np.isfinite(v) for v in losses)


def test_run_off_policy_without_her_still_uses_plain_buffer() -> None:
    """Guard against the HER branch capturing the non-HER path."""
    from srl.cli.train import _run_off_policy

    env = _make_goal_env(episode_len=5)
    agent = _build_sac(
        SACConfig(
            action_dim=ACTION_DIM,
            batch_size=8,
            buffer_size=1000,
            learning_starts=10,
            train_freq=5,
            gradient_steps=1,
        )
    )
    assert isinstance(agent.buffer, ReplayBuffer)

    args = types.SimpleNamespace(
        seed=0, steps=30, env="FakeGoal-v0", env_type="goal", eval_freq=0, log_interval=1000
    )
    _run_off_policy(agent, env, args, [], _StubLogger(), start_step=0, device="cpu")
    assert len(agent.buffer) == 30
