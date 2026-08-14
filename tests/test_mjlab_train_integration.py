"""End-to-end regression test for the isaaclab/mjlab code path in srl-train.

Runs `srl.cli.train.main()` for real (on-policy PPO and off-policy SAC,
including the `--eval-freq` path) against a fake env shaped exactly like
IsaacLabWrapper's real input/output contract (batched dict obs, a lightweight
action-space dataclass with no `.sample()`, separate terminated/truncated).
No GPU, Isaac Sim, or mjlab install required.

This is the regression net for a cluster of bugs that only ever showed up on
the real isaaclab/mjlab path and were invisible to the plain-Gymnasium tests:
`_make_cli_env`'s isaaclab/mjlab branch, the internal-vectorization branch in
`main()`, `_split_vector_transition`, the async-runner-adjacent
`sample_action_space` fallback, and the obs_remap broadcast rule that
`configs/envs/isaaclab_cartpole_ppo.yaml`-shaped configs (one obs group, two
unnamed encoders) depend on.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from srl.cli import train as train_module
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

OBS_DIM = 4
ACTION_DIM = 1
EPISODE_LEN = 5


class _FakeActionSpace:
    """Mirrors isaaclab/mjlab's lightweight space (shape/low/high, no .sample())."""

    def __init__(self, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> None:
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class _FakeMjlabBaseEnv:
    """Duck-typed like mjlab's ManagerBasedRlEnv / isaaclab's ManagerBasedRLEnv."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        # Batched action space (num_envs, action_dim), matching gymnasium's
        # VectorEnv convention -- and single_action_space for per-env sampling,
        # matching the real distinction IsaacLabWrapper.single_act_space relies on.
        self.action_space = _FakeActionSpace((num_envs, ACTION_DIM))
        self.single_action_space = _FakeActionSpace((ACTION_DIM,))
        self._t = np.zeros(num_envs, dtype=np.int64)
        self._rng = np.random.default_rng(0)

    def reset(self, **kwargs):
        self._t[:] = 0
        obs = {"policy": np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)}
        return obs, {}

    def step(self, actions):
        self._t += 1
        obs = {"policy": self._rng.standard_normal((self.num_envs, OBS_DIM)).astype(np.float32)}
        reward = np.ones(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = self._t >= EPISODE_LEN
        self._t = np.where(truncated, 0, self._t)
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        pass


def _fake_make_cli_env(env_name, device, n_envs, env_type="flat"):
    return IsaacLabWrapper(_FakeMjlabBaseEnv(num_envs=n_envs))


PPO_CONFIG = {
    "env_id": "FakeMjlabCartpole",
    "env_type": "mjlab",
    "algo": "ppo",
    "encoders": [
        {
            "name": "actor_state_enc",
            "type": "mlp",
            "input_dim": OBS_DIM,
            "latent_dim": 16,
            "layers": [{"out_features": 16, "activation": "elu", "norm": "none"}],
        },
        {
            "name": "critic_state_enc",
            "type": "mlp",
            "input_dim": OBS_DIM,
            "latent_dim": 16,
            "layers": [{"out_features": 16, "activation": "elu", "norm": "none"}],
        },
    ],
    "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
    "actor": {
        "name": "actor",
        "type": "gaussian",
        "action_dim": ACTION_DIM,
        "log_std_init": -1.0,
        "layers": [{"out_features": 16, "activation": "elu", "norm": "none"}],
    },
    "critic": {
        "name": "critic",
        "type": "value",
        "layers": [{"out_features": 16, "activation": "elu", "norm": "none"}],
    },
    "losses": [
        {"name": "policy", "weight": 1.0},
        {"name": "value", "weight": 1.0},
        {"name": "entropy", "weight": 0.01},
    ],
    "train": {
        "total_steps": 32,
        "n_envs": 4,
        "n_steps": 4,
        "batch_size": 8,
        "n_epochs": 1,
        "lr": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coef": 0.01,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
    },
}

SAC_CONFIG = {
    "env_id": "FakeMjlabCartpole",
    "env_type": "mjlab",
    "algo": "sac",
    "encoders": [
        {
            "name": "actor_state_enc",
            "type": "mlp",
            "input_dim": OBS_DIM,
            "latent_dim": 16,
            "layers": [{"out_features": 16, "activation": "relu", "norm": "none"}],
        },
        {
            "name": "critic_state_enc",
            "type": "mlp",
            "input_dim": OBS_DIM,
            "latent_dim": 16,
            "layers": [{"out_features": 16, "activation": "relu", "norm": "none"}],
        },
    ],
    "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
    "actor": {
        "name": "actor",
        "type": "squashed_gaussian",
        "action_dim": ACTION_DIM,
        "log_std_min": -5.0,
        "log_std_max": 2.0,
        "layers": [{"out_features": 16, "activation": "relu", "norm": "none"}],
    },
    "critic": {
        "name": "critic",
        "type": "twin_q",
        "action_dim": ACTION_DIM,
        "layers": [{"out_features": 16, "activation": "relu", "norm": "none"}],
    },
    "losses": [
        {"name": "sac_q", "weight": 1.0},
        {"name": "sac_policy", "weight": 1.0},
        {"name": "sac_temperature", "weight": 1.0},
    ],
    "train": {
        "total_steps": 40,
        "n_envs": 4,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "lr_alpha": 3e-4,
        "buffer_size": 1000,
        "batch_size": 8,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "auto_entropy_tuning": False,
        "start_steps": 8,
        "update_after": 8,
        "update_every": 4,
        "gradient_steps": 2,
        "replay_n_step": 1,
        "use_fp16": False,
    },
}


def _write_config(tmp_path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


@pytest.mark.parametrize(
    "config_dict, config_name", [(PPO_CONFIG, "ppo.yaml"), (SAC_CONFIG, "sac.yaml")]
)
def test_srl_train_main_runs_full_loop_on_fake_mjlab_env(
    monkeypatch, tmp_path, config_dict, config_name
) -> None:
    monkeypatch.setattr(train_module, "_make_cli_env", _fake_make_cli_env)

    config_path = _write_config(tmp_path, config_name, config_dict)
    total_steps = config_dict["train"]["total_steps"]

    exit_code = train_module.main(
        [
            "--config",
            config_path,
            "--logdir",
            str(tmp_path / "runs"),
            "--ckptdir",
            str(tmp_path / "checkpoints"),
            "--eval-freq",
            str(total_steps // 2),
            "--eval-episodes",
            "1",
            "--no-plots",
            "--seed",
            "0",
        ]
    )

    assert exit_code == 0
    checkpoint_dir = tmp_path / "checkpoints"
    assert any(checkpoint_dir.rglob("*.pt")), "expected at least one checkpoint to be written"


@pytest.mark.parametrize(
    "config_dict, config_name", [(PPO_CONFIG, "ppo.yaml"), (SAC_CONFIG, "sac.yaml")]
)
def test_eval_env_is_built_once_and_reused_across_multiple_eval_cycles(
    monkeypatch, tmp_path, config_dict, config_name
) -> None:
    """Regression test: the eval env must be built once per run and reused,
    not rebuilt every eval cycle. Confirmed on a real mjlab task that
    rebuilding it every cycle produced a wall of repeated construction-
    banner logging (Warp module loads, `Setting seed`, manager tables) that
    reads exactly like the environment "repeatedly reinitializing"."""
    call_count = 0
    real_fake_make_cli_env = _fake_make_cli_env

    def _counting_fake_make_cli_env(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_fake_make_cli_env(*args, **kwargs)

    monkeypatch.setattr(train_module, "_make_cli_env", _counting_fake_make_cli_env)

    config_path = _write_config(tmp_path, config_name, config_dict)
    total_steps = config_dict["train"]["total_steps"]
    # Small enough eval_freq that periodic eval fires several times within
    # total_steps (not just once, and not just the final-step eval).
    eval_freq = max(total_steps // 4, 1)

    exit_code = train_module.main(
        [
            "--config",
            config_path,
            "--logdir",
            str(tmp_path / "runs"),
            "--ckptdir",
            str(tmp_path / "checkpoints"),
            "--eval-freq",
            str(eval_freq),
            "--eval-episodes",
            "1",
            "--no-plots",
            "--seed",
            "0",
        ]
    )

    assert exit_code == 0
    # One call to build the training env, at most one more to build the eval
    # env (lazily, on the first eval cycle that actually fires) -- never one
    # per eval cycle.
    assert call_count <= 2, (
        f"_make_cli_env was called {call_count} times; expected at most 2 "
        "(training env once, eval env once) -- the eval env must be reused "
        "across eval cycles, not rebuilt every time"
    )


def test_resolve_env_action_dim_uses_single_env_space_not_batched() -> None:
    """Regression test: action_dim must be the per-env dimensionality, not
    num_envs * action_dim. Confirmed via a real mjlab run that the old code
    (using env.act_space, the BATCHED space) collapsed SAC's target_entropy
    to -128 (instead of -2) at num_envs=64, driving alpha to ~0 within a few
    thousand steps -- DDPG/TD3 read the same action_dim for their
    exploration-noise generators and were equally affected."""
    env = IsaacLabWrapper(_FakeMjlabBaseEnv(num_envs=64))
    assert env.act_space.shape == (64, ACTION_DIM)  # the batched space
    assert env.single_act_space.shape == (ACTION_DIM,)  # the per-env space

    assert train_module._resolve_env_action_dim(env) == ACTION_DIM


def test_resolve_env_action_dim_falls_back_when_no_single_act_space() -> None:
    """Plain Gymnasium/goal/racecar wrappers have no single_act_space (only
    isaaclab/mjlab envs pre-batch) -- must fall back to act_space directly,
    which is already the correct per-env space for those."""

    class _PlainEnv:
        act_space = _FakeActionSpace((3,))

    assert train_module._resolve_env_action_dim(_PlainEnv()) == 3
