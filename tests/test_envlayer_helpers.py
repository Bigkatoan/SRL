"""Tiny deterministic Gymnasium envs shared by the env-layer regression tests.

Contains no tests itself; importable from other test modules because
pytest prepends ``tests/`` to ``sys.path`` (rootdir has no ``__init__.py``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import yaml

COUNTER_EP_LEN = 4


class CounterEnv(gym.Env):
    """obs = [t, marker]; truncates (never terminates) at ``t == ep_len``.

    ``t`` resets to 0 on reset, so a post-reset obs is trivially
    distinguishable from the true terminal one (``t == ep_len``).
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, ep_len: int = COUNTER_EP_LEN, low: float = -1.0, high: float = 1.0):
        self.ep_len = ep_len
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low, high, shape=(1,), dtype=np.float32)
        self.t = 0
        self.last_action: np.ndarray | None = None

    def _obs(self) -> np.ndarray:
        return np.array([self.t, 1.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.t += 1
        truncated = self.t >= self.ep_len
        return self._obs(), 1.0, False, truncated, {}


class SuccessAtEndEnv(CounterEnv):
    """Reports ``is_success=1`` only on the final step of each episode."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = {"is_success": 1.0 if truncated else 0.0}
        return obs, reward, terminated, truncated, info


def register_test_envs() -> None:
    ids = gym.registry.keys()
    if "SRLCounter-v0" not in ids:
        gym.register("SRLCounter-v0", entry_point=CounterEnv)
    if "SRLCounterWide-v0" not in ids:
        gym.register("SRLCounterWide-v0", entry_point=CounterEnv, kwargs={"low": -3.0, "high": 5.0})
    if "SRLSuccessAtEnd-v0" not in ids:
        gym.register("SRLSuccessAtEnd-v0", entry_point=SuccessAtEndEnv)


# Registered on import so `gym.make("test_envlayer_helpers:SRLCounter-v0")`
# also works inside spawn/forkserver AsyncVectorEnv workers.
register_test_envs()


def mlp_config(algo: str, obs_dim: int = 2, action_dim: int = 1, **train) -> dict:
    """Small YAML model config for SAC or PPO."""
    enc = [
        {
            "name": "state_enc",
            "type": "mlp",
            "input_dim": obs_dim,
            "latent_dim": 16,
            "layers": [{"out_features": 16, "activation": "relu", "norm": "none"}],
        }
    ]
    if algo == "sac":
        cfg = {
            "algo": "sac",
            "encoders": enc,
            "flows": ["state_enc -> actor", "state_enc -> critic"],
            "actor": {
                "name": "actor",
                "type": "squashed_gaussian",
                "action_dim": action_dim,
                "log_std_min": -5.0,
                "log_std_max": 2.0,
            },
            "critic": {"name": "critic", "type": "twin_q", "action_dim": action_dim},
            "losses": [
                {"name": "sac_q", "weight": 1.0},
                {"name": "sac_policy", "weight": 1.0},
                {"name": "sac_temperature", "weight": 1.0},
            ],
            "train": {
                "buffer_size": 1000,
                "batch_size": 8,
                "start_steps": 8,
                "update_after": 8,
                "update_every": 4,
                "gradient_steps": 1,
                "replay_n_step": 1,
                "use_fp16": False,
            },
        }
    elif algo == "ppo":
        cfg = {
            "algo": "ppo",
            "encoders": enc,
            "flows": ["state_enc -> actor", "state_enc -> critic"],
            "actor": {
                "name": "actor",
                "type": "gaussian",
                "action_dim": action_dim,
                "log_std_init": -1.0,
            },
            "critic": {"name": "critic", "type": "value"},
            "losses": [
                {"name": "policy", "weight": 1.0},
                {"name": "value", "weight": 1.0},
                {"name": "entropy", "weight": 0.01},
            ],
            "train": {"n_steps": 8, "batch_size": 8, "n_epochs": 1},
        }
    else:
        raise ValueError(algo)
    cfg["train"].update(train)
    return cfg


def run_cli(monkeypatch, tmp_path, cfg: dict, extra=(), env_id="SRLCounter-v0", on_agent=None):
    from srl.cli import train as train_module

    register_test_envs()
    captured = {}
    for name in ("_run_off_policy", "_run_on_policy"):
        # Always wrap the REAL loop (not a spy left by an earlier run_cli
        # call in the same test), so hooks never stack across runs.
        original = getattr(train_module, name)
        original = getattr(original, "_srl_test_original", original)

        def _spy(agent, *a, _orig=original, **k):
            captured["agent"] = agent
            if on_agent is not None:
                on_agent(agent)
            return _orig(agent, *a, **k)

        _spy._srl_test_original = original
        monkeypatch.setattr(train_module, name, _spy)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    code = train_module.main(
        [
            "--config",
            str(path),
            "--env",
            env_id,
            "--n-envs",
            "2",
            "--vec-mode",
            "sync",
            "--device",
            "cpu",
            "--eval-freq",
            "0",
            "--no-plots",
            "--logdir",
            str(tmp_path / "runs"),
            "--ckptdir",
            str(tmp_path / "ckpt"),
            *extra,
        ]
    )
    assert code == 0
    return captured["agent"]


def record_buffer_adds(store: list):
    def _hook(agent):
        original_add = agent.buffer.add

        def _add(*a, **k):
            store.append(k)
            return original_add(*a, **k)

        agent.buffer.add = _add

    return _hook
