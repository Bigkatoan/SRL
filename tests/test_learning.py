"""End-to-end learning regression tests (``@pytest.mark.slow``).

These train real agents through the ``srl-train`` entry point
(:func:`srl.cli.train.main`) on small, well-understood benchmarks and assert
that the final evaluation clears a conservative threshold. They are the
library's guard against "the code runs but the agent no longer learns"
regressions, which the fast unit suite cannot catch.

They are excluded from the default CI job and run by the nightly/opt-in
``learning`` job in ``.github/workflows/test.yml``::

    pytest -m slow tests/test_learning.py

.. note::
   The thresholds below are the *targets* for the Phase-0 algorithm fixes.
   Until those fixes land some of these tests are expected to fail on score
   (they must still collect and train without raising).

Each test writes its YAML config into ``tmp_path`` (derived from a shipped
config where possible) so the hyperparameters used for the regression gate
are pinned here and do not drift when the example configs are re-tuned.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIGS = REPO_ROOT / "configs" / "envs"

pytestmark = pytest.mark.slow


def _train(tmp_path: Path, cfg: dict, name: str, *, eval_episodes: int, eval_freq: int) -> dict:
    """Run ``srl-train`` on ``cfg`` and return the run's ``last_metrics``."""
    from srl.cli.train import main

    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    logdir = tmp_path / "runs"
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--logdir",
            str(logdir),
            "--ckptdir",
            str(tmp_path / "checkpoints"),
            "--eval-freq",
            str(eval_freq),
            "--eval-episodes",
            str(eval_episodes),
            "--no-plots",
        ]
    )
    assert exit_code in (0, None), f"srl-train exited with {exit_code}"
    algo = cfg["algo"]
    summary_path = logdir / f"{algo}_{name}" / "summary.json"
    assert summary_path.exists(), f"no summary.json at {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("last_metrics") or {}


def _mlp_encoder(name: str, input_dim: int, width: int, activation: str = "relu") -> dict:
    return {
        "name": name,
        "type": "mlp",
        "input_dim": input_dim,
        "latent_dim": width,
        "layers": [
            {"out_features": width, "activation": activation},
            {"out_features": width, "activation": activation},
        ],
    }


def test_sac_pendulum_learns(tmp_path: Path) -> None:
    """SAC, Pendulum-v1, 20k steps -> eval score >= -250 (random policy ~ -1200)."""
    cfg = {
        "env_id": "Pendulum-v1",
        "env_type": "flat",
        "algo": "sac",
        "encoders": [
            _mlp_encoder("actor_state_enc", 3, 256),
            _mlp_encoder("critic_state_enc", 3, 256),
        ],
        "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
        "actor": {
            "name": "actor",
            "type": "squashed_gaussian",
            "action_dim": 1,
            "log_std_min": -5.0,
            "log_std_max": 2.0,
            "layers": [{"out_features": 256, "activation": "relu"}],
        },
        "critic": {
            "name": "critic",
            "type": "twin_q",
            "action_dim": 1,
            "layers": [{"out_features": 256, "activation": "relu"}],
        },
        "losses": [
            {"name": "sac_q", "weight": 1.0},
            {"name": "sac_policy", "weight": 1.0},
            {"name": "sac_temperature", "weight": 1.0},
        ],
        # rl-baselines3-zoo Pendulum-v1 SAC hyperparameters.
        "train": {
            "total_steps": 20_000,
            "n_envs": 1,
            "lr_actor": 1e-3,
            "lr_critic": 1e-3,
            "lr_alpha": 1e-3,
            "buffer_size": 200_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "learning_starts": 1_000,
        },
    }
    metrics = _train(tmp_path, cfg, "sac_pendulum", eval_episodes=10, eval_freq=5_000)
    score = metrics.get("eval/score_mean")
    assert score is not None, f"no eval/score_mean in {sorted(metrics)}"
    assert score >= -250.0, f"SAC Pendulum eval/score_mean={score:.1f} < -250"


def test_ppo_pendulum_learns(tmp_path: Path) -> None:
    """PPO, Pendulum-v1, ~150k steps -> eval score >= -400."""
    cfg = yaml.safe_load((ENV_CONFIGS / "pendulum_ppo.yaml").read_text(encoding="utf-8"))
    cfg["algo"] = "ppo"
    # rl-baselines3-zoo-style Pendulum-v1 PPO hyperparameters (minus gSDE,
    # which SRL does not implement): short horizon (gamma=0.9), large
    # rollouts, high learning rate, no entropy bonus.
    cfg["train"] = {
        "total_steps": 151_552,  # 37 rollouts of 4 envs x 1024 steps
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "lr": 1e-3,
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    metrics = _train(tmp_path, cfg, "ppo_pendulum", eval_episodes=10, eval_freq=50_000)
    score = metrics.get("eval/score_mean")
    assert score is not None, f"no eval/score_mean in {sorted(metrics)}"
    assert score >= -400.0, f"PPO Pendulum eval/score_mean={score:.1f} < -400"


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium_robotics") is None
    or importlib.util.find_spec("mujoco") is None,
    reason="FetchReach needs gymnasium-robotics + MuJoCo: pip install 'srl-rl[robotics]'",
)
def test_sac_her_fetch_reach_learns(tmp_path: Path) -> None:
    """SAC + HER, FetchReach-v4, 10k steps -> eval success rate >= 0.9."""
    cfg = yaml.safe_load((ENV_CONFIGS / "fetch_reach_sac_her.yaml").read_text(encoding="utf-8"))
    cfg["train"]["total_steps"] = 10_000
    cfg["train"]["buffer_size"] = 100_000
    metrics = _train(tmp_path, cfg, "sac_her_fetch_reach", eval_episodes=20, eval_freq=5_000)
    success = metrics.get("eval/success_mean")
    assert success is not None, f"no eval/success_mean in {sorted(metrics)}"
    assert success >= 0.9, f"FetchReach SAC+HER eval/success_mean={success:.2f} < 0.9"
