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


def _train(
    tmp_path: Path,
    cfg: dict,
    name: str,
    *,
    eval_episodes: int,
    eval_freq: int,
    seed: int = 0,
) -> tuple[dict, list[float]]:
    """Run ``srl-train`` on ``cfg``; return the run's ``last_metrics`` and the
    ``eval/score_mean`` history (in step order)."""
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
            str(seed),
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
    run_dir = logdir / f"{algo}_{name}"
    summary_path = run_dir / "summary.json"
    assert summary_path.exists(), f"no summary.json at {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evals: dict[int, float] = {}
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("tag") == "eval/score_mean":
            evals[int(record["step"])] = float(record["value"])
    return summary.get("last_metrics") or {}, [evals[k] for k in sorted(evals)]


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
    metrics, _ = _train(tmp_path, cfg, "sac_pendulum", eval_episodes=10, eval_freq=5_000)
    score = metrics.get("eval/score_mean")
    assert score is not None, f"no eval/score_mean in {sorted(metrics)}"
    assert score >= -250.0, f"SAC Pendulum eval/score_mean={score:.1f} < -250"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ppo_pendulum_learns(tmp_path: Path, seed: int) -> None:
    """PPO, Pendulum-v1, ~200k steps -> mean of the last 3 evals >= -400, per seed.

    The mean of the last three evaluations (160k/180k/200k steps) rather than
    the final one: a single 10-episode eval of a near-deterministic policy is
    noisy, and the regression this guards against is a policy that learns and
    then collapses (see the lr note below), which shows up in all three.
    """
    cfg = yaml.safe_load((ENV_CONFIGS / "pendulum_ppo.yaml").read_text(encoding="utf-8"))
    cfg["algo"] = "ppo"
    # rl-baselines3-zoo-style Pendulum-v1 PPO hyperparameters (minus gSDE,
    # which SRL does not implement): short horizon (gamma=0.9), large
    # rollouts, no entropy bonus, obs + reward normalisation (defaults) --
    # but lr 3e-4 / batch 256 instead of rl-zoo's 1e-3 / 64. Without gSDE
    # the state-independent log-std keeps shrinking (no entropy bonus), and
    # with 640 minibatch steps of lr 1e-3 per rollout the per-update KL
    # grows with it (KL ~ d_mean^2 / std^2) until single updates reach
    # approx_kl ~ 0.4-3.7 and the learned policy collapses (-130 -> -800 on
    # seed 0). 160 steps of lr 3e-4 keep approx_kl ~ 0.01.
    cfg["train"] = {
        "total_steps": 204_800,  # 50 rollouts of 4 envs x 1024 steps
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "lr": 3e-4,
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    _, evals = _train(tmp_path, cfg, "ppo_pendulum", eval_episodes=10, eval_freq=20_000, seed=seed)
    assert len(evals) >= 3, f"expected >= 3 evaluations, got {evals}"
    last3 = sum(evals[-3:]) / 3
    print(f"PPO Pendulum seed {seed}: last3={last3:.1f} evals={[round(e) for e in evals]}")
    assert last3 >= -400.0, (
        f"PPO Pendulum seed {seed}: mean of last 3 evals = {last3:.1f} < -400 "
        f"(eval history: {[round(e) for e in evals]})"
    )


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
    metrics, _ = _train(tmp_path, cfg, "sac_her_fetch_reach", eval_episodes=20, eval_freq=5_000)
    success = metrics.get("eval/success_mean")
    assert success is not None, f"no eval/success_mean in {sorted(metrics)}"
    assert success >= 0.9, f"FetchReach SAC+HER eval/success_mean={success:.2f} < 0.9"
