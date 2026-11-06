"""srl-train algorithm resolution: explicit ``algo:``/``--algo`` required;
file-name guessing is deprecated; never silently defaults to PPO."""

from __future__ import annotations

import pytest
import yaml
from test_public_api import model_config

from srl.cli.train import _resolve_algo, main
from srl.training import ConfigurationError


def test_cli_flag_wins_over_yaml() -> None:
    assert _resolve_algo("SAC", {"algo": "ppo"}, "x.yaml") == "sac"
    assert _resolve_algo(None, {"algo": "TD3"}, "x.yaml") == "td3"


def test_filename_guess_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="algo"):
        assert _resolve_algo(None, {}, "configs/my_sac_run.yaml") == "sac"


def test_missing_algo_is_an_error_not_a_ppo_default() -> None:
    with pytest.raises(ConfigurationError, match="does not declare an algorithm"):
        _resolve_algo(None, {}, "configs/experiment.yaml")


def _run(tmp_path, cfg: dict, name: str, *extra: str) -> int:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    return main(
        [
            "--config",
            str(path),
            "--env",
            "Pendulum-v1",
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


def test_main_exits_2_without_algorithm(tmp_path, capsys) -> None:
    cfg = model_config("ppo")
    del cfg["algo"]
    assert _run(tmp_path, cfg, "experiment.yaml", "--steps", "16") == 2
    assert "does not declare an algorithm" in capsys.readouterr().err


def test_main_unknown_algorithm_exits_1(tmp_path) -> None:
    assert _run(tmp_path, model_config("ppo"), "cfg.yaml", "--algo", "dqn") == 1


def test_save_interval_comes_from_the_train_block(tmp_path) -> None:
    cfg = model_config("ppo")
    cfg["train"]["save_interval"] = 16
    assert _run(tmp_path, cfg, "cfg.yaml", "--steps", "48") == 0
    ckpts = sorted((tmp_path / "ckpt" / "ppo_cfg").glob("ckpt_*"))
    assert len(ckpts) == 3


def test_cli_final_checkpoint_loads_through_the_python_api(tmp_path) -> None:
    import gymnasium as gym
    import numpy as np

    from srl import PPO

    assert _run(tmp_path, model_config("ppo"), "cfg.yaml", "--steps", "32") == 0
    final = sorted((tmp_path / "ckpt" / "ppo_cfg").glob("final_*"))[-1]
    agent = PPO.load(final)
    obs, _ = gym.make("Pendulum-v1").reset(seed=0)
    action, _ = agent.predict(obs, deterministic=True)
    assert action.shape == (1,) and np.all(np.abs(action) <= 2.0)


def test_cli_writes_the_resolved_config_to_the_run_dir(tmp_path) -> None:
    from srl.config import load_experiment

    cfg = model_config("ppo")
    cfg["encoders"] = [dict(cfg["encoders"][0], input_dim="auto")]
    cfg["actor"] = dict(cfg["actor"], action_dim="auto")
    assert _run(tmp_path, cfg, "cfg.yaml", "--steps", "32", "--seed", "3") == 0
    exp = load_experiment(tmp_path / "runs" / "ppo_cfg" / "config.yaml")
    assert exp.model.encoders[0].input_dim == 3 and exp.model.actor.action_dim == 1
    # CLI flags are recorded as what actually ran.
    assert exp.trainer.seed == 3 and exp.trainer.total_steps == 32
    assert exp.env.env_id == "Pendulum-v1"
