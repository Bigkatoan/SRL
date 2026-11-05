"""Off-policy training metrics are averaged over ``metrics_log_interval`` env
steps before reaching the logger; callbacks still see every update round."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from srl import SAC
from srl.config import TrainerConfig
from srl.training import Callback

_CFG = {
    "algo": "sac",
    "encoders": [
        {
            "name": "state_enc",
            "type": "mlp",
            "input_dim": 3,
            "latent_dim": 16,
            "layers": [{"out_features": 16, "activation": "tanh"}],
        }
    ],
    "flows": ["state_enc -> actor", "state_enc -> critic"],
    "actor": {"name": "actor", "type": "squashed_gaussian", "action_dim": 1},
    "critic": {"name": "critic", "type": "twin_q", "action_dim": 1},
    "train": {
        "buffer_size": 500,
        "batch_size": 8,
        "learning_starts": 16,
        "train_freq": 1,
        "gradient_steps": 2,
    },
}


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict]] = []

    def configure_env(self, num_envs: int) -> None:
        pass

    def set_step(self, step: int) -> None:
        pass

    def update_episodes(self, *args, **kwargs) -> None:
        pass

    def record_metrics(self, metrics, *, step, total_steps=None, prefix=None, console=True):
        self.records.append((int(step), dict(metrics)))


class _CountUpdates(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.steps: list[int] = []

    def on_update(self, step: int, metrics: dict) -> bool:
        self.steps.append(step)
        return True


def _train(interval: int | None, steps: int = 116):
    torch.manual_seed(0)
    agent = SAC.from_config(_CFG, env=gym.make("Pendulum-v1"))
    logger, counter = _RecordingLogger(), _CountUpdates()
    kwargs = {} if interval is None else {"metrics_log_interval": interval}
    agent.learn(steps, logger=logger, callbacks=[counter], **kwargs)
    return logger.records, counter.steps


def test_metrics_are_averaged_over_windows() -> None:
    records, update_steps = _train(40)
    # 101 update rounds (steps 16..116); callbacks see each of them.
    assert update_steps == list(range(16, 117))
    # Windows close once they span >= 40 env steps since the last log
    # (start 0): 40, 80, then the final partial window flushed at 116.
    assert [step for step, _ in records] == [40, 80, 116]
    for _, metrics in records:
        assert metrics["train/utd_ratio"] > 0
        assert all(isinstance(v, float) for v in metrics.values())
    # 2 gradient steps per env step inside a full window (steps 41..80).
    assert records[1][1]["train/utd_ratio"] == pytest.approx(2.0)


def test_interval_one_logs_every_update_round() -> None:
    records, update_steps = _train(1)
    assert [step for step, _ in records] == update_steps


def test_default_interval_is_1000() -> None:
    assert TrainerConfig().metrics_log_interval == 1000
    records, _ = _train(None)
    assert [step for step, _ in records] == [116]  # only the final flush


def test_trainer_config_rejects_non_positive_interval() -> None:
    from srl.config import ConfigError

    with pytest.raises(ConfigError):
        TrainerConfig(metrics_log_interval=0)
