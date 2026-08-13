"""Regression test: eval/success_mean must appear in the live [eval] console
line for goal-conditioned tasks.

`_evaluate_agent` already computes and writes `eval/success_mean` to
metrics.jsonl/TensorBoard for env_type: goal runs (Fetch/AntMaze), but the
live console print hardcoded only score_mean/episodes -- a user watching
training against a goal-conditioned task never saw the one number that
matters most (whether the policy is actually succeeding) without digging
into log files after the fact.
"""

from __future__ import annotations

from types import SimpleNamespace

import srl.cli.train as train_module


class _FakeLogger:
    def record_metrics(self, *args, **kwargs) -> None:
        pass


def _fake_args(eval_freq: int) -> SimpleNamespace:
    return SimpleNamespace(
        eval_freq=eval_freq,
        env="FetchReach-v4",
        env_type="goal",
        seed=0,
        eval_episodes=1,
        render=False,
        steps=1000,
    )


def test_eval_line_includes_success_mean_for_goal_conditioned_tasks(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        train_module,
        "_evaluate_agent",
        lambda *a, **k: {
            "eval/score_mean": -49.5,
            "eval/score_max": -40.0,
            "eval/episode_length_mean": 50.0,
            "eval/episodes": 1.0,
            "eval/success_mean": 0.75,
        },
    )

    train_module._maybe_run_evaluation(
        agent=object(),
        args=_fake_args(eval_freq=100),
        logger=_FakeLogger(),
        device="cpu",
        step=100,
        next_eval_step=100,
    )

    out = capsys.readouterr().out
    assert "success_mean=0.7500" in out
    assert "score_mean=-49.5000" in out
    assert "| success_mean=0.7500 | episodes=1" in out


def test_eval_line_omits_success_mean_when_not_present(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        train_module,
        "_evaluate_agent",
        lambda *a, **k: {
            "eval/score_mean": -900.0,
            "eval/score_max": -500.0,
            "eval/episode_length_mean": 200.0,
            "eval/episodes": 1.0,
        },
    )

    train_module._maybe_run_evaluation(
        agent=object(),
        args=_fake_args(eval_freq=100),
        logger=_FakeLogger(),
        device="cpu",
        step=100,
        next_eval_step=100,
    )

    out = capsys.readouterr().out
    assert "success_mean" not in out
    assert "score_mean=-900.0000 | episodes=1" in out
