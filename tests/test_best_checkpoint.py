"""Tests for `--save-best`/`train.save_best`: BestCheckpointTracker and its
wiring into `_maybe_run_evaluation`.

Motivating bug (found on a real 20M-step PPO run against JAVIS's mjlab
balance task): eval score rose to a peak partway through training, then
declined continuously for the rest of the run as policy entropy collapsed.
Only a `final_*` checkpoint was ever saved, so the actual peak policy was
unrecoverable -- there was no mechanism tracking "best eval score seen" at
all, independent of "most recent periodic/final checkpoint." These tests
simulate exactly that rise-then-decline eval trajectory and assert the
`best_*` checkpoint this feature saves reflects the true peak, not
whatever the run's last eval happened to be.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

import srl.cli.train as train_module
from srl.utils.checkpoint import BestCheckpointTracker, CheckpointManager

# ──────────────────────────────────────────────────────────────────────────
# Unit tests: BestCheckpointTracker in isolation
# ──────────────────────────────────────────────────────────────────────────


def test_is_better_max_mode() -> None:
    tracker = BestCheckpointTracker(mode="max")
    assert tracker.is_better(1.0) is True  # nothing seen yet -> anything is best
    tracker.best_score = 1.5
    assert tracker.is_better(2.0) is True
    assert tracker.is_better(1.5) is False  # tie does not count as an improvement
    assert tracker.is_better(1.0) is False


def test_is_better_min_mode() -> None:
    tracker = BestCheckpointTracker(mode="min")
    tracker.best_score = 1.0
    assert tracker.is_better(0.5) is True
    assert tracker.is_better(1.5) is False


def test_invalid_mode_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        BestCheckpointTracker(mode="sideways")


def test_update_only_saves_on_improvement(tmp_path) -> None:
    model = nn.Linear(2, 2)
    cm = CheckpointManager(tmp_path, max_keep=5)
    tracker = BestCheckpointTracker(mode="max")

    saved1 = tracker.update(1.0, model, cm=cm, step=100, metrics={"eval/score_mean": 1.0})
    assert saved1 is not None
    assert tracker.best_score == 1.0
    assert tracker.best_step == 100

    # Worse score: no save, tracker state unchanged.
    saved2 = tracker.update(0.5, model, cm=cm, step=200, metrics={"eval/score_mean": 0.5})
    assert saved2 is None
    assert tracker.best_score == 1.0
    assert tracker.best_step == 100

    # New best: saves again.
    saved3 = tracker.update(1.8, model, cm=cm, step=300, metrics={"eval/score_mean": 1.8})
    assert saved3 is not None
    assert tracker.best_score == 1.8
    assert tracker.best_step == 300
    assert saved3 != saved1


def test_seed_from_checkpoint_reads_stored_metric() -> None:
    tracker = BestCheckpointTracker(mode="max")
    payload = {"step": 5_000_000, "metrics": {"eval/score_mean": 1.87}}
    tracker.seed_from_checkpoint(payload, monitor="eval/score_mean")
    assert tracker.best_score == 1.87
    assert tracker.best_step == 5_000_000


def test_seed_from_checkpoint_missing_monitor_is_noop() -> None:
    tracker = BestCheckpointTracker(mode="max")
    tracker.seed_from_checkpoint({"step": 1, "metrics": {}}, monitor="eval/score_mean")
    assert tracker.best_score is None


# ──────────────────────────────────────────────────────────────────────────
# Eviction isolation: a shared periodic-checkpoint FIFO must not be able to
# evict the best checkpoint before something actually better replaces it.
# ──────────────────────────────────────────────────────────────────────────


def test_best_checkpoint_survives_periodic_checkpoint_churn(tmp_path) -> None:
    """Regression guard for the exact bug BestCheckpointTracker's docstring
    warns about: if "best" saves shared the periodic CheckpointManager's
    save/eviction FIFO (max_keep=5 in real training), enough later periodic
    `ckpt_*` saves would silently evict an early, genuinely-best `best_*`
    checkpoint. Using a *separate* CheckpointManager instance for best-saves
    must prevent that regardless of how much periodic churn follows.
    """
    model = nn.Linear(2, 2)
    periodic_cm = CheckpointManager(tmp_path, max_keep=5)
    best_cm = CheckpointManager(tmp_path, max_keep=1)
    tracker = BestCheckpointTracker(mode="max")

    best_path = tracker.update(1.87, model, cm=best_cm, step=3_500_000, metrics={})
    assert best_path is not None
    assert best_path.exists()

    # 10 periodic saves (twice max_keep) on the *other* manager -- more than
    # enough to have evicted the best checkpoint were they sharing a FIFO.
    for i in range(10):
        periodic_cm.save(model, step=(i + 1) * 1_000_000, tag="ckpt")

    assert best_path.exists(), "periodic checkpoint churn must never evict the best checkpoint"
    assert tracker.best_score == 1.87


# ──────────────────────────────────────────────────────────────────────────
# Integration: drive _maybe_run_evaluation across a rise-then-decline eval
# trajectory (same shape as the real PPO run that motivated this feature)
# and confirm the saved best_*.pt reflects the true peak, not the final
# (worse) score.
# ──────────────────────────────────────────────────────────────────────────


class _FakeLogger:
    def record_metrics(self, *args, **kwargs) -> None:
        pass


def _fake_args(eval_freq: int, steps: int, save_best: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        eval_freq=eval_freq,
        env="Pendulum-v1",
        env_type="flat",
        seed=0,
        eval_episodes=1,
        render=False,
        steps=steps,
        save_best=save_best,
    )


def test_best_checkpoint_tracks_true_peak_across_rise_then_decline(monkeypatch, tmp_path) -> None:
    # Mirrors the real PPO run's eval/score_mean trajectory: rises to a peak
    # then declines continuously for the remainder of training (entropy
    # collapse). The point of this feature is that the *final* value here
    # (1.064) must NOT be what best_*.pt ends up holding -- the peak (1.8748)
    # must.
    score_sequence = [
        1.4461,
        1.599,
        1.623,
        1.7447,
        1.8748,  # <- true peak
        1.7518,
        1.5085,
        1.3913,
        1.2057,
        1.064,  # <- final eval, distinctly worse than the peak
    ]
    call_index = {"i": 0}

    def _fake_evaluate_agent(agent, eval_env, **kwargs):
        score = score_sequence[call_index["i"]]
        call_index["i"] += 1
        return {"eval/score_mean": score, "eval/score_max": score, "eval/episodes": 1.0}

    monkeypatch.setattr(train_module, "_evaluate_agent", _fake_evaluate_agent)
    monkeypatch.setattr(train_module, "_make_cli_env", lambda *a, **k: object())

    eval_freq = 1_000_000
    total_steps = eval_freq * len(score_sequence)
    args = _fake_args(eval_freq=eval_freq, steps=total_steps)

    agent = nn.Linear(2, 2)  # stand-in "agent" -- CheckpointManager just needs an nn.Module
    best_cm = CheckpointManager(tmp_path / "best", max_keep=1)
    tracker = BestCheckpointTracker(mode="max")

    eval_env = None
    next_eval_step = eval_freq
    last_eval_step = None
    for cycle in range(len(score_sequence)):
        step = eval_freq * (cycle + 1)
        next_eval_step, eval_env, last_eval_step = train_module._maybe_run_evaluation(
            agent,
            args,
            _FakeLogger(),
            device="cpu",
            step=step,
            next_eval_step=next_eval_step,
            eval_env=eval_env,
            last_eval_step=last_eval_step,
            best_cm=best_cm,
            best_tracker=tracker,
        )

    assert call_index["i"] == len(score_sequence), "every eval cycle must have actually run"

    # The tracker must hold the true peak, not the final (worse) score.
    assert tracker.best_score == max(score_sequence)
    assert tracker.best_score != score_sequence[-1]
    assert tracker.best_step == eval_freq * 5  # cycle index 4 (0-based) -> 5th eval

    # And the checkpoint actually on disk must agree -- reload it and check
    # the stored metrics, not just the in-memory tracker.
    best_path = tracker.best_path
    assert best_path is not None
    reloaded = nn.Linear(2, 2)
    payload = best_cm.load(reloaded, best_path, device="cpu")
    assert payload["metrics"]["eval/score_mean"] == max(score_sequence)
    assert payload["step"] == eval_freq * 5


def test_save_best_disabled_by_default_never_saves(monkeypatch, tmp_path) -> None:
    """best_cm/best_tracker being None (the --save-best-off default) must be
    a true no-op: no best_* file, no error, existing eval behavior unchanged.
    """

    def _fake_evaluate_agent(agent, eval_env, **kwargs):
        return {"eval/score_mean": 1.0, "eval/score_max": 1.0, "eval/episodes": 1.0}

    monkeypatch.setattr(train_module, "_evaluate_agent", _fake_evaluate_agent)
    monkeypatch.setattr(train_module, "_make_cli_env", lambda *a, **k: object())

    args = _fake_args(eval_freq=1000, steps=1000, save_best=False)
    next_eval_step, eval_env, last_eval_step = train_module._maybe_run_evaluation(
        nn.Linear(2, 2),
        args,
        _FakeLogger(),
        device="cpu",
        step=1000,
        next_eval_step=1000,
        eval_env=None,
        last_eval_step=None,
        best_cm=None,
        best_tracker=None,
    )
    assert last_eval_step == 1000
    assert not list(tmp_path.glob("best_*"))


# ──────────────────────────────────────────────────────────────────────────
# --save-best / --no-save-best CLI parsing
# ──────────────────────────────────────────────────────────────────────────


def test_save_best_cli_flag_defaults_to_none_for_yaml_fallback() -> None:
    from srl.cli.train import _build_parser

    args = _build_parser().parse_args(["--config", "cfg.yaml"])
    assert args.save_best is None


def test_save_best_cli_flag_explicit_true() -> None:
    from srl.cli.train import _build_parser

    args = _build_parser().parse_args(["--config", "cfg.yaml", "--save-best"])
    assert args.save_best is True


def test_save_best_cli_flag_explicit_false() -> None:
    from srl.cli.train import _build_parser

    args = _build_parser().parse_args(["--config", "cfg.yaml", "--no-save-best"])
    assert args.save_best is False


# ──────────────────────────────────────────────────────────────────────────
# --resume seeding
# ──────────────────────────────────────────────────────────────────────────


def test_seed_best_tracker_from_existing_checkpoint_on_disk(tmp_path) -> None:
    model = nn.Linear(2, 2)
    best_cm = CheckpointManager(tmp_path, max_keep=1)
    seed_tracker = BestCheckpointTracker(mode="max")
    seed_tracker.update(1.6, model, cm=best_cm, step=4_000_000, metrics={"eval/score_mean": 1.6})

    # Simulate a fresh process (a real --resume): a brand new tracker with no
    # in-memory history, seeded only from what's on disk.
    resumed_tracker = BestCheckpointTracker(mode="max")
    train_module._seed_best_tracker(best_cm, resumed_tracker, device="cpu")

    assert resumed_tracker.best_score == 1.6
    assert resumed_tracker.best_step == 4_000_000

    # A worse post-resume eval must NOT overwrite the pre-resume best.
    saved = resumed_tracker.update(
        1.2, model, cm=best_cm, step=4_500_000, metrics={"eval/score_mean": 1.2}
    )
    assert saved is None
    assert resumed_tracker.best_score == 1.6


def test_seed_best_tracker_noop_when_no_prior_checkpoint(tmp_path) -> None:
    best_cm = CheckpointManager(tmp_path, max_keep=1)
    tracker = BestCheckpointTracker(mode="max")
    train_module._seed_best_tracker(best_cm, tracker, device="cpu")
    assert tracker.best_score is None
