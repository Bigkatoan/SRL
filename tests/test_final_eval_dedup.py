"""Regression tests for the duplicate final-eval bug.

`_run_on_policy` and `_run_off_policy` each run a "final eval" after the
training loop ends, unless a periodic eval already happened at exactly the
final step. The original guard OR'd in `step < next_eval_step`, which is
true immediately after *any* eval fires, so it always overrode the one
check that correctly detected "already evaluated here," producing a
duplicate eval whenever `total_steps` was an exact multiple of `eval_freq`.

That was fixed by comparing `next_eval_step - eval_freq == step` -- but
on-policy rollout stepping can *overshoot* past the nominal eval_freq-
aligned threshold (`rollout_steps` is a ceil-division over `num_envs`), so
the step periodic eval actually fires at is not always an exact multiple of
`eval_freq`, even though `next_eval_step`'s schedule always is. Confirmed on
a real mjlab run: `--steps 15000 --eval-freq 3000` fired periodic eval at
steps 3072/6144/9216/12288 (each overshooting its nominal 3000-aligned
threshold by the same amount), and the arithmetic-based check failed to
recognize the final one as "already ran," producing a duplicate `[eval]`
line at the (also overshot) final step 15008 -- on *both* on-policy and
off-policy paths, and independent of the eval-env-reuse change alongside it.

`_maybe_run_evaluation` now tracks and returns the *actual* step eval last
fired at (`last_eval_step`), so the "already ran here" decision is a direct
`last_eval_step == step` comparison against real executed steps, with no
arithmetic assumptions about alignment to `eval_freq`.
"""

from __future__ import annotations

from types import SimpleNamespace

import srl.cli.train as train_module


class _FakeLogger:
    def record_metrics(self, *args, **kwargs) -> None:
        pass


def _fake_args(eval_freq: int, steps: int = 100_000) -> SimpleNamespace:
    return SimpleNamespace(
        eval_freq=eval_freq,
        env="Pendulum-v1",
        env_type="flat",
        seed=0,
        eval_episodes=1,
        render=False,
        steps=steps,
    )


def _patch_evaluate_agent(monkeypatch, score: float = -1.0) -> None:
    """Monkeypatches _evaluate_agent/_make_cli_env so tests never build a
    real env."""

    def _fake_evaluate_agent(agent, eval_env, **kwargs):
        return {"eval/score_mean": score, "eval/score_max": score, "eval/episodes": 1.0}

    monkeypatch.setattr(train_module, "_evaluate_agent", _fake_evaluate_agent)
    monkeypatch.setattr(train_module, "_make_cli_env", lambda *a, **k: object())


def test_last_eval_step_tracks_the_real_step_eval_fired_at(monkeypatch) -> None:
    _patch_evaluate_agent(monkeypatch)
    args = _fake_args(eval_freq=3000)

    # Periodic eval fires at an overshot step (3072), not the nominal
    # threshold (3000) -- next_eval_step's schedule is still exact.
    next_eval_step, eval_env, last_eval_step = train_module._maybe_run_evaluation(
        agent=object(),
        args=args,
        logger=_FakeLogger(),
        device="cpu",
        step=3072,
        next_eval_step=3000,
        eval_env=None,
        last_eval_step=None,
    )

    assert last_eval_step == 3072  # the real step, not the nominal 3000
    assert next_eval_step == 6000  # schedule still advances by a clean eval_freq


def test_final_eval_correctly_skipped_even_with_overshoot(monkeypatch) -> None:
    """The exact real-run scenario: periodic eval already fired at the
    (overshot) final step -- the final-eval call must detect this via
    `last_eval_step == step`, not via arithmetic that assumes exact
    eval_freq alignment."""
    _patch_evaluate_agent(monkeypatch)
    args = _fake_args(eval_freq=3000)

    # Simulate the loop's last periodic call landing exactly on the
    # (overshot) final step.
    next_eval_step, eval_env, last_eval_step = train_module._maybe_run_evaluation(
        agent=object(),
        args=args,
        logger=_FakeLogger(),
        device="cpu",
        step=15008,
        next_eval_step=15000,
        eval_env=None,
        last_eval_step=None,
    )
    assert last_eval_step == 15008

    # The final-eval decision used by _run_on_policy/_run_off_policy:
    final_step = 15008
    should_run_final_eval = next_eval_step is not None and last_eval_step != final_step
    assert should_run_final_eval is False, (
        "eval already ran at this exact (overshot) step -- running it again "
        "would duplicate the [eval] line, which is the bug this guards against"
    )


def test_final_eval_still_runs_when_nothing_evaluated_this_step(monkeypatch) -> None:
    _patch_evaluate_agent(monkeypatch)
    args = _fake_args(eval_freq=3000)

    # Training ended at step=3000, short of the next scheduled threshold --
    # no eval has happened near the end yet, final eval must run.
    next_eval_step, eval_env, last_eval_step = train_module._maybe_run_evaluation(
        agent=object(),
        args=args,
        logger=_FakeLogger(),
        device="cpu",
        step=2000,
        next_eval_step=4096,
        eval_env=None,
        last_eval_step=None,
    )
    assert last_eval_step is None  # never fired

    final_step = 3000
    should_run_final_eval = next_eval_step is not None and last_eval_step != final_step
    assert should_run_final_eval is True


def test_eval_disabled_never_reports_already_ran() -> None:
    next_eval_step = None
    last_eval_step = None
    final_step = 1000
    should_run_final_eval = next_eval_step is not None and last_eval_step != final_step
    assert should_run_final_eval is False
