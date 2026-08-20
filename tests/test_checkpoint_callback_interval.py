"""Regression test for CheckpointCallback's step-interval trigger.

Motivating bug: a real 120M-step PPO run against JAVIS's mjlab balance task
(n_steps=24, n_envs=4096, so `step` advances 98304 at a time) wrote ZERO
periodic checkpoints across the whole run. `CheckpointCallback.on_step_end`
triggered on `step % save_interval == 0` -- exact equality -- and with the
default `save_interval=100_000`, `gcd(98304, 100_000) == 32`, so the first
step that is also an exact multiple of 100_000 is
`lcm(98304, 100_000) == 307,200,000`, past the end of every run this project
has actually run. Any rollout granularity without a large common factor
with `save_interval` hits the same silent no-op -- `on_step_end` never once
returns True's worth of a save, with no error or warning anywhere.

Fixed to trigger on CROSSING a multiple of `save_interval` instead of
landing exactly on one.
"""

from __future__ import annotations

from srl.utils.callbacks import CheckpointCallback


class _FakeCheckpointManager:
    def __init__(self) -> None:
        self.saved_steps: list[int] = []

    def save(self, model, optimizer, step, metrics) -> None:
        self.saved_steps.append(step)


def test_default_interval_never_fires_with_98304_step_increments_before_fix_demo():
    """Sanity check on the bug's own arithmetic (not the class): confirms
    98304 (n_steps=24 * n_envs=4096, this project's real PPO config) never
    lands on an exact multiple of the default save_interval=100_000 within
    a 120,000,000-step budget -- i.e. this is a real, not hypothetical,
    trigger gap for the exact granularity that motivated the fix.
    """
    save_interval = 100_000
    step = 0
    hit_exact_multiple = False
    while step <= 120_000_000:
        step += 98304
        if step % save_interval == 0:
            hit_exact_multiple = True
            break
    assert not hit_exact_multiple


def test_checkpoint_saves_despite_step_increments_that_never_hit_exact_multiples():
    cm = _FakeCheckpointManager()
    cb = CheckpointCallback(cm, save_interval=100_000, model=object())

    step = 0
    for _ in range(1220):  # ~120M steps worth of 98304-sized increments
        step += 98304
        cb.on_step_end(step, {})

    assert len(cm.saved_steps) > 0, "no periodic checkpoint was ever saved"
    # Roughly one save per 100_000 steps of the ~120M-step run.
    assert 1100 <= len(cm.saved_steps) <= 1300


def test_saves_exactly_once_per_crossed_interval_not_once_per_call():
    cm = _FakeCheckpointManager()
    cb = CheckpointCallback(cm, save_interval=1000, model=object())

    for step in (500, 999, 1000, 1001, 1500, 1999, 2000, 2500):
        cb.on_step_end(step, {})

    # Crossed 1000 once (at step=1000, the first call >= 1000) and 2000 once
    # (at step=2000) -- not re-triggered by every subsequent call before the
    # next boundary.
    assert cm.saved_steps == [1000, 2000]


def test_a_call_that_jumps_past_multiple_intervals_still_saves_and_reschedules():
    """A single on_step_end call whose step jumped past more than one
    save_interval boundary at once (e.g. a very large rollout granularity
    relative to save_interval) must still save (once) and correctly
    reschedule the NEXT boundary above the new step -- not get stuck
    re-triggering on every subsequent call, and not silently skip the
    interval(s) it jumped over.
    """
    cm = _FakeCheckpointManager()
    cb = CheckpointCallback(cm, save_interval=100, model=object())

    cb.on_step_end(350, {})  # jumps straight past the 100/200/300 boundaries
    assert cm.saved_steps == [350]
    assert cb._next_save_step == 400

    cb.on_step_end(399, {})  # still before the next boundary
    assert cm.saved_steps == [350]

    cb.on_step_end(400, {})
    assert cm.saved_steps == [350, 400]


def test_no_save_before_model_is_bound():
    cm = _FakeCheckpointManager()
    cb = CheckpointCallback(cm, save_interval=100)  # model=None
    cb.on_step_end(100, {})
    assert cm.saved_steps == []

    cb.bind(model=object())
    cb.on_step_end(200, {})
    assert cm.saved_steps == [200]
