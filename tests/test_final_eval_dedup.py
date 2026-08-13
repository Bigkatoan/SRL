"""Regression tests for the duplicate final-eval bug.

`_run_on_policy` and `_run_off_policy` each run a "final eval" after the
training loop ends, unless a periodic eval already happened at exactly the
final step. The guard used to OR in `step < next_eval_step`, which is true
immediately after *any* eval fires (by construction of
`_maybe_run_evaluation`'s return value) -- so it always overrode the one
check that correctly detected "already evaluated here," producing a
duplicate eval (and duplicate `[eval]` log line) whenever `total_steps` was
an exact multiple of `eval_freq`.
"""

from __future__ import annotations

from srl.cli.train import _final_eval_already_ran_at


def test_detects_eval_that_just_ran_at_this_exact_step() -> None:
    # _maybe_run_evaluation just fired at step=4096 and advanced the
    # threshold to 4096 + 2048 -- a final eval here would be a duplicate.
    assert _final_eval_already_ran_at(next_eval_step=6144, eval_freq=2048, step=4096) is True


def test_does_not_flag_a_step_that_was_never_evaluated() -> None:
    # Training ended at step=3000, short of the next scheduled threshold
    # (4096) -- no eval has happened near the end yet, final eval must run.
    assert _final_eval_already_ran_at(next_eval_step=4096, eval_freq=2048, step=3000) is False


def test_eval_disabled_never_reports_already_ran() -> None:
    assert _final_eval_already_ran_at(next_eval_step=None, eval_freq=0, step=1000) is False


def test_previous_buggy_guard_would_have_duplicated_here() -> None:
    """Pins down the exact bug: the old `step < next_eval_step or
    next_eval_step - eval_freq != step` guard evaluates True (run again) in
    precisely the case this fix must evaluate False (already ran)."""
    next_eval_step, eval_freq, step = 6144, 2048, 4096

    old_buggy_guard = step < next_eval_step or next_eval_step - eval_freq != step
    assert old_buggy_guard is True  # would have re-run eval -- the bug

    assert _final_eval_already_ran_at(next_eval_step, eval_freq, step) is True  # correctly skips
