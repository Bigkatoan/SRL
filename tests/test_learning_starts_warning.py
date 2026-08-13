"""Regression test: warn when --steps won't reach learning_starts/update_after.

Off-policy algorithms only take their first gradient step once at least
`update_after` (falling back to `learning_starts`) env steps have been
collected. A smoke-test run with a small --steps below that threshold
completes "successfully" with zero gradient updates and byte-identical
eval scores across every checkpoint -- confirmed via a real run on
fetch_reach_sac.yaml at --steps 4000 against its learning_starts: 10_000 --
easy to misdiagnose as a training bug rather than an unmet warmup budget.
"""

from __future__ import annotations

from types import SimpleNamespace

from srl.cli.train import _warn_if_no_updates_will_run


def _fake_agent(**cfg_fields):
    return SimpleNamespace(cfg=SimpleNamespace(**cfg_fields))


def test_warns_when_steps_budget_is_below_learning_starts(capsys) -> None:
    agent = _fake_agent(learning_starts=10_000, update_after=None)
    args = SimpleNamespace(steps=4000)

    _warn_if_no_updates_will_run(agent, args, start_step=0)

    err = capsys.readouterr().err
    assert "learning_starts is 10000" in err
    assert "no gradient update will run" in err


def test_no_warning_when_update_after_is_explicitly_lower(capsys) -> None:
    # update_after is checked first and overrides learning_starts when set --
    # a real shipped config (halfcheetah_sac.yaml) does exactly this.
    agent = _fake_agent(learning_starts=10_000, update_after=1_000)
    args = SimpleNamespace(steps=1024)

    _warn_if_no_updates_will_run(agent, args, start_step=0)

    assert capsys.readouterr().err == ""


def test_no_warning_when_steps_budget_is_sufficient(capsys) -> None:
    agent = _fake_agent(learning_starts=10_000, update_after=None)
    args = SimpleNamespace(steps=50_000)

    _warn_if_no_updates_will_run(agent, args, start_step=0)

    assert capsys.readouterr().err == ""


def test_accounts_for_resume_start_step(capsys) -> None:
    # Resuming from step 8000 with --steps 9000 leaves only 1000 remaining
    # env steps -- still short of learning_starts=10_000 from a fresh start,
    # but the check must be against *remaining* budget, not raw --steps.
    agent = _fake_agent(learning_starts=10_000, update_after=None)
    args = SimpleNamespace(steps=9_000)

    _warn_if_no_updates_will_run(agent, args, start_step=8_000)

    assert "learning_starts is 10000" in capsys.readouterr().err


def test_no_warning_when_config_has_neither_field() -> None:
    # e.g. a config class without learning_starts/update_after at all --
    # must not crash, must not warn.
    agent = _fake_agent()
    args = SimpleNamespace(steps=10)

    _warn_if_no_updates_will_run(agent, args, start_step=0)  # no exception
