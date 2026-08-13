"""Regression test for the env_type: goal registration-warning spam.

`gymnasium_robotics.register_robotics_envs()` unconditionally re-registers
every Fetch/AntMaze/... variant on each call, and gymnasium.register() warns
loudly on every id that's already there. `_make_cli_env` runs once for the
training env and again for every eval cycle -- confirmed via a real run to
produce ~876 lines of output (mostly duplicate "Overriding environment X
already in registry" warnings) for a trivial 2000-step FetchReach-v4 run,
dropping to 8 after this fix.
"""

from __future__ import annotations

import srl.cli.train as train_module


def test_register_robotics_envs_once_only_calls_underlying_register_a_single_time(
    monkeypatch,
) -> None:
    monkeypatch.setattr(train_module, "_robotics_envs_registered", False)
    call_count = 0

    class _FakeGymnasiumRobotics:
        @staticmethod
        def register_robotics_envs():
            nonlocal call_count
            call_count += 1

    monkeypatch.setitem(__import__("sys").modules, "gymnasium_robotics", _FakeGymnasiumRobotics())

    train_module._register_robotics_envs_once()
    train_module._register_robotics_envs_once()
    train_module._register_robotics_envs_once()

    assert call_count == 1


def test_register_robotics_envs_once_suppresses_the_overriding_warning(capsys) -> None:
    import warnings

    import srl.cli.train as tm

    tm._robotics_envs_registered = False
    import gymnasium_robotics

    tm._register_robotics_envs_once()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gymnasium_robotics.register_robotics_envs()  # the raw, unguarded call
    assert any(
        "already in registry" in str(w.message) for w in caught
    ), "test setup assumption broke: the raw call should still warn"

    # ...but going through the guard a second time in the same process must
    # not re-emit it (or even attempt the call at all).
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        tm._register_robotics_envs_once()
    assert not any("already in registry" in str(w.message) for w in caught_again)
