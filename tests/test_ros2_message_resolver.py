from types import ModuleType
import sys

import pytest

from srl.ros2.message_resolver import resolve_msg_type


def test_resolve_msg_type_uses_default_when_none() -> None:
    default = object()
    assert resolve_msg_type(None, default) is default


def test_resolve_msg_type_with_dotted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("fake_msgs")

    class FakeMessage:
        pass

    module.FakeMessage = FakeMessage
    monkeypatch.setitem(sys.modules, "fake_msgs", module)

    assert resolve_msg_type("fake_msgs.FakeMessage") is FakeMessage


def test_resolve_msg_type_rejects_unknown_short_name() -> None:
    with pytest.raises(ValueError):
        resolve_msg_type("UnknownMessage")