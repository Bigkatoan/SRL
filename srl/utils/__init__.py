"""srl.utils — logger, normalizer, GAE, callbacks, checkpoint."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Logger",
    "LoggerConfig",
    "RunningNormalizer",
    "compute_gae",
    "CheckpointManager",
    "BaseCallback",
    "LogCallback",
    "CheckpointCallback",
    "EarlyStopping",
]

_EXPORTS = {
    "Logger": ("srl.utils.logger", "Logger"),
    "LoggerConfig": ("srl.utils.logger", "LoggerConfig"),
    "RunningNormalizer": ("srl.utils.normalizer", "RunningNormalizer"),
    "compute_gae": ("srl.utils.gae", "compute_gae"),
    "CheckpointManager": ("srl.utils.checkpoint", "CheckpointManager"),
    "BaseCallback": ("srl.utils.callbacks", "BaseCallback"),
    "LogCallback": ("srl.utils.callbacks", "LogCallback"),
    "CheckpointCallback": ("srl.utils.callbacks", "CheckpointCallback"),
    "EarlyStopping": ("srl.utils.callbacks", "EarlyStopping"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'srl.utils' has no attribute '{name}'")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
