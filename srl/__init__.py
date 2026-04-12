"""
SRL — Simple Reinforcement Learning Library
============================================
Continuous action spaces | PPO · SAC · DDPG · TD3 · A2C · A3C
Gymnasium & Isaac Lab compatible | Config-driven model builder | ROS2 deployable
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.0"
__all__ = [
	"PPO",
	"SAC",
	"DDPG",
	"TD3",
	"A2C",
	"A3C",
	"ModelBuilder",
	"BaseAgent",
	"CheckpointManager",
]

_EXPORTS = {
	"PPO": ("srl.algorithms.ppo", "PPO"),
	"SAC": ("srl.algorithms.sac", "SAC"),
	"DDPG": ("srl.algorithms.ddpg", "DDPG"),
	"TD3": ("srl.algorithms.td3", "TD3"),
	"A2C": ("srl.algorithms.a2c", "A2C"),
	"A3C": ("srl.algorithms.a3c", "A3C"),
	"ModelBuilder": ("srl.registry.builder", "ModelBuilder"),
	"BaseAgent": ("srl.core.base_agent", "BaseAgent"),
	"CheckpointManager": ("srl.utils.checkpoint", "CheckpointManager"),
}


def __getattr__(name: str) -> Any:
	if name not in _EXPORTS:
		raise AttributeError(f"module 'srl' has no attribute '{name}'")
	module_name, attr_name = _EXPORTS[name]
	module = import_module(module_name)
	value = getattr(module, attr_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted(list(globals().keys()) + __all__)
