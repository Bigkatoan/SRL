"""srl.algorithms — PPO, SAC, DDPG, TD3, A2C, A3C.

Algorithms are registered in :data:`srl.registry.AlgorithmRegistry` (built-ins
and plugins alike, see :mod:`srl.registry.builtins`); that registry is the one
source of truth for ``algo:`` / ``srl-train --algo`` dispatch.
:data:`ALGORITHMS` is a read-only compatibility view of it mapping each name
to ``(agent_class, config_class)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from srl.algorithms.a2c import A2C
from srl.algorithms.a3c import A3C
from srl.algorithms.ddpg import DDPG
from srl.algorithms.ppo import PPO
from srl.algorithms.sac import SAC
from srl.algorithms.td3 import TD3

__all__ = ["ALGORITHMS", "A2C", "A3C", "DDPG", "PPO", "SAC", "TD3", "get_algorithm"]


class _AlgorithmsView(Mapping[str, tuple[type, type]]):
    """``name -> (agent class, default config class)``, read live from the registry."""

    def __getitem__(self, name: str) -> tuple[type, type]:
        from srl.registry.registry import AlgorithmRegistry

        key = str(name).strip().lower()
        if key not in AlgorithmRegistry:
            raise KeyError(name)
        spec = AlgorithmRegistry.get(key)
        return spec.agent_class(), spec.config_cls

    def __iter__(self) -> Iterator[str]:
        from srl.registry.registry import AlgorithmRegistry

        return iter(AlgorithmRegistry.available())

    def __len__(self) -> int:
        from srl.registry.registry import AlgorithmRegistry

        return len(AlgorithmRegistry.available())

    def __repr__(self) -> str:
        return f"ALGORITHMS({sorted(self)})"


#: ``name -> (agent class, default config class)`` -- a live, read-only view of
#: :data:`srl.registry.AlgorithmRegistry` (kept for backwards compatibility).
ALGORITHMS: Mapping[str, tuple[type, type]] = _AlgorithmsView()


def get_algorithm(name: str) -> tuple[type, type]:
    """Look up ``(agent_class, config_class)`` by registry name (case-insensitive)."""
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise KeyError(
            f"Unknown algorithm {name!r}; expected one of {sorted(ALGORITHMS)}"
        ) from None
