"""srl.algorithms — PPO, SAC, DDPG, TD3, A2C, A3C.

:data:`ALGORITHMS` maps each registry name (the YAML ``algo:`` key /
``srl-train --algo``) to ``(agent_class, config_class)``.
"""

from __future__ import annotations

from srl.algorithms.a2c import A2C
from srl.algorithms.a3c import A3C
from srl.algorithms.ddpg import DDPG
from srl.algorithms.ppo import PPO
from srl.algorithms.sac import SAC
from srl.algorithms.td3 import TD3

__all__ = ["ALGORITHMS", "A2C", "A3C", "DDPG", "PPO", "SAC", "TD3", "get_algorithm"]

#: ``name -> (agent class, default config class)``.
ALGORITHMS: dict[str, tuple[type, type]] = {
    cls.algo_name: (cls, cls.config_cls) for cls in (PPO, A2C, A3C, SAC, DDPG, TD3)
}


def get_algorithm(name: str) -> tuple[type, type]:
    """Look up ``(agent_class, config_class)`` by registry name (case-insensitive)."""
    key = str(name).strip().lower()
    try:
        return ALGORITHMS[key]
    except KeyError:
        raise KeyError(
            f"Unknown algorithm {name!r}; expected one of {sorted(ALGORITHMS)}"
        ) from None
