"""srl.algorithms — PPO, SAC, DDPG, TD3, A2C, A3C."""

from srl.algorithms.a2c import A2C
from srl.algorithms.a3c import A3C
from srl.algorithms.ddpg import DDPG
from srl.algorithms.ppo import PPO
from srl.algorithms.sac import SAC
from srl.algorithms.td3 import TD3

__all__ = ["PPO", "SAC", "DDPG", "TD3", "A2C", "A3C"]
