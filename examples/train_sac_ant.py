"""Train SAC on Ant-v5 with the Python API.

Usage: python examples/train_sac_ant.py --steps 3000000 --device cuda"""

import argparse

import gymnasium as gym

from srl import SAC
from srl.training import CheckpointCallback


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    agent = SAC.from_config("configs/envs/ant_sac.yaml", env=gym.make("Ant-v5"), device=args.device)
    agent.learn(
        args.steps,
        eval_env=gym.make("Ant-v5"),
        eval_freq=50_000,
        callbacks=[CheckpointCallback("checkpoints/sac_ant", save_interval=100_000)],
        log_dir="runs/sac_ant",
        progress=True,
    )
    agent.save("runs/sac_ant/final")


if __name__ == "__main__":  # guard needed: vector envs spawn worker processes
    main()
