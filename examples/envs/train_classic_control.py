"""Train PPO on classic-control envs (Pendulum-v1, MountainCarContinuous-v0).

Usage: python examples/envs/train_classic_control.py --env Pendulum-v1"""

import argparse
from pathlib import Path

import yaml

from srl import BaseAgent

CONFIGS = {
    "Pendulum-v1": "configs/envs/pendulum_ppo.yaml",
    "MountainCarContinuous-v0": "configs/envs/mountain_car_continuous_ppo.yaml",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1", choices=sorted(CONFIGS))
    p.add_argument("--steps", type=int, default=None)  # default: train.total_steps
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    raw = yaml.safe_load(Path(CONFIGS[args.env]).read_text())
    train = raw.get("train", {})
    agent = BaseAgent.from_config(raw, device=args.device)  # env built from env_id/n_envs
    agent.learn(
        args.steps or int(train.get("total_steps", 1_000_000)),
        eval_env=raw["env_id"],
        eval_freq=int(train.get("eval_freq", 50_000)),
        log_dir=f"runs/{args.env}",
        progress=True,
    )
    agent.save(f"runs/{args.env}/final")


if __name__ == "__main__":  # guard needed: vector envs spawn worker processes
    main()
