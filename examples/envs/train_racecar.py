"""Train PPO on racecar_gym (env_type: racecar).

Usage: pip install racecar_gym; python examples/envs/train_racecar.py"""

import argparse
from pathlib import Path

import yaml

from srl import BaseAgent

CONFIGS = {
    "SingleAgentAustria-v0": "configs/envs/racecar_austria_ppo.yaml",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="SingleAgentAustria-v0", choices=sorted(CONFIGS))
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
