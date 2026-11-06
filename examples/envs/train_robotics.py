"""Train SAC on gymnasium-robotics Fetch tasks (env_type: goal; HER via the YAML).

Usage: pip install "srl-rl[robotics]"; python examples/envs/train_robotics.py --env FetchReach-v4"""

import argparse
from pathlib import Path

import yaml

from srl import BaseAgent

CONFIGS = {
    "FetchReach-v4": "configs/envs/fetch_reach_sac_her.yaml",
    "FetchPush-v4": "configs/envs/fetch_push_sac.yaml",
    "FetchPickAndPlace-v4": "configs/envs/fetch_pick_and_place_sac.yaml",
    "FetchSlide-v4": "configs/envs/fetch_slide_sac.yaml",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="FetchReach-v4", choices=sorted(CONFIGS))
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


if __name__ == "__main__":
    main()
