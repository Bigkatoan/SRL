"""Train PPO on Isaac Lab tasks (run inside an Isaac Lab Python env).

Usage: python examples/envs/train_isaaclab.py --env Isaac-Cartpole-v0"""

import argparse
from pathlib import Path

import yaml

from srl import BaseAgent
from srl.training.envs import bootstrap_isaac_sim

CONFIGS = {
    "Isaac-Cartpole-v0": "configs/envs/isaaclab_cartpole_ppo.yaml",
    "Isaac-Ant-v0": "configs/envs/isaaclab_ant_ppo.yaml",
    "Isaac-Humanoid-v0": "configs/envs/isaaclab_humanoid_ppo.yaml",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="Isaac-Cartpole-v0", choices=sorted(CONFIGS))
    p.add_argument("--steps", type=int, default=None)  # default: train.total_steps
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    bootstrap_isaac_sim()  # must run before any isaaclab/omni import
    raw = yaml.safe_load(Path(CONFIGS[args.env]).read_text())
    train = raw.get("train", {})
    agent = BaseAgent.from_config(raw, device=args.device)  # env built from env_id/n_envs
    agent.learn(
        args.steps or int(train.get("total_steps", 1_000_000)),
        log_dir=f"runs/{args.env}",
        progress=True,
    )
    agent.save(f"runs/{args.env}/final")


if __name__ == "__main__":
    main()
