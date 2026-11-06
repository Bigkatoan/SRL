"""Train PPO on HalfCheetah-v5 (8 vectorized envs, from the YAML) with the Python API.

Usage: python examples/train_ppo_halfcheetah.py --steps 1000000"""

import argparse

import gymnasium as gym

from srl import PPO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # env=None -> the YAML's env_id / train.n_envs build the (vectorized) training env.
    agent = PPO.from_config("configs/envs/halfcheetah_ppo.yaml", device=args.device)
    agent.learn(
        args.steps,
        eval_env="HalfCheetah-v5",
        eval_freq=50_000,
        log_dir="runs/ppo_halfcheetah",
        progress=True,
    )
    agent.save("runs/ppo_halfcheetah/final")

    # Roll out the trained policy: numpy obs in, env-unit actions out.
    env = gym.make("HalfCheetah-v5")
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":  # guard needed: vector envs spawn worker processes
    main()
