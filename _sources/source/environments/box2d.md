# Gymnasium Box2D Environments

Continuous Box2D tasks are useful mid-complexity benchmarks between classic
control and MuJoCo.

## Environments

- BipedalWalker-v3 (obs 24, act 4)
- LunarLanderContinuous-v3 (obs 8, act 2)
- CarRacing-v3 (obs 96x96x3, act 3)

## Install dependencies

In the tests Python 3.11 environment, these packages were required:
- box2d
- pygame

## Training

Use [examples/envs/train_box2d.py](https://github.com/Bigkatoan/SRL/blob/main/examples/envs/train_box2d.py):

- python examples/envs/train_box2d.py --env BipedalWalker-v3
- python examples/envs/train_box2d.py --env LunarLanderContinuous-v3
- python examples/envs/train_box2d.py --env CarRacing-v3

Configs:
- [configs/envs/bipedal_walker_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/bipedal_walker_ppo.yaml)
- [configs/envs/lunar_lander_continuous_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/lunar_lander_continuous_ppo.yaml)
- [configs/envs/car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml)
