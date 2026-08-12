# racecar_gym Environments

SRL includes support for racecar_gym via a dedicated wrapper.

Known caveat: upstream racecar_gym currently has Python 3.11 compatibility
issues in some versions. If import fails, use Python 3.10 or patch racecar_gym.

## Supported tracks

- SingleAgentAustria-v0
- SingleAgentBerlin-v0
- SingleAgentMontreal-v0
- SingleAgentTorino-v0
- SingleAgentCircle-v0
- SingleAgentPlechaty-v0

## Wrapper

See source file:
- https://github.com/Bigkatoan/SRL/blob/main/srl/envs/racecar_wrapper.py

The wrapper:
- flattens Dict observations into one state vector
- converts a flat 2D action vector into racecar_gym Dict action format

## Training

Use training script:
- https://github.com/Bigkatoan/SRL/blob/main/examples/envs/train_racecar.py

- python examples/envs/train_racecar.py --env SingleAgentAustria-v0
- python examples/envs/train_racecar.py --env SingleAgentBerlin-v0

Base config:
- https://github.com/Bigkatoan/SRL/blob/main/configs/envs/racecar_austria_ppo.yaml
