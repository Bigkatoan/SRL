Welcome to SRL!
================

**SRL** is a modular reinforcement learning library for continuous action-space
environments: PPO, SAC, DDPG, TD3, A2C, and A3C, built around a YAML-first model
system where encoders, heads, flows, and multi-modal observation routing are all
declared as data rather than hand-wired Python.

.. note::

   ``srl-rl`` is not published on PyPI yet. Install from GitHub — see
   :doc:`source/installation`.

**Key features:**

- **YAML-first model system:** declare encoders, heads, flows, and multi-modal
  routing as data, shared across training, evaluation, and ROS 2 inference
- **Multi-modal inputs:** state, pixels, lidar, text — all in one model
- **Vectorised training:** ``SyncVectorEnv`` and ``AsyncVectorEnv`` for fast data
  collection, plus an async off-policy runner for GPU-batched sims
- **Goal-conditioned RL:** ``GoalEnvWrapper`` for gymnasium-robotics Fetch tasks
- **Isaac Lab and mjlab support:** headless training and evaluation for PPO, A2C,
  SAC, DDPG, and TD3 against GPU-batched simulators
- **ROS 2 Python API:** integrate trained agents into your own ROS 2 code
- **Live training viewer:** ``srl-train --visualize`` runs one extra env doing live
  inference with the current model, rendered while training continues

**Try it now:**

.. code-block:: bash

   git clone https://github.com/Bigkatoan/SRL.git && cd SRL
   pip install -e .
   srl-train --config configs/envs/pendulum_ppo.yaml --env Pendulum-v1 --algo ppo --steps 100000

Five-minute example
--------------------

Training is driven by the ``srl-train`` CLI; the Python side loads the
resulting checkpoint for inference. Run from a clone of the repository so the
``configs/`` directory is available:

.. code-block:: bash

   srl-train --config configs/envs/pendulum_ppo.yaml --steps 100000 --device auto

.. code-block:: python

   import gymnasium as gym
   import torch

   from srl.algorithms.ppo import PPO
   from srl.envs.gymnasium_wrapper import GymnasiumWrapper
   from srl.registry.builder import ModelBuilder
   from srl.utils.checkpoint import CheckpointManager

   device = "cuda" if torch.cuda.is_available() else "cpu"

   model = ModelBuilder.from_yaml("configs/envs/pendulum_ppo.yaml")
   agent = PPO(model, device=device)
   cm = CheckpointManager("checkpoints/ppo_pendulum_ppo")
   cm.load(agent.model, sorted(cm.save_dir.glob("final_*.pt"))[-1], device=device)

   env = GymnasiumWrapper(gym.make("Pendulum-v1"))
   obs, _ = env.reset(seed=0)
   done, episode_return = False, 0.0
   while not done:
       obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items()}
       action, *_ = agent.predict(obs_t, deterministic=True)
       obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
       episode_return += float(reward)
       done = terminated or truncated
   print(f"episode return: {episode_return:.1f}")

If you want to understand SRL properly, start from the YAML layer rather than the
algorithm pages: :doc:`source/yaml_core` explains the declarative model graph and
observation routing, :doc:`source/config_reference` lists the schema fields in
detail, and :doc:`source/quickstart` shows how the same YAML file is consumed by
the CLI and the Python API.

Table of Contents
------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/installation
   source/quickstart
   source/cli

.. toctree::
   :maxdepth: 1
   :caption: YAML Configuration

   source/yaml_core
   source/yaml/index
   source/yaml/encoders
   source/yaml/heads_flows
   source/yaml/auxiliary
   source/yaml/training_block

.. toctree::
   :maxdepth: 1
   :caption: Training System

   source/training/index
   source/algorithms
   source/training/runners
   source/training/buffers
   source/checkpointing
   source/encoders
   source/async_runner
   source/gpu_replay_buffer

.. toctree::
   :maxdepth: 1
   :caption: Integrations

   source/integrations/index
   source/integrations/isaaclab
   source/integrations/mjlab
   source/integrations/gymnasium
   source/ros2

.. toctree::
   :maxdepth: 1
   :caption: Environments

   source/environments/index
   source/environments/classic_control
   source/environments/box2d
   source/environments/mujoco
   source/environments/robotics
   source/environments/racecar
   source/environments/isaaclab
   source/environments/mjlab
   source/environments/m3bot

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   source/config_reference
   source/limitations
   source/examples/encoder_examples
   source/contributing

License
-------

SRL is licensed under the MIT License. See the
`LICENSE file <https://github.com/Bigkatoan/SRL/blob/main/LICENSE>`_ for details.
