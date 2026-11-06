# SRL ROS 2 Starter Package

This directory is a starter `ament_python` package for deploying an SRL policy as a ROS 2 node.

It is a template, not an installed part of the SRL Python package.

## What it does

- exposes one ROS 2 executable: `srl_inference_node`
- loads a trained policy with `srl.load_policy(policy_path)` -- an `agent.save()`
  directory or an `srl-train` checkpoint directory (format v1; legacy `.pt`
  checkpoints also load)
- takes the observation/action topics from the `ros2:` block of the policy's
  saved `config.yaml` (or from an optional `config_path` override)
- creates `RLInferenceNode` from SRL's Python API
- gives you a concrete `ament` package path to customize instead of starting from a raw script

## Intended workflow

1. Copy this directory into a ROS 2 workspace under `src/`
2. Adjust package name, topics, and parameter defaults if needed
3. Build with `colcon build`
4. Launch with the path of your trained policy

## Build example

```bash
cd ~/ros2_ws
colcon build --packages-select srl_inference_pkg
source install/setup.bash
```

## Launch example

```bash
ros2 launch srl_inference_pkg inference.launch.py \
  policy_path:=/absolute/path/to/checkpoints/sac_sac_multimodal \
  checkpoint:=best \
  device:=cpu \
  hz:=50.0
# optional: config_path:=/absolute/path/to/topics.yaml  (a YAML with a ros2: block)
```

## Important constraints

- The active ROS 2 Python environment must also be able to import `srl`, `torch`, and any model dependencies.
- Topics, message types and queue sizes come from the `ros2` block of the training config (saved in the policy's `config.yaml`) unless `config_path` overrides them.
- This package does not remove the need to align observation topic names and model encoder routing.