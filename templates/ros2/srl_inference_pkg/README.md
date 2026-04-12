# SRL ROS 2 Starter Package

This directory is a starter `ament_python` package for deploying an SRL policy as a ROS 2 node.

It is a template, not an installed part of the SRL Python package.

## What it does

- exposes one ROS 2 executable: `srl_inference_node`
- loads an SRL YAML config and checkpoint from ROS parameters
- creates `RLInferenceNode` from SRL's Python API
- gives you a concrete `ament` package path to customize instead of starting from a raw script

## Intended workflow

1. Copy this directory into a ROS 2 workspace under `src/`
2. Adjust package name, topics, and parameter defaults if needed
3. Build with `colcon build`
4. Launch with your chosen YAML config and checkpoint paths

## Build example

```bash
cd ~/ros2_ws
colcon build --packages-select srl_inference_pkg
source install/setup.bash
```

## Launch example

```bash
ros2 launch srl_inference_pkg inference.launch.py \
  config_path:=/absolute/path/to/configs/sac_multimodal.yaml \
  checkpoint_path:=/absolute/path/to/checkpoint.pt \
  device:=cpu \
  hz:=50.0
```

## Important constraints

- The active ROS 2 Python environment must also be able to import `srl`, `torch`, and any model dependencies.
- Message types and queue sizes still come from the SRL YAML `ros2` block unless you customize the node further.
- This package does not remove the need to align observation topic names and model encoder routing.