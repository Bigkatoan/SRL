# Limitations

SRL already has a strong YAML-first model system, but it is not yet a fully declarative end-to-end RL framework.

This page exists to make the current system boundary explicit.

## Currently unsupported or incomplete

### Discrete-action algorithms

The current library targets continuous-action control. The built-in actor heads and algorithm docs are centered on PPO, SAC, DDPG, TD3, A2C, and A3C in continuous settings.

### Fully declarative training orchestration

YAML currently defines model structure and part of training configuration, but it does not yet define the whole training loop as a first-class graph.

### Generic reward shaping pipelines in YAML

There is no schema-supported reward wrapper or reward transformation stack yet.

### Generic environment wrapper stacks in YAML

Environment creation is still handled procedurally by the CLI and wrapper classes.

### Arbitrary custom loss registration from YAML

Built-in losses can be selected, but there is not yet a general registry that lets users declare any custom loss purely from YAML.

### Multi-agent RL

The current package is not documented or structured as a multi-agent RL framework.

### ROS 2 as a full package

SRL currently exposes a ROS 2 Python API, not a full `ament` ROS 2 package with launch files and install targets.

### ROS 2 deployment parity

ROS 2 deployment exists, but the current inference node still needs stronger alignment with the observation remapping semantics used in training.

## What this means in practice

The right mental model is:

- YAML is the core abstraction for model graph definition.
- CLI covers the common operational workflows.
- Python is still required for some advanced extension points.
- Isaac Lab and ROS 2 are important targets, but they still have active engineering gaps.

If you are using SRL for research, treat it as a capable alpha-stage framework with a strong declarative model layer rather than a finished no-code training platform.