"""Launcher for SRL's ROS 2 inference node.

Parameters (see launch/inference.launch.py):

* ``policy_path`` -- an ``agent.save()`` directory or an ``srl-train``
  checkpoint directory (``checkpoints/<run>``); anything ``srl.load_policy``
  accepts.
* ``checkpoint`` -- ``auto`` (best, else final, else latest), ``best``,
  ``final`` or ``latest`` when ``policy_path`` is a checkpoint directory.
* ``config_path`` -- optional YAML with a ``ros2:`` block that overrides the
  topics stored in the policy's ``config.yaml``.
* ``device``, ``hz``.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node

from srl.ros2.rl_node import RLInferenceNode


class SRLInferenceLauncher(Node):
    def __init__(self) -> None:
        super().__init__("srl_inference_launcher")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("checkpoint", "auto")
        self.declare_parameter("config_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("hz", 50.0)

    def build_runtime_node(self) -> RLInferenceNode:
        policy_path = Path(str(self.get_parameter("policy_path").value)).expanduser()
        config_path = str(self.get_parameter("config_path").value) or None
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        if config_path is not None and not Path(config_path).expanduser().exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return RLInferenceNode.from_policy_path(
            str(policy_path),
            device=str(self.get_parameter("device").value),
            checkpoint=str(self.get_parameter("checkpoint").value),
            config_path=str(Path(config_path).expanduser()) if config_path else None,
            hz=float(self.get_parameter("hz").value),
        )


def main() -> None:
    rclpy.init()
    launcher = SRLInferenceLauncher()
    runtime_node = launcher.build_runtime_node()
    launcher.destroy_node()
    try:
        rclpy.spin(runtime_node)
    finally:
        runtime_node.destroy_node()
        rclpy.shutdown()
