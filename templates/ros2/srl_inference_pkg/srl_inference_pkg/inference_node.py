from __future__ import annotations

from pathlib import Path

import rclpy
import torch
import yaml
from rclpy.node import Node

from srl.registry.builder import ModelBuilder
from srl.ros2.rl_node import RLInferenceNode


class SRLInferenceLauncher(Node):
    def __init__(self) -> None:
        super().__init__("srl_inference_launcher")
        self.declare_parameter("config_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("hz", 50.0)

    def build_runtime_node(self) -> RLInferenceNode:
        config_path = Path(str(self.get_parameter("config_path").value)).expanduser()
        checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value)).expanduser()
        device = str(self.get_parameter("device").value)
        hz = float(self.get_parameter("hz").value)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model = ModelBuilder.from_yaml(config_path)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"])
        else:
            model.load_state_dict(payload)

        ros2_cfg = config_data.get("ros2") or {}
        obs_topics = ros2_cfg.get("observations") or ros2_cfg.get("obs_topics") or {}
        action_topic = ros2_cfg.get("action_topic", "/actions")
        return RLInferenceNode(
            model=model,
            obs_topics=obs_topics,
            action_topic=action_topic,
            hz=hz,
            device=device,
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