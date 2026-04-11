"""ROS2 inference node for SRL policies (optional).

Provides a Python API for deploying trained SRL policies as ROS2 nodes.
Requires rclpy (part of ROS2 install) to use.

Example usage::

    from srl.ros2.rl_node import RLInferenceNode
    from srl.registry.builder import ModelBuilder
    import rclpy

    model = ModelBuilder.from_yaml("my_config.yaml")
    model.load_state_dict(...)

    rclpy.init()
    node = RLInferenceNode(
        model=model,
        obs_topics={
            "state_enc": "/sensor/state",
        },
        action_topic="/robot/cmd_vel",
        hz=20.0,
    )
    node.run()
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from srl.registry.config_schema import AgentModelConfig, ROS2Config
from srl.utils.obs_remap import apply_obs_remap

# -------------------------------------------------------------------------
# Guard: rclpy is an optional dependency
# -------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

    class Node:  # type: ignore[no-redef]
        """Stub when rclpy is not available."""
        def __init__(self, *args, **kwargs): pass


class RLInferenceNode(Node):
    """ROS2 lifecycle-aware inference node.

    Parameters
    ----------
    model:
        A trained :class:`~srl.networks.agent_model.AgentModel`.
    obs_topics:
        Mapping from encoder name → ROS2 topic string.
    action_topic:
        Topic on which to publish actions.
    action_msg_type:
        ROS2 message type for the action topic (e.g. ``Float32MultiArray``).
    obs_msg_type:
        ROS2 message type for observation topics.
    hz:
        Inference frequency (Hz).
    device:
        PyTorch device.
    """

    def __init__(
        self,
        model: Any,
        obs_topics: dict[str, str],
        action_topic: str = "/srl/action",
        action_msg_type: Any = None,
        obs_msg_type: Any = None,
        hz: float = 20.0,
        device: str = "cpu",
    ) -> None:
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy is not installed. Install ROS2 and source it before using RLInferenceNode."
            )

        super().__init__("rl_inference_node")

        import torch
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._encoder_names = list(getattr(self.model, "encoders", {}).keys())
        self._encoder_input_names = dict(getattr(self.model, "encoder_input_names", {}))

        self._hidden: dict[str, Any] = {}
        self._obs_buffers: dict[str, deque] = {
            name: deque(maxlen=1) for name in obs_topics
        }
        self._lock = threading.Lock()

        # Register subscribers
        for enc_name, topic in obs_topics.items():
            if obs_msg_type is not None:
                self.create_subscription(
                    obs_msg_type,
                    topic,
                    lambda msg, n=enc_name: self._obs_callback(n, msg),
                    10,
                )

        # Action publisher
        self._action_pub = None
        if action_msg_type is not None:
            self._action_pub = self.create_publisher(action_msg_type, action_topic, 10)

        # Main inference timer
        period = 1.0 / hz
        self.create_timer(period, self._timer_callback)

        self.get_logger().info(
            f"RLInferenceNode started @ {hz} Hz | device={device}"
        )

    def _obs_callback(self, enc_name: str, msg: Any) -> None:
        """Convert incoming ROS2 message to numpy and store."""
        arr = self._msg_to_numpy(msg)
        with self._lock:
            self._obs_buffers[enc_name].append(arr)

    def _timer_callback(self) -> None:
        """Run inference and publish action."""
        import torch

        with self._lock:
            obs_dict: dict[str, np.ndarray] = {}
            for name, buf in self._obs_buffers.items():
                if not buf:
                    return  # wait until all observations received
                obs_dict[name] = buf[-1]

        obs_dict = apply_obs_remap(
            obs_dict,
            self._encoder_names,
            self._encoder_input_names,
        )

        obs_t = {
            k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
            for k, v in obs_dict.items()
        }

        with torch.no_grad():
            action, self._hidden = self.model.act(
                obs_t, hidden_states=self._hidden, deterministic=True
            )

        action_np = action.squeeze(0).cpu().numpy()
        self._publish_action(action_np)

    def _publish_action(self, action: np.ndarray) -> None:
        if self._action_pub is None:
            return
        try:
            from std_msgs.msg import Float32MultiArray
            msg = Float32MultiArray()
            msg.data = action.tolist()
            self._action_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Publish failed: {e}")

    @staticmethod
    def _msg_to_numpy(msg: Any) -> np.ndarray:
        """Extract array data from a ROS2 message.

        Override this method in a subclass for custom message types.
        """
        # Float32MultiArray / sensor_msgs/Image
        if hasattr(msg, "data"):
            return np.array(msg.data, dtype=np.float32)
        raise ValueError(f"Cannot convert message of type {type(msg)} to numpy array.")


def main(args=None):
    """Entry point for ``ros2 run`` / ``console_scripts``."""
    if not _ROS2_AVAILABLE:
        raise RuntimeError("rclpy not found — cannot start ROS2 node.")

    import yaml
    import torch
    from srl.registry.builder import ModelBuilder

    rclpy.init(args=args)

    # Load config path from ROS2 parameter
    tmp_node = rclpy.create_node("srl_param_loader")
    tmp_node.declare_parameter("config_path", "")
    tmp_node.declare_parameter("checkpoint_path", "")
    tmp_node.declare_parameter("device", "cpu")
    tmp_node.declare_parameter("hz", 20.0)

    config_path = tmp_node.get_parameter("config_path").get_parameter_value().string_value
    ckpt_path = tmp_node.get_parameter("checkpoint_path").get_parameter_value().string_value
    device = tmp_node.get_parameter("device").get_parameter_value().string_value
    hz = tmp_node.get_parameter("hz").get_parameter_value().double_value
    tmp_node.destroy_node()

    if not config_path:
        raise ValueError("Must provide config_path parameter.")

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    config = AgentModelConfig.from_dict(cfg)

    model = ModelBuilder.from_dict(cfg)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))

    ros2_cfg: ROS2Config = config.ros2
    obs_topics = {name: spec.topic for name, spec in ros2_cfg.observations.items()}
    if not obs_topics:
        raise ValueError("ROS2 config must define ros2.observations or ros2.obs_topics.")
    action_topic = ros2_cfg.action_topic

    from std_msgs.msg import Float32MultiArray
    node = RLInferenceNode(
        model=model,
        obs_topics=obs_topics,
        action_topic=action_topic,
        action_msg_type=Float32MultiArray,
        obs_msg_type=Float32MultiArray,
        hz=hz,
        device=device,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
