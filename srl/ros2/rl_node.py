"""ROS 2 inference node for SRL policies (optional; needs ``rclpy``).

The node is a thin ROS shell around :class:`PolicyRunner`, a pure-Python
object holding the latest observation per input and running the policy
(:func:`srl.load_policy`) on them. Everything except subscribing /
publishing is in :class:`PolicyRunner`, so it is testable without ROS.

Example::

    import rclpy
    from srl.ros2.rl_node import RLInferenceNode

    rclpy.init()
    node = RLInferenceNode.from_policy_path(
        "checkpoints/sac_pendulum_sac",        # any srl.load_policy() path
        hz=50.0,
    )                                           # topics from the policy's config.yaml ros2: block
    rclpy.spin(node)

or ``ros2 run`` / the template package with the ``policy_path`` parameter
(see ``templates/ros2/srl_inference_pkg``).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from srl.registry.config_schema import ROS2Config
from srl.ros2.message_resolver import resolve_msg_type

__all__ = ["PolicyRunner", "RLInferenceNode", "main", "msg_to_numpy", "ros2_config_for"]

# -------------------------------------------------------------------------
# Guard: rclpy is an optional dependency
# -------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

    class Node:  # type: ignore[no-redef]
        """Stub when rclpy is not available."""

        def __init__(self, *args, **kwargs):
            pass


def msg_to_numpy(msg: Any) -> np.ndarray:
    """Array data of a ROS 2 message (``Float32MultiArray`` / ``Image`` / ``JointState``)."""
    if hasattr(msg, "data"):
        data = msg.data
        if isinstance(data, (bytes, bytearray)):
            return np.frombuffer(bytes(data), dtype=np.uint8).astype(np.float32)
        return np.asarray(data, dtype=np.float32)
    if hasattr(msg, "position"):  # sensor_msgs/JointState
        return np.asarray(
            list(msg.position) + list(getattr(msg, "velocity", []) or []), dtype=np.float32
        )
    raise ValueError(f"Cannot convert message of type {type(msg)} to numpy array.")


class PolicyRunner:
    """The ROS-free core of the inference node.

    Holds the latest observation per input name, and on :meth:`step` runs
    ``policy.predict(obs_dict)`` once every input has been received.
    Flat arrays (as carried by ``Float32MultiArray`` / ``Image``) are
    reshaped to the encoder's input shape from the policy's ``obs_spec``
    when the sizes match.

    Parameters
    ----------
    policy:
        A :class:`srl.policy.Policy` (or anything with ``predict(obs) -> action``).
    input_names:
        Observation keys to wait for (encoder names or env obs keys); usually
        the keys of the ``ros2.observations`` config block.
    deterministic:
        Passed to ``policy.predict``.
    """

    def __init__(self, policy: Any, input_names: list[str], *, deterministic: bool = True) -> None:
        if not input_names:
            raise ValueError("PolicyRunner needs at least one observation input")
        self.policy = policy
        self.input_names = list(input_names)
        self.deterministic = bool(deterministic)
        self._latest: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        spec = getattr(policy, "obs_spec", None) or {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        for name, enc in (spec.get("encoders") or {}).items():
            shape = enc.get("shape")
            if shape:
                self._shapes[name] = tuple(int(d) for d in shape)
            if enc.get("input_name") and shape:
                self._shapes.setdefault(enc["input_name"], tuple(int(d) for d in shape))
        if spec.get("obs_key") and spec.get("obs_shape"):
            self._shapes.setdefault(spec["obs_key"], tuple(int(d) for d in spec["obs_shape"]))

    def update(self, name: str, value: Any) -> None:
        """Store the newest observation for *name* (array, list, or ROS message)."""
        if name not in self.input_names:
            raise KeyError(f"unknown observation input {name!r}; expected {self.input_names}")
        arr = value if isinstance(value, np.ndarray) else _as_array(value)
        arr = np.asarray(arr, dtype=np.float32)
        shape = self._shapes.get(name)
        if shape is not None and arr.shape != shape and arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)
        with self._lock:
            self._latest[name] = arr

    def ready(self) -> bool:
        with self._lock:
            return all(name in self._latest for name in self.input_names)

    def observation(self) -> dict[str, np.ndarray] | None:
        """The current observation dict, or ``None`` until every input arrived."""
        with self._lock:
            if not all(name in self._latest for name in self.input_names):
                return None
            return {name: self._latest[name] for name in self.input_names}

    def step(self) -> np.ndarray | None:
        """Run the policy on the latest observations; ``None`` while waiting for inputs."""
        obs = self.observation()
        if obs is None:
            return None
        action = self.policy.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        with self._lock:
            self._latest.clear()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "data") or hasattr(value, "position"):
        return msg_to_numpy(value)
    return np.asarray(value, dtype=np.float32)


def ros2_config_for(policy: Any, config_path: str | None = None) -> ROS2Config:
    """The ``ros2:`` block: from *config_path* if given, else the policy's own config.yaml."""
    if config_path:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        # Either a full SRL config (its ros2: block) or a bare ros2 mapping.
        return ROS2Config.from_dict(data.get("ros2", data) if isinstance(data, dict) else data)
    exp = getattr(policy, "experiment", None)
    if exp is not None:
        return exp.model.ros2
    return ROS2Config()


class RLInferenceNode(Node):
    """ROS 2 node: subscribe to observation topics, publish the policy's action.

    Parameters
    ----------
    policy:
        A :class:`srl.policy.Policy` (see :meth:`from_policy_path`).
    obs_topics:
        Mapping input name (encoder name or env obs key) -> topic.
    obs_msg_types, obs_queue_sizes:
        Per-input message class / queue size (default: *obs_msg_type*, 10).
    action_topic, action_msg_type, action_queue_size:
        Where the action goes (default message: ``Float32MultiArray``).
    hz:
        Inference rate.
    """

    def __init__(
        self,
        policy: Any,
        obs_topics: dict[str, str],
        obs_msg_types: dict[str, Any] | None = None,
        obs_queue_sizes: dict[str, int] | None = None,
        action_topic: str = "/srl/action",
        action_msg_type: Any = None,
        obs_msg_type: Any = None,
        action_queue_size: int = 10,
        hz: float = 20.0,
        node_name: str = "rl_inference_node",
    ) -> None:
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy is not installed. Install ROS2 and source it before using RLInferenceNode."
            )
        super().__init__(node_name)
        if not obs_topics:
            raise ValueError("RLInferenceNode needs obs_topics (ros2.observations in the config)")
        self.runner = PolicyRunner(policy, list(obs_topics))
        obs_msg_types = obs_msg_types or {}
        obs_queue_sizes = obs_queue_sizes or {}
        for name, topic in obs_topics.items():
            msg_type = obs_msg_types.get(name, obs_msg_type)
            if msg_type is None:
                from std_msgs.msg import Float32MultiArray

                msg_type = Float32MultiArray
            self.create_subscription(
                msg_type,
                topic,
                lambda msg, n=name: self.runner.update(n, msg),
                int(obs_queue_sizes.get(name, 10)),
            )
        if action_msg_type is None:
            from std_msgs.msg import Float32MultiArray

            action_msg_type = Float32MultiArray
        self._action_msg_type = action_msg_type
        self._action_pub = self.create_publisher(
            action_msg_type, action_topic, int(action_queue_size)
        )
        self.create_timer(1.0 / float(hz), self._timer_callback)
        self.get_logger().info(f"RLInferenceNode started @ {hz} Hz | {policy!r}")

    @classmethod
    def from_policy_path(
        cls,
        policy_path: str,
        *,
        device: str = "cpu",
        checkpoint: str = "auto",
        config_path: str | None = None,
        hz: float = 20.0,
    ) -> RLInferenceNode:
        """Load the policy (:func:`srl.load_policy`) and wire topics from its ``ros2:`` block."""
        import srl

        policy = srl.load_policy(policy_path, device=device, checkpoint=checkpoint)
        ros2_cfg = ros2_config_for(policy, config_path)
        return cls(policy, hz=hz, **node_kwargs(ros2_cfg))

    def _timer_callback(self) -> None:
        action = self.runner.step()
        if action is not None:
            self._publish_action(action)

    def _publish_action(self, action: np.ndarray) -> None:
        msg = self._action_msg_type()
        if hasattr(msg, "data"):
            msg.data = action.tolist()
        elif hasattr(msg, "linear") and hasattr(msg, "angular"):  # geometry_msgs/Twist
            a = list(action.tolist()) + [0.0] * 6
            msg.linear.x, msg.linear.y, msg.linear.z = a[0], a[1], a[2]
            msg.angular.x, msg.angular.y, msg.angular.z = a[3], a[4], a[5]
        else:
            self.get_logger().error(f"Cannot fill action message {type(msg).__name__}")
            return
        self._action_pub.publish(msg)


def node_kwargs(ros2_cfg: ROS2Config) -> dict[str, Any]:
    """:class:`RLInferenceNode` keyword arguments from a ``ros2:`` config block."""
    return {
        "obs_topics": {name: spec.topic for name, spec in ros2_cfg.observations.items()},
        "obs_msg_types": {
            name: resolve_msg_type(spec.msg_type) for name, spec in ros2_cfg.observations.items()
        },
        "obs_queue_sizes": {name: spec.queue_size for name, spec in ros2_cfg.observations.items()},
        "action_topic": ros2_cfg.action_topic,
        "action_msg_type": resolve_msg_type(ros2_cfg.action_msg_type),
        "action_queue_size": ros2_cfg.action_queue_size,
    }


def main(args=None):
    """Entry point for ``ros2 run`` / ``console_scripts``.

    ROS parameters: ``policy_path`` (required; any :func:`srl.load_policy`
    path), ``checkpoint`` (auto/best/final/latest), ``config_path``
    (optional YAML whose ``ros2:`` block overrides the policy's),
    ``device``, ``hz``.
    """
    if not _ROS2_AVAILABLE:
        raise RuntimeError("rclpy not found — cannot start ROS2 node.")

    rclpy.init(args=args)
    params = rclpy.create_node("srl_param_loader")
    params.declare_parameter("policy_path", "")
    params.declare_parameter("checkpoint", "auto")
    params.declare_parameter("config_path", "")
    params.declare_parameter("device", "cpu")
    params.declare_parameter("hz", 20.0)

    def _str(name: str) -> str:
        return params.get_parameter(name).get_parameter_value().string_value

    policy_path, checkpoint = _str("policy_path"), _str("checkpoint") or "auto"
    config_path, device = _str("config_path") or None, _str("device") or "cpu"
    hz = params.get_parameter("hz").get_parameter_value().double_value
    params.destroy_node()
    if not policy_path:
        raise ValueError("Must provide the policy_path parameter (an srl.load_policy() path).")

    node = RLInferenceNode.from_policy_path(
        policy_path, device=device, checkpoint=checkpoint, config_path=config_path, hz=hz
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
