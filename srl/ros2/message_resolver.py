"""Resolve configured ROS 2 message type names to message classes."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_KNOWN_MESSAGE_TYPES = {
    "Float32MultiArray": "std_msgs.msg.Float32MultiArray",
    "Image": "sensor_msgs.msg.Image",
    "JointState": "sensor_msgs.msg.JointState",
    "Twist": "geometry_msgs.msg.Twist",
}


def resolve_msg_type(msg_type_name: str | None, default: Any = None) -> Any:
    """Return a ROS 2 message class from a short or dotted type name.

    Parameters
    ----------
    msg_type_name:
        Either a short known message name like ``Float32MultiArray`` or a
        dotted import path like ``std_msgs.msg.Float32MultiArray``.
    default:
        Value returned when ``msg_type_name`` is ``None``.
    """
    if msg_type_name is None:
        return default

    import_path = _KNOWN_MESSAGE_TYPES.get(msg_type_name, msg_type_name)
    if "." not in import_path:
        raise ValueError(
            f"Unknown ROS2 message type '{msg_type_name}'. Use a supported short name "
            "or a full dotted import path like 'std_msgs.msg.Float32MultiArray'."
        )

    module_name, attr_name = import_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import ROS2 message module '{module_name}' while resolving '{msg_type_name}'."
        ) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"ROS2 message class '{attr_name}' not found in module '{module_name}'."
        ) from exc