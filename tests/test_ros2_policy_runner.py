"""The ROS 2 node's inference core (:class:`PolicyRunner`) -- no rclpy needed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import srl
from srl import SAC
from srl.ros2 import rl_node
from srl.ros2.rl_node import PolicyRunner, msg_to_numpy, ros2_config_for

sys.path.insert(0, str(Path(__file__).parent))
from test_public_api import model_config  # noqa: E402


@pytest.fixture(scope="module")
def policy_and_agent(tmp_path_factory):
    cfg = model_config("sac")
    # `auto` dims: the node must not need them resolved by hand.
    cfg["encoders"] = [dict(cfg["encoders"][0], input_dim="auto")]
    cfg["actor"] = dict(cfg["actor"], action_dim="auto")
    cfg["critic"] = dict(cfg["critic"], action_dim="auto")
    cfg["ros2"] = {
        "observations": {"state_enc": {"topic": "/robot/state", "queue_size": 3}},
        "action_topic": "/robot/cmd",
    }
    agent = SAC.from_config(cfg, env=gym.make("Pendulum-v1"))
    agent.learn(64)
    path = agent.save(tmp_path_factory.mktemp("ros") / "policy")
    return srl.load_policy(path), agent


def test_runner_waits_for_every_input_then_matches_agent(policy_and_agent) -> None:
    policy, agent = policy_and_agent
    ros2 = ros2_config_for(policy)
    assert ros2.action_topic == "/robot/cmd"
    assert ros2.observations["state_enc"].topic == "/robot/state"

    runner = PolicyRunner(policy, list(ros2.observations))
    assert not runner.ready() and runner.step() is None
    obs = np.array([0.3, -0.9, 1.5], dtype=np.float32)
    runner.update("state_enc", SimpleNamespace(data=obs.tolist()))  # Float32MultiArray-like
    assert runner.ready()
    action = runner.step()
    expected, _ = agent.predict(obs, deterministic=True)
    np.testing.assert_array_equal(action, expected.reshape(-1))
    assert action.shape == (1,) and -2.0 <= float(action[0]) <= 2.0


def test_runner_reshapes_flat_messages_and_rejects_unknown_inputs(policy_and_agent) -> None:
    policy, _ = policy_and_agent
    runner = PolicyRunner(policy, ["state_enc"])
    runner.update("state_enc", np.zeros((1, 3), dtype=np.float32))  # flattened size matches
    assert runner.observation()["state_enc"].shape == (3,)
    with pytest.raises(KeyError):
        runner.update("camera", [0.0])
    runner.reset()
    assert not runner.ready()
    with pytest.raises(ValueError):
        PolicyRunner(policy, [])


def test_ros2_config_override_file(policy_and_agent, tmp_path) -> None:
    policy, _ = policy_and_agent
    override = tmp_path / "topics.yaml"
    override.write_text("ros2:\n  observations:\n    state_enc: /other\n  action_topic: /cmd2\n")
    cfg = ros2_config_for(policy, str(override))
    assert cfg.observations["state_enc"].topic == "/other" and cfg.action_topic == "/cmd2"


def test_msg_to_numpy() -> None:
    np.testing.assert_array_equal(msg_to_numpy(SimpleNamespace(data=[1, 2])), [1.0, 2.0])
    joint = SimpleNamespace(position=[0.1], velocity=[0.2])
    np.testing.assert_allclose(msg_to_numpy(joint), [0.1, 0.2])
    np.testing.assert_array_equal(msg_to_numpy(SimpleNamespace(data=b"\x01\x02")), [1.0, 2.0])
    with pytest.raises(ValueError):
        msg_to_numpy(object())


@pytest.mark.skipif(rl_node._ROS2_AVAILABLE, reason="rclpy is installed")
def test_node_requires_rclpy(policy_and_agent) -> None:
    policy, _ = policy_and_agent
    with pytest.raises(RuntimeError, match="rclpy"):
        rl_node.RLInferenceNode(policy, {"state_enc": "/robot/state"})
