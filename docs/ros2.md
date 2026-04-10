# ROS 2 Integration

SRL includes a ROS 2 node for deploying trained policies as real-time action publishers.

---

## Architecture

```
[Sensor topics] → rl_agent_node → [Action topics]
                       ↑
              [Checkpoint .pt file]
```

The `rl_agent_node`:

1. Subscribes to one or more observation topics
2. Runs inference with the loaded policy at a configurable rate
3. Publishes actions on a `Float32MultiArray` topic

---

## Launch

```bash
# Source ROS 2 and SRL
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch srl rl_agent.launch.py \
    checkpoint:=/path/to/checkpoint.pt \
    config:=/path/to/model.yaml \
    obs_topic:=/robot/state \
    action_topic:=/robot/cmd \
    rate:=50
```

---

## Python API

```python
import rclpy
from srl.ros2.rl_node import RLAgentNode

rclpy.init()
node = RLAgentNode(
    config_path="configs/envs/halfcheetah_sac.yaml",
    checkpoint_path="checkpoints/sac_halfcheetah_v5/model_1000000.pt",
    obs_topic="/robot/state",
    action_topic="/robot/cmd",
    rate_hz=50,
)
rclpy.spin(node)
```

---

## Message types

| Topic | Message type | Description |
|---|---|---|
| `obs_topic` | `std_msgs/Float32MultiArray` | Flattened observation |
| `action_topic` | `std_msgs/Float32MultiArray` | Continuous action vector |

---

## Installing ROS 2 extras

```bash
pip install "srl-rl[ros2]"
# or
pip install rclpy std_msgs sensor_msgs
```
