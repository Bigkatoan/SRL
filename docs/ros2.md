# ROS 2 Python API

SRL exposes ROS 2 integration as an optional Python API. It is not a ROS 2 package and it does not ship launch files.

## What it provides

- A Python class for embedding an SRL policy into your ROS 2 application
- Observation-topic to model-input routing
- Action publication from model output
- Optional integration when `rclpy` is available in the local ROS 2 installation

## Usage

```python
import rclpy
import torch

from srl.registry.builder import ModelBuilder
from srl.ros2.rl_node import RLInferenceNode

rclpy.init()

model = ModelBuilder.from_yaml("configs/envs/halfcheetah_sac.yaml")
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])

node = RLInferenceNode(
    model=model,
    obs_topics={"state_enc": "/robot/state"},
    action_topic="/robot/cmd",
    hz=50.0,
    device="cpu",
)

rclpy.spin(node)
```

## Message model

| Topic kind | Default message type | Purpose |
|---|---|---|
| Observation | `std_msgs/Float32MultiArray` | Flattened sensor/state vector |
| Action | `std_msgs/Float32MultiArray` | Continuous action vector |

## Dependency model

SRL does not install ROS 2 Python packages for you. Use the Python environment provided by your ROS 2 installation, for example:

```bash
source /opt/ros/humble/setup.bash
python -c "import rclpy"
```

If `rclpy` is unavailable, the ROS 2 API remains optional and the rest of SRL still works.