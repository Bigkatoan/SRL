# ROS 2 Python API

SRL exposes ROS 2 integration as an optional Python API. It is not a ROS 2 package and it does not ship launch files.

## What it provides

- A Python class for embedding an SRL policy into your ROS 2 application
- Observation-topic to model-input routing
- Action publication from model output
- Optional integration when `rclpy` is available in the local ROS 2 installation
- YAML-driven topic configuration through a top-level `ros2` block consumed by the current inference node

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

## YAML topic configuration

The current ROS 2 node reads topic mappings from a top-level `ros2` block in the YAML file.

Example:

```yaml
encoders:
    - name: visual_enc
        type: cnn
        input_shape: [3, 64, 64]
        latent_dim: 128

    - name: state_enc
        type: mlp
        input_dim: 18
        latent_dim: 64

ros2:
    observations:
        visual_enc: /camera/image_raw
        state_enc: /robot/joint_states
    action_topic: /robot/cmd_vel
```

This pattern already exists in [sac_multimodal.yaml](/home/ubuntu/antd/SRL/configs/sac_multimodal.yaml).

The current preferred key is `ros2.observations`. Legacy `ros2.obs_topics` remains the backward-compatible fallback for older configs.

## Message model

| Topic kind | Default message type | Purpose |
|---|---|---|
| Observation | `std_msgs/Float32MultiArray` | Flattened sensor/state vector |
| Action | `std_msgs/Float32MultiArray` | Continuous action vector |

The current default path is best suited for flattened vector observations. More complex message types such as camera images or richer robot state messages still need careful adaptation or subclassing.

## Dependency model

SRL does not install ROS 2 Python packages for you. Use the Python environment provided by your ROS 2 installation, for example:

```bash
source /opt/ros/humble/setup.bash
python -c "import rclpy"
```

If `rclpy` is unavailable, the ROS 2 API remains optional and the rest of SRL still works.

## Current limitations

- SRL is not yet distributed as an `ament` ROS 2 package.
- The current node is Python-API-first and does not ship launch files.
- Topic routing exists in YAML, but that section is not yet validated by `config_schema.py`.
- The inference node still needs stronger parity with the observation remapping logic used during training and normal runtime model execution.