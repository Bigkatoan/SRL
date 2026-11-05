# ROS 2 Python API

SRL exposes ROS 2 integration as an optional Python API. The runtime node is still Python-first, but the repo now also includes an `ament` starter template under `templates/ros2/srl_inference_pkg/`.

## What it provides

- {class}`srl.ros2.rl_node.RLInferenceNode`: subscribes to observation topics,
  runs a policy loaded with {func}`srl.load_policy`, publishes the action (in env
  units) at a fixed rate
- {class}`srl.ros2.rl_node.PolicyRunner`: the ROS-free core of the node (latest
  observation per input -> `policy.predict`), usable and testable without `rclpy`
- YAML-driven topic configuration through the `ros2:` block of the training
  config -- saved in the checkpoint's `config.yaml`, so the topics travel with
  the policy
- Configurable per-topic message types and queue sizes

Because the node loads a checkpoint with `srl.load_policy`, the model is built
from the checkpoint's resolved `config.yaml` (explicit dims): configs that used
`input_dim: auto` / `action_dim: auto` for training need nothing extra, and the
observation normalizer and action rescaling the policy was trained with are
applied automatically.

## Usage

```python
import rclpy

from srl.ros2.rl_node import RLInferenceNode

rclpy.init()
node = RLInferenceNode.from_policy_path(
    "checkpoints/sac_sac_multimodal",   # agent.save() dir or srl-train checkpoint dir
    checkpoint="best",                  # auto | best | final | latest
    hz=50.0,
    device="cpu",
)                                       # topics: the policy's config.yaml ros2: block
rclpy.spin(node)
```

With an explicit policy and topics:

```python
import srl

policy = srl.load_policy("runs/my_agent")
node = RLInferenceNode(policy, obs_topics={"state_enc": "/robot/state"}, action_topic="/robot/cmd")
```

`python -m srl.ros2.rl_node` (or the template package) reads the ROS parameters
`policy_path` (required), `checkpoint`, `config_path` (an optional YAML whose
`ros2:` block overrides the stored topics), `device` and `hz`.

### Testing without ROS

```python
from srl.ros2.rl_node import PolicyRunner

runner = PolicyRunner(srl.load_policy("runs/my_agent"), ["state_enc"])
runner.update("state_enc", [0.1, -0.3, 0.5])     # array, list or a message with .data
action = runner.step()                           # None until every input arrived
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
        visual_enc:
            topic: /camera/image_raw
            msg_type: sensor_msgs/Image
            queue_size: 2
        state_enc:
            topic: /robot/joint_states
            msg_type: std_msgs/Float32MultiArray
            queue_size: 10
    action_topic: /robot/cmd_vel
    action_msg_type: std_msgs/Float32MultiArray
    action_queue_size: 10
```

This pattern already exists in [sac_multimodal.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/sac_multimodal.yaml).

The current preferred key is `ros2.observations`. Legacy `ros2.obs_topics` remains the backward-compatible fallback for older configs.

Each observation entry may be either:

- a plain topic string
- or a structured mapping with `topic`, optional `msg_type`, and optional `queue_size`

`msg_type` accepts:

- the short names `Float32MultiArray`, `Image`, `JointState`, `Twist`
- ROS interface names such as `std_msgs/Float32MultiArray` (or `std_msgs/msg/Float32MultiArray`)
- dotted Python import paths such as `std_msgs.msg.Float32MultiArray`

Observation keys are encoder names (or the env obs keys the encoders'
`input_name`s refer to). Flat message data (`Float32MultiArray`, `Image.data`)
is reshaped to the encoder's input shape when the sizes match; `JointState`
becomes `position + velocity`.

## Message model

| Topic kind | Default message type | Purpose |
|---|---|---|
| Observation | `std_msgs/Float32MultiArray` | Flattened sensor/state vector |
| Action | `std_msgs/Float32MultiArray` | Continuous action vector |

The current default path is best suited for flattened vector observations. More complex message types such as camera images or richer robot state messages still need careful adaptation or subclassing.

Queue sizing is also configurable from YAML:

- per observation via `ros2.observations.<name>.queue_size`
- for the action publisher via `ros2.action_queue_size`

If queue sizes are omitted, the node falls back to its default queue settings.

## Deployment path

There are now two supported starting points:

1. Direct Python embedding with `RLInferenceNode`
2. The `ament_python` starter package template in `templates/ros2/srl_inference_pkg/`

The template includes:

- `package.xml`
- `setup.py` / `setup.cfg`
- a launch file
- a small launcher node that loads the policy from the `policy_path` ROS parameter

This is still a starter path, not a fully productized ROS 2 distribution.

## Dependency model

SRL does not install ROS 2 Python packages for you. Use the Python environment provided by your ROS 2 installation, for example:

```bash
source /opt/ros/humble/setup.bash
python -c "import rclpy"
```

If `rclpy` is unavailable, the ROS 2 API remains optional and the rest of SRL still works.

## Current limitations

- SRL itself is not yet distributed as an installable `ament` ROS 2 package.
- The starter template is intentionally minimal and expects you to adapt package naming, dependencies, and launch behavior.
- The built-in node assumes observation messages can be converted into arrays that match the model input contract.
- Complex message decoding, image preprocessing, transforms, and robot-specific adapters are still deployment-side work.