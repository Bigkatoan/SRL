# srl.policy — inference & deployment

{func}`srl.load_policy` loads a checkpoint as a {class}`~srl.policy.Policy`
(model + observation normalizer + action rescale map; no env, optimizer or
buffer). See [Saving, loading & deployment](../deployment.md).

```{eval-rst}
.. autofunction:: srl.policy.load_policy

.. autofunction:: srl.policy.resolve_policy_path

.. autoclass:: srl.policy.Policy
   :members: predict, reset, export_torchscript, input_keys, is_recurrent, from_checkpoint
```

## ROS 2

```{eval-rst}
.. autoclass:: srl.ros2.rl_node.PolicyRunner
   :members: update, ready, observation, step, reset

.. autoclass:: srl.ros2.rl_node.RLInferenceNode
   :members: from_policy_path

.. autofunction:: srl.ros2.rl_node.ros2_config_for

.. autofunction:: srl.ros2.rl_node.msg_to_numpy
```
