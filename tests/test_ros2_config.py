from srl.registry.config_schema import AgentModelConfig


def test_ros2_config_parses_structured_observations() -> None:
    cfg = AgentModelConfig.from_dict(
        {
            "encoders": [
                {"name": "state_enc", "type": "mlp", "input_dim": 4, "latent_dim": 8},
            ],
            "flows": ["state_enc -> actor"],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
            "ros2": {
                "observations": {
                    "state_enc": {"topic": "/robot/state", "queue_size": 5},
                },
                "action_topic": "/robot/cmd",
            },
        }
    )

    assert cfg.ros2.observations["state_enc"].topic == "/robot/state"
    assert cfg.ros2.observations["state_enc"].queue_size == 5
    assert cfg.ros2.action_topic == "/robot/cmd"