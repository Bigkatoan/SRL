from pathlib import Path

from srl.registry.builder import ModelBuilder


def test_model_builder_from_dict_sets_encoder_input_names() -> None:
    model = ModelBuilder.from_dict(
        {
            "encoders": [
                {
                    "name": "state_enc",
                    "type": "mlp",
                    "input_dim": 4,
                    "latent_dim": 8,
                    "input_name": "state",
                }
            ],
            "flows": ["state_enc -> actor"],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
        }
    )

    assert model.encoder_input_names == {"state_enc": "state"}


def test_model_builder_from_yaml_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
encoders:
  - name: state_enc
    type: mlp
    input_dim: 4
    latent_dim: 8
flows:
  - "state_enc -> actor"
actor:
  name: actor
  type: gaussian
  action_dim: 1
""".strip(),
        encoding="utf-8",
    )

    model = ModelBuilder.from_yaml(config_path)
    assert "state_enc" in model.encoders