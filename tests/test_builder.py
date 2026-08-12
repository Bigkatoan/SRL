from pathlib import Path

import pytest

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


def _base_config(**overrides) -> dict:
    cfg = {
        "encoders": [
            {"name": "state_enc", "type": "mlp", "input_dim": 4, "latent_dim": 8},
        ],
        "flows": ["state_enc -> actor"],
        "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
    }
    cfg.update(overrides)
    return cfg


def test_missing_action_dim_raises_clear_error() -> None:
    cfg = _base_config(actor={"name": "actor", "type": "gaussian"})  # no action_dim
    with pytest.raises(ValueError, match="action_dim is required"):
        ModelBuilder.from_dict(cfg)


def test_mlp_encoder_missing_input_dim_raises_clear_error() -> None:
    cfg = _base_config(
        encoders=[{"name": "state_enc", "type": "mlp", "latent_dim": 8}]  # no input_dim
    )
    with pytest.raises(ValueError, match="requires 'input_dim'"):
        ModelBuilder.from_dict(cfg)


def test_unknown_encoder_type_error_lists_builtins() -> None:
    cfg = _base_config(
        encoders=[{"name": "state_enc", "type": "mlpp", "input_dim": 4, "latent_dim": 8}]
    )
    with pytest.raises(ValueError) as exc_info:
        ModelBuilder.from_dict(cfg)
    message = str(exc_info.value)
    assert "mlp" in message
    assert "cnn" in message


def test_duplicate_encoder_name_with_different_config_raises_clear_error() -> None:
    cfg = _base_config(
        encoders=[
            {"name": "state_enc", "type": "mlp", "input_dim": 4, "latent_dim": 8},
            {"name": "state_enc", "type": "mlp", "input_dim": 4, "latent_dim": 16},
        ]
    )
    with pytest.raises(ValueError, match="declared more than once"):
        ModelBuilder.from_dict(cfg)
