"""Tests for srl-train's Visual* algorithm-config auto-detection.

``VisualPPOConfig`` / ``VisualSACConfig`` are what actually switch PPO and SAC
onto their auxiliary-encoder-loss code paths -- ``ppo.py`` branches on
``hasattr(self.cfg, "encoder_lr")`` and ``sac.py`` on
``isinstance(self.cfg, VisualSACConfig)``.  The CLI used to hardcode the plain
``PPOConfig``/``SACConfig``, so every auxiliary loss configured through YAML
was dead code: the aux heads were built from the encoder's ``aux_type`` and
then never trained.

These tests pin the detection rules and, crucially, that the *built* config
satisfies the exact predicate each algorithm branches on.
"""

from __future__ import annotations

import pytest
import torch

from srl.cli.train import (
    _build_algo_config,
    _resolve_algo_config_cls,
    _train_section,
)
from srl.core.config import (
    DDPGConfig,
    PPOConfig,
    SACConfig,
    VisualPPOConfig,
    VisualSACConfig,
)

_CNN_ENCODER = {
    "name": "visual_enc",
    "type": "cnn",
    "input_shape": [3, 96, 96],
    "latent_dim": 256,
    "layers": [[32, 8, 4, "relu"], [64, 4, 2, "relu"], [64, 3, 1, "relu"]],
}


def _cfg(aux_type: str | None = None, train: dict | None = None) -> tuple[dict, dict]:
    """Build a (train_cfg, raw_cfg) pair like ``_train_section`` returns."""
    encoder = dict(_CNN_ENCODER)
    if aux_type is not None:
        encoder["aux_type"] = aux_type
    raw_cfg = {
        "encoders": [encoder],
        "flows": ["visual_enc -> actor", "visual_enc -> critic"],
        "train": train or {},
    }
    return raw_cfg["train"], raw_cfg


# ──────────────────────────────────────────────────────────────────────────────
# The shipped config -- the end-to-end case this fix exists for
# ──────────────────────────────────────────────────────────────────────────────


def test_shipped_car_racing_visual_config_builds_visual_ppo_config() -> None:
    """configs/envs/car_racing_ppo_visual.yaml must drive the aux path.

    It declares ``aux_type: autoencoder`` on its one encoder and sets *no*
    Visual-only field in ``train:``, so detection has to come from the encoder.
    """
    train_cfg, raw_cfg = _train_section("configs/envs/car_racing_ppo_visual.yaml")

    config_cls, extra = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg, train_cfg)
    assert config_cls is VisualPPOConfig

    config = _build_algo_config(config_cls, train_cfg, num_envs=4, **extra)
    # This is the exact predicate srl/algorithms/ppo.py branches on.
    assert hasattr(config, "encoder_lr")
    # An autoencoder head needs the reconstruction ("ae") loss, not the
    # VisualPPOConfig default of "curl".
    assert config.aux_loss_type == "ae"
    # Plain fields from train: must still be honoured.
    assert config.lr == pytest.approx(3e-4)
    assert config.n_epochs == 10


# ──────────────────────────────────────────────────────────────────────────────
# Detection signals
# ──────────────────────────────────────────────────────────────────────────────


def test_no_aux_type_anywhere_keeps_plain_config() -> None:
    train_cfg, raw_cfg = _cfg(aux_type=None, train={"lr": 3e-4})

    config_cls, extra = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg, train_cfg)

    assert config_cls is PPOConfig
    assert extra == {}
    # Guards the negative side of ppo.py's predicate: a state-based run must
    # not silently get a head-only optimizer.
    assert not hasattr(_build_algo_config(config_cls, train_cfg), "encoder_lr")


@pytest.mark.parametrize(
    ("aux_type", "expected_loss"),
    [("autoencoder", "ae"), ("contrastive", "curl"), ("byol", "byol")],
)
def test_encoder_aux_type_maps_to_matching_aux_loss(aux_type: str, expected_loss: str) -> None:
    """Every registered aux head selects the loss that can consume it.

    SAC dispatches on ``aux_loss_type``: "ae" looks for a ConvDecoderHead,
    "curl"/"byol" for a ProjectionHead.  Leaving the class default ("curl")
    against an autoencoder head makes ``_compute_aux_loss`` return None and the
    aux loss silently never runs.
    """
    train_cfg, raw_cfg = _cfg(aux_type=aux_type)

    config_cls, extra = _resolve_algo_config_cls("sac", SACConfig, raw_cfg, train_cfg)

    assert config_cls is VisualSACConfig
    assert extra["aux_loss_type"] == expected_loss
    config = _build_algo_config(config_cls, train_cfg, **extra)
    assert isinstance(config, VisualSACConfig)  # sac.py's _is_visual predicate
    assert config.aux_weight > 0.0  # sac.py skips the aux loss when this is 0


def test_visual_only_train_field_opts_in_without_encoder_aux_type() -> None:
    """Setting e.g. encoder_lr in train: is enough on its own."""
    train_cfg, raw_cfg = _cfg(aux_type=None, train={"encoder_lr": 5e-5, "aux_weight": 0.25})

    config_cls, _ = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg, train_cfg)

    assert config_cls is VisualPPOConfig
    config = _build_algo_config(config_cls, train_cfg)
    assert config.encoder_lr == pytest.approx(5e-5)
    assert config.aux_weight == pytest.approx(0.25)


def test_explicit_aux_loss_type_in_train_block_is_not_overridden() -> None:
    """A user's explicit choice beats the aux_type-derived default."""
    train_cfg, raw_cfg = _cfg(aux_type="autoencoder", train={"aux_loss_type": "vae"})

    config_cls, extra = _resolve_algo_config_cls("sac", SACConfig, raw_cfg, train_cfg)

    assert config_cls is VisualSACConfig
    assert "aux_loss_type" not in extra
    assert _build_algo_config(config_cls, train_cfg, **extra).aux_loss_type == "vae"


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases -- warn, never crash
# ──────────────────────────────────────────────────────────────────────────────


def test_aux_type_with_algo_that_has_no_visual_variant_warns_and_stays_plain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """DDPG/TD3 have no Visual config; the run stays valid but says so."""
    train_cfg, raw_cfg = _cfg(aux_type="autoencoder")

    config_cls, extra = _resolve_algo_config_cls("ddpg", DDPGConfig, raw_cfg, train_cfg)

    assert config_cls is DDPGConfig
    assert extra == {}
    err = capsys.readouterr().err
    assert "no visual config variant" in err
    assert "visual_enc" in err


def test_unrecognized_aux_type_warns_and_does_not_switch_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo'd aux_type builds no head, so it must not select Visual*.

    Switching on it would hand PPO a head-only optimizer with no aux modules,
    leaving the encoder with no gradient source at all.
    """
    train_cfg, raw_cfg = _cfg(aux_type="ae")  # not a registered head name

    config_cls, _ = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg, train_cfg)

    assert config_cls is PPOConfig
    err = capsys.readouterr().err
    assert "not a known aux head" in err


def test_multiple_distinct_aux_types_warn_and_pick_the_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_cfg = {
        "encoders": [
            dict(_CNN_ENCODER, name="enc_a", aux_type="autoencoder"),
            dict(_CNN_ENCODER, name="enc_b", aux_type="byol"),
        ],
        "train": {},
    }

    config_cls, extra = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg["train"], {})
    assert config_cls is PPOConfig  # no encoders in the (empty) cfg passed above

    config_cls, extra = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg, raw_cfg["train"])
    assert config_cls is VisualPPOConfig
    assert extra["aux_loss_type"] == "ae"
    assert "multiple encoder aux_types" in capsys.readouterr().err


def test_missing_or_malformed_encoders_section_is_tolerated() -> None:
    for raw_cfg in ({}, {"encoders": None}, {"encoders": ["not-a-dict"]}):
        config_cls, extra = _resolve_algo_config_cls("ppo", PPOConfig, raw_cfg, {})
        assert config_cls is PPOConfig
        assert extra == {}


# ──────────────────────────────────────────────────────────────────────────────
# The aux loss actually runs and is observable
# ──────────────────────────────────────────────────────────────────────────────


def test_ppo_update_surfaces_nonzero_aux_loss_metric() -> None:
    """End-to-end on the algorithm side: aux loss runs and reaches metrics.

    Without a surfaced metric there is no way to tell a working aux loss from
    a silently-skipped one, which is exactly how this stayed broken.
    """
    from srl.algorithms.ppo import PPO
    from srl.registry.builder import ModelBuilder

    model = ModelBuilder.from_dict(
        {
            "encoders": [dict(_CNN_ENCODER, input_shape=[3, 32, 32], aux_type="autoencoder")],
            "flows": ["visual_enc -> actor", "visual_enc -> critic"],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 2},
            "critic": {"name": "critic", "type": "value"},
        }
    )
    config = VisualPPOConfig(n_steps=8, batch_size=4, n_epochs=2, num_envs=1, aux_weight=0.5)
    agent = PPO(model, config=config, device="cpu")
    assert agent.encoder_optimizer is not None

    rng = torch.Generator().manual_seed(0)
    for _ in range(config.n_steps):
        agent.buffer.add(
            obs={"visual_enc": torch.rand(1, 3, 32, 32, generator=rng).numpy()},
            action=torch.zeros(1, 2).numpy(),
            reward=torch.zeros(1).numpy(),
            done=torch.zeros(1).numpy(),
            log_prob=torch.zeros(1).numpy(),
            value=torch.zeros(1).numpy(),
        )
    agent.buffer.compute_returns_and_advantages(last_value=0.0)

    metrics = agent.update()

    assert "aux_loss" in metrics, f"aux loss never surfaced; got {sorted(metrics)}"
    assert metrics["aux_loss"] > 0.0, "aux loss is exactly zero -- it never engaged"
    assert metrics["aux_loss_weighted"] == pytest.approx(
        metrics["aux_loss"] * config.aux_weight, rel=1e-3
    )


def test_shipped_car_racing_sac_visual_config_computes_nonzero_aux_loss() -> None:
    """The shipped visual SAC config must actually reach SAC's aux loss.

    SAC's aux helpers find the encoder by substring-matching the raw
    observation key, so this also pins the config's `state_enc` naming: rename
    the encoder and `_compute_aux_loss` silently returns None.
    """
    import copy

    from srl.algorithms.sac import SAC
    from srl.registry.builder import ModelBuilder

    path = "configs/envs/car_racing_sac_visual.yaml"
    train_cfg, raw_cfg = _train_section(path)

    config_cls, extra = _resolve_algo_config_cls("sac", SACConfig, raw_cfg, train_cfg)
    assert config_cls is VisualSACConfig
    assert extra["aux_loss_type"] == "ae"

    config = _build_algo_config(config_cls, train_cfg, action_dim=3, **extra)
    model = ModelBuilder.from_yaml(path)
    agent = SAC(model, copy.deepcopy(model), config=config, device="cpu")
    assert agent._is_visual

    # "state" is the observation key GymnasiumWrapper emits.
    aux = agent._compute_aux_loss({"state": torch.rand(2, 3, 96, 96)}, torch.zeros(2, 3))

    assert aux is not None, "aux loss returned None -- encoder/obs key never matched"
    assert aux.detach().item() > 0.0


def test_plain_ppo_config_reports_no_aux_loss_metric() -> None:
    """The negative control for the test above."""
    from srl.algorithms.ppo import PPO
    from srl.registry.builder import ModelBuilder

    model = ModelBuilder.from_dict(
        {
            "encoders": [dict(_CNN_ENCODER, input_shape=[3, 32, 32], aux_type="autoencoder")],
            "flows": ["visual_enc -> actor", "visual_enc -> critic"],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 2},
            "critic": {"name": "critic", "type": "value"},
        }
    )
    agent = PPO(model, config=PPOConfig(n_steps=4, batch_size=4, n_epochs=1, num_envs=1))
    assert agent.encoder_optimizer is None

    for _ in range(4):
        agent.buffer.add(
            obs={"visual_enc": torch.rand(1, 3, 32, 32).numpy()},
            action=torch.zeros(1, 2).numpy(),
            reward=torch.zeros(1).numpy(),
            done=torch.zeros(1).numpy(),
            log_prob=torch.zeros(1).numpy(),
            value=torch.zeros(1).numpy(),
        )
    agent.buffer.compute_returns_and_advantages(last_value=0.0)

    assert "aux_loss" not in agent.update()
