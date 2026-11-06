"""Strict config validation: unknown keys, enums, ranges, schema_version, deprecations."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from srl.config import (
    SCHEMA_VERSION,
    ConfigDeprecationWarning,
    ConfigError,
    build_algo_config,
    build_runner_config,
    load_experiment,
)
from srl.core.config import (
    AsyncRunnerConfig,
    DDPGConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    VisualSACConfig,
)
from srl.registry.builder import ModelBuilder
from srl.registry.config_schema import AgentModelConfig, EncoderConfig, HeadConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = sorted(
    p for p in (REPO_ROOT / "configs").rglob("*.yaml") if "benchmarks" not in p.parts
)


def _base(**overrides) -> dict:
    cfg = {
        "algo": "ppo",
        "env_id": "Pendulum-v1",
        "encoders": [
            {"name": "state_enc", "type": "mlp", "input_dim": 3, "latent_dim": 16, "layers": [16]}
        ],
        "flows": ["state_enc -> actor", "state_enc -> critic"],
        "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
        "critic": {"name": "critic", "type": "value"},
        "train": {"lr": 3e-4, "total_steps": 1000},
    }
    cfg.update(overrides)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Shipped configs
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: p.name)
def test_shipped_config_passes_strict_validation_without_warnings(path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigDeprecationWarning)
        exp = load_experiment(path)
    assert exp.schema_version == SCHEMA_VERSION
    assert exp.algo is not None and exp.algo_config is not None


# ──────────────────────────────────────────────────────────────────────────────
# Unknown keys -> error with a suggestion
# ──────────────────────────────────────────────────────────────────────────────


def test_unknown_train_key_suggests_closest_field() -> None:
    cfg = _base(train={"lrr": 1e-3})
    with pytest.raises(ConfigError, match=r"train\.lrr: unknown key \(did you mean 'lr'\?\)"):
        load_experiment(cfg)


def test_train_key_of_another_algorithm_is_named() -> None:
    with pytest.raises(ConfigError, match=r"train\.tau: not a PPO hyperparameter.*SAC"):
        load_experiment(_base(train={"tau": 0.01}))


def test_unknown_encoder_and_head_keys_suggest() -> None:
    cfg = _base()
    cfg["encoders"][0]["latnet_dim"] = 32
    cfg["actor"]["log_std_mni"] = -3
    with pytest.raises(ConfigError) as info:
        load_experiment(cfg)
    text = str(info.value)
    assert "encoders[0].latnet_dim" in text and "did you mean 'latent_dim'" in text
    assert "actor.log_std_mni" in text and "did you mean 'log_std_min'" in text
    # ...and both are reported in one go.
    assert len(info.value.errors) == 2


def test_unknown_top_level_key_suggests() -> None:
    cfg = _base()
    cfg["trian"] = cfg.pop("train")
    with pytest.raises(
        ConfigError, match=r"trian: unknown top-level key \(did you mean 'train'\?\)"
    ):
        load_experiment(cfg)


def test_model_builder_is_strict_too() -> None:
    cfg = _base()
    cfg["encoders"][0]["laten_dim"] = 32
    with pytest.raises(ConfigError, match="did you mean 'latent_dim'"):
        ModelBuilder.from_dict(cfg)


def test_unknown_encoder_type_suggests_and_lists_builtins() -> None:
    cfg = _base()
    cfg["encoders"][0]["type"] = "mlpp"
    with pytest.raises(ConfigError, match=r"unknown encoder type 'mlpp' \(did you mean 'mlp'\?\)"):
        AgentModelConfig.from_dict(cfg)
    with pytest.raises(ConfigError, match=r"'cnn'.*'lstm'.*'mlp'.*'text'"):
        AgentModelConfig.from_dict(cfg)


def test_flow_referencing_unknown_node_suggests() -> None:
    cfg = _base(flows=["state_en -> actor"])
    with pytest.raises(ConfigError, match=r"source 'state_en'.*did you mean 'state_enc'"):
        AgentModelConfig.from_dict(cfg)


def test_build_algo_config_is_strict() -> None:
    with pytest.raises(
        ConfigError, match=r"train\.batch_sise: unknown key \(did you mean 'batch_size'"
    ):
        build_algo_config("sac", {"batch_sise": 64})
    # trainer / runner keys of the train: block are accepted and skipped
    cfg = build_algo_config(
        "sac", {"total_steps": 10, "seed": 1, "use_async": True, "batch_size": 64}
    )
    assert type(cfg) is SACConfig and cfg.batch_size == 64


def test_build_algo_config_overrides_and_visual_selection() -> None:
    cfg = build_algo_config("sac", {"aux_weight": 0.3}, overrides={"action_dim": 4, "nope": 1})
    assert isinstance(cfg, VisualSACConfig)
    assert cfg.action_dim == 4 and cfg.aux_weight == pytest.approx(0.3)
    model = {"encoders": [{"name": "e", "type": "cnn", "aux_type": "autoencoder"}]}
    cfg = build_algo_config("sac", {}, model=model)
    assert isinstance(cfg, VisualSACConfig) and cfg.aux_loss_type == "ae"
    assert type(build_algo_config("sac", {}, config_cls=SACConfig)) is SACConfig


def test_extra_keys_of_a_type_and_explicit_extra() -> None:
    cfg = AgentModelConfig.from_dict(
        {
            "encoders": [
                {"name": "lang", "type": "text", "latent_dim": 8, "vocab_size": 64, "max_len": 8},
                {"name": "s", "type": "mlp", "input_dim": 2, "extra": {"anything": 1}},
            ]
        }
    )
    assert cfg.encoders[0].extra == {"vocab_size": 64, "max_len": 8}
    assert cfg.encoders[1].extra == {"anything": 1}
    # mlp does not declare vocab_size -> error, not silently stored
    with pytest.raises(ConfigError, match="encoders\\[0\\].vocab_size: unknown key"):
        EncoderConfig.from_dict(
            {"name": "s", "type": "mlp", "input_dim": 2, "vocab_size": 3}, "encoders[0]"
        )


def test_text_encoder_extra_keys_reach_the_module() -> None:
    model = ModelBuilder.from_dict(
        {
            "encoders": [{"name": "lang", "type": "text", "latent_dim": 8, "vocab_size": 50}],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
        }
    )
    assert model.encoders["lang"].embed.num_embeddings == 50


# ──────────────────────────────────────────────────────────────────────────────
# Types, enums, ranges
# ──────────────────────────────────────────────────────────────────────────────


def test_yaml_string_numbers_are_coerced() -> None:
    exp = load_experiment(_base(train={"lr": "3e-4", "n_steps": "2e3", "total_steps": "1e6"}))
    assert exp.algo_config.lr == pytest.approx(3e-4)
    assert exp.algo_config.n_steps == 2000 and isinstance(exp.algo_config.n_steps, int)
    assert exp.trainer.total_steps == 1_000_000


@pytest.mark.parametrize(
    ("train", "pattern"),
    [
        ({"lr": "fast"}, r"train\.lr: expected a number"),
        ({"n_steps": 3.5}, r"train\.n_steps: expected an integer"),
        ({"normalize_obs": "yes"}, r"train\.normalize_obs: expected a boolean"),
        (
            {"lr_schedule": "adaptiv"},
            r"lr_schedule: invalid value 'adaptiv' \(did you mean 'adaptive'",
        ),
        ({"gamma": 1.5}, r"train\.gamma: must be in \(0, 1\], got 1\.5"),
        ({"lr": -1e-3}, r"train\.lr: must be > 0"),
        ({"batch_size": 0}, r"train\.batch_size: must be >= 1"),
        ({"min_lr": 1.0, "max_lr": 0.1}, r"train\.min_lr: must be <= max_lr"),
    ],
)
def test_type_enum_and_range_errors(train: dict, pattern: str) -> None:
    with pytest.raises(ConfigError, match=pattern):
        load_experiment(_base(train=train))


def test_off_policy_enum_errors() -> None:
    with pytest.raises(ConfigError, match=r"action_noise: invalid value 'uo'"):
        build_algo_config("ddpg", {"action_noise": "uo"})
    with pytest.raises(ConfigError, match=r"her_strategy: invalid value 'futur'.*'future'"):
        build_algo_config("sac", {"her_strategy": "futur"})
    with pytest.raises(ConfigError, match=r"aux_loss_type: invalid value 'curll'"):
        build_algo_config("sac", {"aux_loss_type": "curll"})
    assert build_algo_config("sac", {"target_entropy": -2}).target_entropy == -2.0
    with pytest.raises(ConfigError, match="target_entropy"):
        build_algo_config("sac", {"target_entropy": "automatic"})


def test_direct_construction_is_validated() -> None:
    with pytest.raises(ConfigError, match=r"PPOConfig\.gamma: must be in \(0, 1\]"):
        PPOConfig(gamma=0.0)
    with pytest.raises(ConfigError, match="action_noise"):
        DDPGConfig(action_noise="uniform")
    with pytest.raises(ConfigError, match="log_std_min"):
        HeadConfig(name="actor", type="gaussian", log_std_min=1.0, log_std_max=0.0)
    # valid values, including dataclasses.replace, still work
    cfg = dataclasses.replace(SACConfig(), gamma=1.0, action_dim=3)
    assert cfg.gamma == 1.0


def test_env_section_validation() -> None:
    with pytest.raises(
        ConfigError, match=r"env_type: unknown env type 'gaol' \(did you mean 'goal'"
    ):
        load_experiment(_base(env_type="gaol"))
    with pytest.raises(ConfigError, match=r"train\.env_id: .*conflicts with top-level"):
        load_experiment(_base(train={"env_id": "Other-v0"}))
    with pytest.warns(ConfigDeprecationWarning, match="train.include_goal is deprecated"):
        exp = load_experiment(_base(train={"include_goal": False}))
    assert exp.env.include_goal is False


# ──────────────────────────────────────────────────────────────────────────────
# schema_version, deprecations, dead fields
# ──────────────────────────────────────────────────────────────────────────────


def test_schema_version() -> None:
    assert load_experiment(_base()).schema_version == 1
    assert load_experiment(_base(schema_version=1)).schema_version == 1
    with pytest.raises(ConfigError, match="schema_version: unsupported version 2"):
        load_experiment(_base(schema_version=2))


def test_losses_block_warns_and_is_ignored() -> None:
    with pytest.warns(ConfigDeprecationWarning, match="'losses:' is deprecated"):
        cfg = AgentModelConfig.from_dict(_base(losses=[{"name": "policy", "weight": 1.0}]))
    assert not hasattr(cfg, "losses")


def test_frame_stack_removed() -> None:
    enc = {"name": "s", "type": "mlp", "input_dim": 2}
    with pytest.warns(ConfigDeprecationWarning, match="frame_stack"):
        EncoderConfig.from_dict(dict(enc, frame_stack=1))
    with pytest.raises(ConfigError, match="frame stacking is not implemented"):
        EncoderConfig.from_dict(dict(enc, frame_stack=4))


def test_autoencoder_aux_on_flat_encoder_is_rejected() -> None:
    with pytest.raises(ConfigError, match="'autoencoder' reconstructs an image"):
        EncoderConfig.from_dict(
            {"name": "s", "type": "mlp", "input_dim": 2, "aux_type": "autoencoder"}
        )


def test_runner_config_from_train_block() -> None:
    assert build_runner_config({"use_async": True, "lr": 1}) == AsyncRunnerConfig(use_async=True)
    with pytest.warns(ConfigDeprecationWarning, match="queue_maxsize is deprecated"):
        build_runner_config({"queue_maxsize": 4})
    exp = load_experiment(
        _base(
            algo="sac",
            actor={"name": "actor", "type": "squashed_gaussian", "action_dim": 1},
            critic={"name": "critic", "type": "twin_q", "action_dim": 1},
            train={"use_gpu_buffer": True},
        )
    )
    assert exp.runner.use_gpu_buffer is True


def test_td3_train_freq_is_a_real_field() -> None:
    assert build_algo_config("td3", {"train_freq": 4}).train_freq == 4
    assert TD3Config().train_freq == 1


def test_algo_inferred_from_filename(tmp_path: Path) -> None:
    import yaml

    cfg = _base()
    del cfg["algo"]
    path = tmp_path / "my_ppo_run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_experiment(path).algo == "ppo"
    assert load_experiment(cfg).algo is None  # dict, no hint: train keys still checked
    with pytest.raises(ConfigError, match="did you mean 'lr'"):
        load_experiment({**cfg, "train": {"lrr": 1}})
    with pytest.raises(ConfigError, match=r"algo: unknown algorithm 'spc' \(did you mean 'sac'"):
        load_experiment({**cfg, "algo": "spc"})
