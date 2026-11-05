"""auto dims, off-policy schedule aliases, registry plugins, resolved-config round trip."""

from __future__ import annotations

import dataclasses
import sys
import types
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces

from srl.config import (
    ConfigDeprecationWarning,
    ConfigError,
    build_algo_config,
    dump_resolved_config,
    load_experiment,
    normalize_offpolicy_schedule,
    resolve_auto_dims,
    resolve_auto_dims_from_env,
)
from srl.core.config import DDPGConfig, SACConfig, TD3Config
from srl.registry import registry as registry_mod
from srl.registry.builder import ModelBuilder
from srl.registry.config_schema import AgentModelConfig
from srl.registry.registry import (
    AlgorithmRegistry,
    AlgorithmSpec,
    EncoderRegistry,
    EnvRegistry,
    HeadRegistry,
    register_encoder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = sorted(
    p for p in (REPO_ROOT / "configs").rglob("*.yaml") if "benchmarks" not in p.parts
)


def _model(input_dim="auto", action_dim=None, critic_type="twin_q") -> dict:
    actor = {"name": "actor", "type": "squashed_gaussian"}
    critic = {"name": "critic", "type": critic_type}
    if action_dim is not None:
        actor["action_dim"] = action_dim
    enc = {"name": "state_enc", "type": "mlp", "latent_dim": 8, "layers": [8]}
    if input_dim is not None:
        enc["input_dim"] = input_dim
    return {
        "encoders": [enc],
        "flows": ["state_enc -> actor", "state_enc -> critic"],
        "actor": actor,
        "critic": critic,
    }


# ──────────────────────────────────────────────────────────────────────────────
# auto dims
# ──────────────────────────────────────────────────────────────────────────────

OBS = spaces.Box(-1, 1, shape=(11,), dtype=np.float32)
ACT = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)


@pytest.mark.parametrize("input_dim", ["auto", None])
def test_auto_and_omitted_dims_resolve_from_spaces(input_dim) -> None:
    cfg = AgentModelConfig.from_dict(_model(input_dim=input_dim))
    assert cfg.encoders[0].input_dim is None and cfg.actor.action_dim is None
    resolved = resolve_auto_dims(cfg, OBS, ACT)
    assert resolved.encoders[0].input_dim == 11
    assert resolved.actor.action_dim == 3 and resolved.critic.action_dim == 3
    assert cfg.encoders[0].input_dim is None  # input not mutated


def test_value_critic_gets_no_action_dim() -> None:
    resolved = resolve_auto_dims(_model(critic_type="value"), OBS, ACT)
    assert resolved.critic.action_dim is None


def test_explicit_dims_are_cross_checked() -> None:
    resolve_auto_dims(_model(input_dim=11, action_dim=3), OBS, ACT)  # consistent: fine
    with pytest.raises(ConfigError) as info:
        resolve_auto_dims(_model(input_dim=17, action_dim=6), OBS, ACT)
    text = str(info.value)
    assert "input_dim 17 does not match the environment observation size 11" in text
    assert "actor.action_dim 6 does not match the environment action dimension 3" in text


def test_unresolved_auto_dim_fails_before_forward() -> None:
    with pytest.raises(ConfigError, match="requires 'input_dim', which is 'auto'"):
        ModelBuilder.from_dict(_model())


def test_model_builder_resolves_and_forwards() -> None:
    model = ModelBuilder.from_dict(_model(), obs_space=OBS, act_space=ACT)
    out = model({"state": torch.zeros(2, 11)}, action=torch.zeros(2, 3))
    assert out["actor_out"]["action"].shape == (2, 3)


def test_dict_obs_with_input_name_and_cnn_hwc() -> None:
    obs = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
            "proprio": spaces.Box(-1, 1, shape=(2, 5), dtype=np.float32),
        }
    )
    cfg = {
        "encoders": [
            {
                "name": "cam",
                "type": "cnn",
                "input_name": "rgb",
                "input_shape": "auto",
                "layers": [[8, 3, 2]],
            },
            {"name": "body", "type": "mlp", "input_name": "proprio"},
        ],
        "flows": ["cam -> actor", "body -> actor"],
        "actor": {"name": "actor", "type": "gaussian"},
    }
    resolved = resolve_auto_dims(cfg, obs, spaces.Box(-1, 1, shape=(2,)))
    assert resolved.encoders[0].input_shape == [3, 64, 64]  # HWC -> CHW like the wrappers
    assert resolved.encoders[1].input_dim == 10
    assert resolved.actor.action_dim == 2
    with pytest.raises(ConfigError, match="Missing observation key 'missing'"):
        bad = dict(
            cfg, encoders=[dict(cfg["encoders"][1], input_name="missing")], flows=["body -> actor"]
        )
        resolve_auto_dims(bad, obs, None)


def test_resolve_from_batched_env() -> None:
    env = types.SimpleNamespace(
        num_envs=4,
        obs_space=spaces.Box(-1, 1, shape=(4, 11)),
        act_space=spaces.Box(-1, 1, shape=(4, 3)),
        single_act_space=ACT,
    )
    resolved = resolve_auto_dims_from_env(_model(), env)
    assert resolved.encoders[0].input_dim == 11 and resolved.actor.action_dim == 3


# ──────────────────────────────────────────────────────────────────────────────
# off-policy schedule aliases
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_schedule_defaults_and_canonical_names() -> None:
    s = normalize_offpolicy_schedule(SACConfig())
    assert (s.learning_starts, s.update_after, s.train_freq, s.gradient_steps) == (
        10_000,
        10_000,
        1,
        1,
    )
    s = normalize_offpolicy_schedule(
        TD3Config(learning_starts=500, update_after=100, train_freq=4, gradient_steps=2)
    )
    assert (s.learning_starts, s.update_after, s.train_freq, s.gradient_steps) == (500, 100, 4, 2)


def test_normalize_schedule_legacy_aliases_keep_semantics() -> None:
    cfg = DDPGConfig(learning_starts=7, start_steps=3, update_every=5, train_freq=2)
    with pytest.warns(ConfigDeprecationWarning):
        s = normalize_offpolicy_schedule(cfg)
    # start_steps drives the warmup; update_after falls back to learning_starts.
    assert (s.learning_starts, s.update_after, s.train_freq) == (3, 7, 5)


def test_build_algo_config_folds_aliases() -> None:
    with pytest.warns(ConfigDeprecationWarning, match="start_steps is deprecated"):
        cfg = build_algo_config("sac", {"start_steps": 1000, "update_every": 50})
    assert cfg.start_steps is None and cfg.update_every is None
    assert (cfg.learning_starts, cfg.update_after, cfg.train_freq) == (1000, 10_000, 50)
    with pytest.raises(ConfigError, match="both train_freq .* update_every"):
        build_algo_config("td3", {"train_freq": 2, "update_every": 50})


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: p.name)
def test_strict_builder_matches_legacy_cli_builder(path: Path) -> None:
    """build_algo_config() yields the same effective config the CLI used to build."""
    from srl.cli.train import _build_algo_config, _resolve_algo_config_cls

    exp = load_experiment(path)
    raw = yaml.safe_load(path.read_text())
    train = raw.get("train") or {}
    spec = AlgorithmRegistry.get(exp.algo)
    legacy_cls, extra = _resolve_algo_config_cls(exp.algo, spec.config_cls, raw, train)
    legacy = _build_algo_config(legacy_cls, train, **extra)
    assert type(legacy) is type(exp.algo_config)
    if spec.off_policy:
        assert normalize_offpolicy_schedule(legacy) == normalize_offpolicy_schedule(exp.algo_config)
    skip = {"start_steps", "update_after", "update_every", "learning_starts", "train_freq"}
    for f in dataclasses.fields(legacy):
        if f.name not in skip:
            assert getattr(legacy, f.name) == getattr(exp.algo_config, f.name), f.name


# ──────────────────────────────────────────────────────────────────────────────
# registry
# ──────────────────────────────────────────────────────────────────────────────


def test_builtins_live_in_the_registry() -> None:
    assert {"mlp", "cnn", "lstm", "text"} <= set(EncoderRegistry.available())
    assert {"gaussian", "squashed_gaussian", "deterministic", "value", "twin_q", "q"} <= set(
        HeadRegistry.available()
    )
    assert {"ppo", "a2c", "a3c", "sac", "ddpg", "td3"} <= set(AlgorithmRegistry.available())
    assert {"flat", "goal", "racecar", "isaaclab", "mjlab"} <= set(EnvRegistry.available())
    assert EncoderRegistry.entry("mlp").source == "builtin"
    with pytest.raises(KeyError, match="already registered"):
        register_encoder("mlp", lambda cfg: None)


def test_user_registered_encoder_builds() -> None:
    @register_encoder("unit_test_identity", extra_keys=("scale",))
    class Identity(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.scale = cfg.extra.get("scale", 1.0)
            self.proj = torch.nn.Linear(cfg.input_dim, cfg.latent_dim)

        def forward(self, x):
            return self.proj(x) * self.scale

    try:
        model = ModelBuilder.from_dict(
            {
                "encoders": [
                    {
                        "name": "e",
                        "type": "unit_test_identity",
                        "input_dim": 3,
                        "latent_dim": 4,
                        "scale": 2.0,
                    }
                ],
                "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
            }
        )
        assert model.encoders["e"].scale == 2.0
    finally:
        EncoderRegistry.unregister("unit_test_identity")


@pytest.fixture
def fake_plugins(monkeypatch):
    """Install a fake distribution exposing srl.* entry points."""
    module = types.ModuleType("srl_fake_plugin")

    class PluginEncoder(torch.nn.Module):
        srl_requires = ("input_dim",)

        def __init__(self, cfg):
            super().__init__()
            self.net = torch.nn.Linear(cfg.input_dim, cfg.latent_dim)

        def forward(self, x):
            return self.net(x)

    def make_env(env_id, *, device="cpu", n_envs=1, render=False):
        return ("fake-env", env_id)

    module.PluginEncoder = PluginEncoder
    module.make_env = make_env
    module.ALGO = AlgorithmSpec(
        "fakealgo",
        SACConfig,
        "srl.algorithms.sac:SAC",
        off_policy=True,
        actor_types=("squashed_gaussian",),
        critic_types=("twin_q",),
    )
    monkeypatch.setitem(sys.modules, "srl_fake_plugin", module)
    eps = {
        "srl.encoders": [
            metadata.EntryPoint("plugin_enc", "srl_fake_plugin:PluginEncoder", "srl.encoders")
        ],
        "srl.envs": [metadata.EntryPoint("fakesim", "srl_fake_plugin:make_env", "srl.envs")],
        "srl.algorithms": [
            metadata.EntryPoint("fakealgo", "srl_fake_plugin:ALGO", "srl.algorithms")
        ],
        "srl.heads": [metadata.EntryPoint("broken_head", "srl_fake_plugin:Missing", "srl.heads")],
    }
    monkeypatch.setattr(registry_mod, "_entry_points", lambda group: eps.get(group, []))
    regs = (EncoderRegistry, HeadRegistry, AlgorithmRegistry, EnvRegistry)
    for reg in regs:
        reg.refresh_plugins()
    yield module
    for reg in regs:
        for name in ("plugin_enc", "fakesim", "fakealgo", "broken_head"):
            reg.unregister(name)
        reg.refresh_plugins()


def test_entry_point_plugins_are_discovered_lazily(fake_plugins) -> None:
    # listed without being loaded
    assert "plugin_enc" in EncoderRegistry.available()
    assert "plugin_enc" not in EncoderRegistry._store
    model = ModelBuilder.from_dict(
        {
            "encoders": [{"name": "e", "type": "plugin_enc", "input_dim": 3, "latent_dim": 4}],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 1},
        }
    )
    assert isinstance(model.encoders["e"], fake_plugins.PluginEncoder)
    assert EncoderRegistry.entry("plugin_enc").source == "entry-point"
    assert EncoderRegistry.entry("plugin_enc").requires == ("input_dim",)
    # plugin env type + algorithm work through load_experiment
    exp = load_experiment(
        {
            "algo": "fakealgo",
            "env_type": "fakesim",
            "env_id": "X-v0",
            "encoders": [{"name": "e", "type": "plugin_enc", "input_dim": 3}],
            "actor": {"name": "actor", "type": "squashed_gaussian", "action_dim": 1},
            "train": {"batch_size": 32},
        }
    )
    assert exp.algo == "fakealgo" and exp.algo_config.batch_size == 32
    assert EnvRegistry.get("fakesim")("X-v0") == ("fake-env", "X-v0")
    # a broken plugin fails loudly, naming the entry point
    with pytest.raises(ImportError, match="broken_head.*srl_fake_plugin:Missing"):
        HeadRegistry.get("broken_head")


# ──────────────────────────────────────────────────────────────────────────────
# resolved-config dump round trip
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: p.name)
def test_dump_round_trip(path: Path, tmp_path: Path) -> None:
    exp = load_experiment(path)
    out = tmp_path / "run" / "config.yaml"
    text = dump_resolved_config(exp, out)
    assert out.read_text() == text
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigDeprecationWarning)
        again = load_experiment(out)
    assert again == exp
    # stable: dumping the reloaded config gives the same text
    assert dump_resolved_config(again) == text


def test_dump_round_trip_after_runtime_overrides(tmp_path: Path) -> None:
    exp = load_experiment(REPO_ROOT / "configs/envs/halfcheetah_sac.yaml")
    model = resolve_auto_dims(exp.model, spaces.Box(-1, 1, (17,)), spaces.Box(-1, 1, (6,)))
    final = dataclasses.replace(
        exp,
        model=model,
        algo_config=dataclasses.replace(
            exp.algo_config, action_dim=6, replay_num_envs=4, lr_actor=1e-5, her_reward_fn=print
        ),
    )
    path = tmp_path / "c.yaml"
    dump_resolved_config(final, path)
    assert "her_reward_fn" not in path.read_text()
    again = load_experiment(path)
    assert again == final
    assert again.algo_config.lr_actor == 1e-5 and again.algo_config.action_dim == 6
