"""Every shipped YAML config must parse, build, and run a forward pass.

This is a cheap guard against configs silently rotting as the model builder
evolves: a typo'd layer key, a stale ``action_dim`` or a removed head type is
caught here instead of minutes into an ``srl-train`` run.

Two layers of checks:

* ``test_config_builds_and_forwards`` -- pure model check, no simulator
  needed. Builds the model with :class:`ModelBuilder` and pushes dummy
  observations (shaped from each encoder's ``input_dim``/``input_shape``)
  through it.
* ``test_config_matches_env_spaces`` -- constructs the real environment the
  config targets (via the same factory ``srl-train`` uses) and checks the
  observation/action dims agree with the YAML. Skipped, with the reason, when
  the env's optional runtime (MuJoCo, Box2D, gymnasium-robotics, Isaac Lab,
  racecar_gym, ...) is not installed.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import yaml

from srl.registry.builder import ModelBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
# configs/benchmarks/*.yaml are score-threshold tables for the matrix
# benchmark script, not model configs -- validated separately below.
BENCHMARK_DIR = CONFIG_DIR / "benchmarks"

MODEL_CONFIGS = sorted(p for p in CONFIG_DIR.rglob("*.yaml") if BENCHMARK_DIR not in p.parents)
BENCHMARK_CONFIGS = sorted(BENCHMARK_DIR.glob("*.yaml"))

_COMPATIBLE_HEADS = {
    "ppo": ({"gaussian"}, {"value"}),
    "a2c": ({"gaussian"}, {"value"}),
    "a3c": ({"gaussian"}, {"value"}),
    "sac": ({"squashed_gaussian"}, {"twin_q"}),
    "ddpg": ({"deterministic"}, {"q", "q_function", "twin_q"}),
    "td3": ({"deterministic"}, {"twin_q"}),
}


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(CONFIG_DIR)) for p in paths]


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def _dummy_obs(raw: dict, batch: int = 2) -> dict[str, torch.Tensor]:
    obs: dict[str, torch.Tensor] = {}
    for enc in raw.get("encoders") or []:
        enc_type = str(enc.get("type", "mlp")).lower()
        name = enc["name"]
        if enc_type == "text":
            vocab = int(enc.get("vocab_size", 128))
            seq_len = int(enc.get("max_len", 16))
            obs[name] = torch.randint(0, max(vocab, 2), (batch, seq_len))
        elif "input_shape" in enc:
            obs[name] = torch.randn(batch, *[int(d) for d in enc["input_shape"]])
        else:
            obs[name] = torch.randn(batch, int(enc["input_dim"]))
    return obs


def test_configs_were_discovered() -> None:
    assert len(MODEL_CONFIGS) >= 20, MODEL_CONFIGS


@pytest.mark.parametrize("config_path", MODEL_CONFIGS, ids=_ids(MODEL_CONFIGS))
def test_config_builds_and_forwards(config_path: Path) -> None:
    raw = _load(config_path)
    assert raw.get("encoders"), "config declares no encoders"
    assert raw.get("actor"), "config declares no actor head"

    model = ModelBuilder.from_yaml(str(config_path))
    model.eval()

    algo = str(raw.get("algo") or "").lower()
    if algo:
        assert algo in _COMPATIBLE_HEADS, f"unknown algo {algo!r}"
        actors, critics = _COMPATIBLE_HEADS[algo]
        assert raw["actor"].get("type") in actors
        assert (raw.get("critic") or {}).get("type") in critics

    obs = _dummy_obs(raw)
    critic_cfg = raw.get("critic") or {}
    action = None
    if critic_cfg.get("action_dim"):
        action = torch.zeros(2, int(critic_cfg["action_dim"]))
    with torch.no_grad():
        out = model(obs, action=action)

    actor_out = out["actor_out"]
    assert actor_out is not None
    act = actor_out["action"] if isinstance(actor_out, dict) else actor_out
    assert act.shape == (2, int(raw["actor"]["action_dim"]))
    assert torch.isfinite(act).all()


_ENV_RUNTIME = {
    # env_type -> (importable module, human-readable install hint)
    "isaaclab": ("isaaclab", "Isaac Lab is not installed"),
    "mjlab": ("mjlab", "mjlab is not installed"),
    "racecar": ("racecar_gym", "racecar_gym is not installed (see installation docs)"),
    "goal": ("gymnasium_robotics", "pip install 'srl-rl[robotics]'"),
}


_MUJOCO_ENV_KEYWORDS = (
    "cheetah",
    "ant",
    "hopper",
    "walker",
    "humanoid",
    "swimmer",
    "pusher",
    "reacher",
)


def _skip_reason_for(raw: dict) -> str | None:
    env_type = str(raw.get("env_type") or "flat").lower()
    env_id = str(raw.get("env_id") or "")
    if not env_id:
        return "config declares no env_id (model-only config)"
    if env_type in ("isaaclab", "mjlab"):
        # GPU-batched simulators: far too heavy for a unit test even when
        # installed. Covered by tests/test_isaac_lab_wrapper.py & friends.
        return f"{env_type} env construction is out of scope for unit tests"
    module, hint = _ENV_RUNTIME.get(env_type, (None, None))
    if module and importlib.util.find_spec(module) is None:
        return hint
    if env_type == "flat":
        lowered = env_id.lower()
        if any(k in lowered for k in ("bipedal", "lunar", "carracing")):
            if importlib.util.find_spec("Box2D") is None:
                return "Box2D is not installed: pip install 'srl-rl[box2d]'"
        if any(k in lowered for k in _MUJOCO_ENV_KEYWORDS):
            if importlib.util.find_spec("mujoco") is None:
                return "MuJoCo is not installed: pip install 'srl-rl[mujoco]'"
    return None


ENV_CONFIGS = sorted((CONFIG_DIR / "envs").glob("*.yaml"))


@pytest.mark.parametrize("config_path", ENV_CONFIGS, ids=_ids(ENV_CONFIGS))
def test_config_matches_env_spaces(config_path: Path) -> None:
    import numpy as np

    from srl.training.envs import env_action_dim as _resolve_env_action_dim
    from srl.training.envs import make_env as _make_cli_env
    from srl.training.envs import resolve_env_spec as _resolve_env_spec

    raw = _load(config_path)
    reason = _skip_reason_for(raw)
    if reason:
        pytest.skip(reason)

    env_name, env_type = _resolve_env_spec(None, raw)
    env = _make_cli_env(env_name, "cpu", 1, env_type)
    try:
        obs, _ = env.reset(seed=0)
        assert _resolve_env_action_dim(env) == int(raw["actor"]["action_dim"])
        assert len(obs) == 1, f"expected single-key obs dict, got {list(obs)}"
        obs_shape = tuple(np.asarray(next(iter(obs.values()))).shape)
        for enc in raw["encoders"]:
            if "input_shape" in enc:
                assert obs_shape == tuple(enc["input_shape"]), enc["name"]
            else:
                assert math.prod(obs_shape) == int(enc["input_dim"]), enc["name"]
    finally:
        env.close()


@pytest.mark.parametrize("config_path", BENCHMARK_CONFIGS, ids=_ids(BENCHMARK_CONFIGS))
def test_benchmark_targets_reference_real_configs(config_path: Path) -> None:
    targets = _load(config_path)
    env_stems = {p.stem for p in ENV_CONFIGS}
    for key, target in targets.items():
        assert key in env_stems, f"{key!r} has no matching configs/envs/{key}.yaml"
        assert isinstance(target, dict) and "eval_score_min" in target, key
        float(target["eval_score_min"])
