"""Checkpoint format v1, ``srl.load_policy`` and legacy-checkpoint compatibility."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

import srl
from srl import PPO, SAC, TD3, BaseAgent
from srl.core.agent_io import CheckpointFormatError, read_checkpoint
from srl.utils.checkpoint import CheckpointManager
from srl.utils.safe_serialization import LegacyCheckpointWarning

sys.path.insert(0, str(Path(__file__).parent))
from test_public_api import model_config  # noqa: E402

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def _obs_batch(n: int = 6, scale: float = 3.0) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32) * scale


def _trained(algo: str, steps: int = 1_000, **train):
    cfg = model_config(algo)
    cfg["train"].update(train)
    torch.manual_seed(0)
    agent = ALGOS[algo].from_config(cfg, env=gym.make("Pendulum-v1"))
    agent.learn(steps)
    return agent


def _assert_same_actions(agent, policy, obs: np.ndarray) -> None:
    expected, _ = agent.predict(obs, deterministic=True)
    np.testing.assert_array_equal(policy.predict(obs), expected)
    np.testing.assert_array_equal(policy(obs[0]), agent.predict(obs[0], deterministic=True)[0])


# ──────────────────────────────────────────────────────────────────────────────
# End to end: Python API
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_python_api_train_save_load_policy_matches_agent(algo, tmp_path) -> None:
    agent = _trained(algo, normalize_obs=True)
    out = agent.save(tmp_path / "agent")
    files = {p.name for p in out.iterdir()}
    assert {"meta.json", "config.yaml", "model.safetensors", "train_state.pt"} <= files
    assert "normalizers.json" in files
    assert "replay_buffer" not in files and "agent.pt" not in files

    meta = json.loads((out / "meta.json").read_text())
    assert meta["format"] == "srl-agent" and meta["format_version"] == 1
    assert meta["algo"] == algo and meta["torch_version"] == torch.__version__
    assert meta["action_spec"]["shape"] == [1]
    assert meta["action_spec"]["low"] == [-2.0] and meta["action_spec"]["high"] == [2.0]
    assert meta["obs_spec"]["obs_shape"] == [3]
    assert "obs" in meta["normalizers"]

    policy = srl.load_policy(out)
    assert isinstance(policy, srl.Policy)
    _assert_same_actions(agent, policy, _obs_batch())
    actions = policy(_obs_batch(64, scale=50.0))
    assert np.all(np.abs(actions) <= 2.0)

    # The full agent round-trips too (no env needed).
    again = BaseAgent.load(out, device="cpu")
    assert type(again) is type(agent) and again.num_timesteps == agent.num_timesteps
    _assert_same_actions(again, policy, _obs_batch())


def test_auto_dims_config_round_trips_without_env(tmp_path) -> None:
    agent = SAC.from_config(
        str(Path(__file__).resolve().parents[1] / "configs/envs/pendulum_sac.yaml"),
        env=gym.make("Pendulum-v1"),
        learning_starts=32,
        batch_size=16,
        buffer_size=1_000,
    )
    agent.learn(64)
    agent.save(tmp_path / "a")
    policy = srl.load_policy(tmp_path / "a")
    _assert_same_actions(agent, policy, _obs_batch())
    assert policy.experiment.model.encoders[0].input_dim == 3


def test_inference_only_checkpoint(tmp_path) -> None:
    agent = _trained("sac", steps=200)
    out = agent.save(tmp_path / "inf", include_train_state=False)
    assert not (out / "train_state.pt").exists()
    _assert_same_actions(agent, srl.load_policy(out), _obs_batch())
    loaded = SAC.load(out, device="cpu")
    for p, q in zip(
        loaded.target_model.parameters(), loaded.model.parameters(), strict=True
    ):  # target starts from the trained weights
        assert torch.equal(p, q)


def test_replay_buffer_only_on_request(tmp_path) -> None:
    agent = _trained("sac", steps=200)
    agent.save(tmp_path / "with", include_replay_buffer=True)
    assert (tmp_path / "with" / "replay_buffer" / "state.pt").exists()
    restored = SAC.load(tmp_path / "with", device="cpu")
    assert len(restored.buffer) == len(agent.buffer) > 0
    assert len(SAC.load(tmp_path / "with", device="cpu").buffer) == len(agent.buffer)

    # Periodic CheckpointManager saves leave it out unless asked.
    path = CheckpointManager(tmp_path / "ckpt").save(agent, step=200)
    assert not (path / "replay_buffer").exists()
    path = CheckpointManager(tmp_path / "ckpt2", include_replay_buffer=True).save(agent, step=200)
    assert (path / "replay_buffer" / "state.pt").exists()


def test_v1_loads_with_weights_only(tmp_path, monkeypatch) -> None:
    agent = _trained("sac", steps=200)
    agent.save(tmp_path / "a", include_replay_buffer=True)
    calls: list[bool] = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    SAC.load(tmp_path / "a", device="cpu")
    srl.load_policy(tmp_path / "a")
    assert calls and all(flag is True for flag in calls)


def test_checkpoint_manager_dir_restores_rng_and_step(tmp_path) -> None:
    import random

    agent = _trained("td3", steps=200)
    cm = CheckpointManager(tmp_path, max_keep=2)
    path = cm.save(agent, step=200, metrics={"eval/score_mean": -3.5})
    assert path.is_dir() and path.name == "ckpt_0000000200"
    first = cm.load(agent, path)
    draw = (random.random(), np.random.rand(), torch.rand(2))
    second = cm.load(agent, path)
    assert (random.random(), np.random.rand()) == draw[:2] and torch.equal(torch.rand(2), draw[2])
    assert first["step"] == second["step"] == 200
    assert first["metrics"] == {"eval/score_mean": -3.5}
    # Rotation deletes whole directories.
    cm.save(agent, step=300)
    cm.save(agent, step=400)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_0000000300", "ckpt_0000000400"]
    assert cm.latest() == tmp_path / "ckpt_0000000400"


def test_torchscript_export_matches_predict(tmp_path) -> None:
    agent = _trained("ppo", steps=500, normalize_obs=True)
    agent.save(tmp_path / "a")
    policy = srl.load_policy(tmp_path / "a")
    path = policy.export_torchscript(tmp_path / "policy.ts")
    module = torch.jit.load(str(path))
    obs = _obs_batch()
    np.testing.assert_allclose(
        module(torch.as_tensor(obs)).numpy(), policy.predict(obs), rtol=1e-5, atol=1e-5
    )


def test_policy_module_cli(tmp_path, capsys) -> None:
    from srl.policy import main

    _trained("td3", steps=100).save(tmp_path / "a")
    assert main(["info", str(tmp_path / "a")]) == 0
    assert json.loads(capsys.readouterr().out)["algo"] == "td3"
    assert main(["export", str(tmp_path / "a"), "-o", str(tmp_path / "p.ts")]) == 0
    assert (tmp_path / "p.ts").exists()


# ──────────────────────────────────────────────────────────────────────────────
# End to end: CLI
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_cli_run_dir_loads_as_policy(algo, tmp_path) -> None:
    from srl.cli.train import main

    cfg = model_config(algo)
    cfg["train"]["save_interval"] = 500
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg))
    argv = [
        "--config",
        str(config),
        "--env",
        "Pendulum-v1",
        "--steps",
        "1000",
        "--device",
        "cpu",
        "--logdir",
        str(tmp_path / "runs"),
        "--ckptdir",
        str(tmp_path / "ckpt"),
        "--no-plots",
        "--eval-freq",
        "500",
        "--eval-episodes",
        "1",
        "--save-best",
    ]
    assert main(argv) == 0
    run = tmp_path / "ckpt" / f"{algo}_cfg"
    names = sorted(p.name for p in run.iterdir())
    assert any(n.startswith("final_") for n in names) and any(n.startswith("best_") for n in names)
    assert any(n.startswith("ckpt_") for n in names)
    for name in names:  # every checkpoint is a v1 directory without a replay buffer
        assert json.loads((run / name / "meta.json").read_text())["format_version"] == 1
        assert not (run / name / "replay_buffer").exists()

    final = next(run.glob("final_*"))
    agent = BaseAgent.load(final, device="cpu")
    _assert_same_actions(agent, srl.load_policy(run, checkpoint="final"), _obs_batch())
    assert srl.load_policy(run).source.name.startswith("best_")  # auto: best first


# ──────────────────────────────────────────────────────────────────────────────
# Legacy formats
# ──────────────────────────────────────────────────────────────────────────────


def _write_v0_dir(agent, path: Path) -> None:
    """What ``agent.save`` wrote before format v1 (pickled ``agent.pt`` + yaml)."""
    path.mkdir(parents=True)
    payload = agent.checkpoint_payload()
    payload.pop("replay_buffer_state", None)
    torch.save(payload, path / "agent.pt")
    meta = dict(payload["srl_agent"])
    spec = meta.pop("model_spec")
    meta.pop("env_action_rescale")
    (path / "meta.json").write_text(
        json.dumps({"format": "srl-agent", "format_version": 0, **meta}, default=str)
    )
    (path / "model.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))


def test_legacy_v0_directory_loads_with_warning(tmp_path) -> None:
    agent = _trained("sac", steps=200)
    _write_v0_dir(agent, tmp_path / "v0")
    with pytest.warns(LegacyCheckpointWarning):
        loaded = SAC.load(tmp_path / "v0", device="cpu")
    with pytest.warns(LegacyCheckpointWarning):
        policy = srl.load_policy(tmp_path / "v0")
    _assert_same_actions(loaded, policy, _obs_batch())
    _assert_same_actions(agent, policy, _obs_batch())
    # Re-saving upgrades it to v1.
    loaded.save(tmp_path / "v1")
    assert json.loads((tmp_path / "v1" / "meta.json").read_text())["format_version"] == 1


def test_legacy_single_file_checkpoint_loads_with_warning(tmp_path) -> None:
    agent = _trained("ppo", steps=200)
    payload = agent.checkpoint_payload()
    payload.update(step=200, metrics={})
    torch.save(payload, tmp_path / "final_0000000200.pt")  # pre-v1 srl-train file
    with pytest.warns(LegacyCheckpointWarning):
        loaded = PPO.load(tmp_path / "final_0000000200.pt", device="cpu")
    with pytest.warns(LegacyCheckpointWarning):
        policy = srl.load_policy(tmp_path)  # a directory of legacy files
    _assert_same_actions(agent, policy, _obs_batch())
    _assert_same_actions(loaded, policy, _obs_batch())


# ──────────────────────────────────────────────────────────────────────────────
# Tampering / clear errors
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("v1") / "agent"
    _trained("td3", steps=100).save(path)
    return path


def _copy(saved: Path, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(saved, target)
    return target


def _edit_meta(path: Path, **changes) -> None:
    meta = json.loads((path / "meta.json").read_text())
    meta.update(changes)
    (path / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("missing", ["model.safetensors", "config.yaml"])
def test_missing_file_is_a_clear_error(saved, tmp_path, missing) -> None:
    path = _copy(saved, tmp_path)
    (path / missing).unlink()
    with pytest.raises(CheckpointFormatError, match=missing.replace(".", r"\.")):
        srl.load_policy(path)
    with pytest.raises(CheckpointFormatError):
        BaseAgent.load(path, device="cpu")


def test_newer_format_version_is_rejected(saved, tmp_path) -> None:
    path = _copy(saved, tmp_path)
    _edit_meta(path, format_version=99)
    with pytest.raises(CheckpointFormatError, match="format_version 99.*Upgrade SRL"):
        srl.load_policy(path)


@pytest.mark.parametrize("value", ["1", -1, None])
def test_invalid_format_version_is_rejected(saved, tmp_path, value) -> None:
    path = _copy(saved, tmp_path)
    _edit_meta(path, format_version=value)
    with pytest.raises(CheckpointFormatError, match="format_version"):
        read_checkpoint(path)


def test_foreign_format_is_rejected(saved, tmp_path) -> None:
    path = _copy(saved, tmp_path)
    _edit_meta(path, format="something-else")
    with pytest.raises(CheckpointFormatError, match="not an SRL agent checkpoint"):
        srl.load_policy(path)


def test_config_is_revalidated_on_load(saved, tmp_path) -> None:
    path = _copy(saved, tmp_path)
    cfg = yaml.safe_load((path / "config.yaml").read_text())
    cfg["train"]["learnin_rate"] = 1.0
    (path / "config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(CheckpointFormatError, match="learnin_rate"):
        srl.load_policy(path)


def test_weights_must_match_the_config(saved, tmp_path) -> None:
    from safetensors.torch import load_file, save_file

    path = _copy(saved, tmp_path)
    state = load_file(str(path / "model.safetensors"))
    state["unexpected.weight"] = torch.zeros(1)
    save_file(state, str(path / "model.safetensors"))
    with pytest.raises(CheckpointFormatError, match="do not match"):
        srl.load_policy(path)


def test_not_an_agent_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        srl.load_policy(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        srl.load_policy(tmp_path / "empty")
    with pytest.raises(CheckpointFormatError, match="not an SRL agent directory"):
        BaseAgent.load(tmp_path / "empty", device="cpu")
