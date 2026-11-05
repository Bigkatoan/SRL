"""Public Python API: ``from_config`` / ``learn`` / ``predict`` / ``save`` / ``load``,
callbacks and early stopping, for every algorithm, on a tiny Pendulum run."""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch

from srl import A2C, A3C, DDPG, PPO, SAC, TD3, BaseAgent
from srl.envs.action_rescale import RescaleActionWrapper
from srl.training import Callback, CheckpointCallback, EarlyStopping, EvalCallback

OBS_DIM, ACT_DIM = 3, 1  # Pendulum-v1; action bounds [-2, 2]

_ENC = [
    {
        "name": "state_enc",
        "type": "mlp",
        "input_dim": OBS_DIM,
        "latent_dim": 16,
        "layers": [{"out_features": 16, "activation": "tanh"}],
    }
]
_FLOWS = ["state_enc -> actor", "state_enc -> critic"]
_ON_POLICY_TRAIN = {"n_steps": 16, "batch_size": 16, "n_epochs": 1}
_OFF_POLICY_TRAIN = {
    "buffer_size": 500,
    "batch_size": 8,
    "learning_starts": 16,
    "train_freq": 4,
    "gradient_steps": 1,
}


def model_config(algo: str) -> dict:
    if algo in ("ppo", "a2c", "a3c"):
        actor = {"name": "actor", "type": "gaussian", "action_dim": ACT_DIM, "log_std_init": -0.5}
        critic = {"name": "critic", "type": "value"}
        train = dict(_ON_POLICY_TRAIN)
        if algo == "a3c":
            train = {"n_steps": 8, "n_envs": 1}
    else:
        actor_type = "squashed_gaussian" if algo == "sac" else "deterministic"
        actor = {"name": "actor", "type": actor_type, "action_dim": ACT_DIM}
        critic = {"name": "critic", "type": "q" if algo == "ddpg" else "twin_q"}
        critic["action_dim"] = ACT_DIM
        train = dict(_OFF_POLICY_TRAIN)
    return {
        "algo": algo,
        "encoders": _ENC,
        "flows": _FLOWS,
        "actor": actor,
        "critic": critic,
        "train": train,
    }


ALGOS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG}


def _obs(seed: int = 0) -> np.ndarray:
    obs, _ = gym.make("Pendulum-v1").reset(seed=seed)
    return obs


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_learn_predict_save_load_round_trip(algo, tmp_path) -> None:
    cls = ALGOS[algo]
    torch.manual_seed(0)
    agent = cls.from_config(model_config(algo), env=gym.make("Pendulum-v1"))
    assert isinstance(agent, cls)
    # Raw gym env wrapped with SRL's wrapper + action rescaling automatically.
    assert isinstance(agent.env, RescaleActionWrapper)

    assert agent.learn(64) is agent
    assert agent.num_timesteps >= 64

    obs = _obs()
    action, state = agent.predict(obs, deterministic=True)
    assert isinstance(action, np.ndarray) and action.shape == (ACT_DIM,)
    assert state is None or isinstance(state, dict)
    batch, _ = agent.predict(np.stack([obs, _obs(1), _obs(2)]), deterministic=True)
    assert batch.shape == (3, ACT_DIM)
    np.testing.assert_allclose(batch[0], action, atol=1e-5)

    agent.save(tmp_path / "agent")
    assert (tmp_path / "agent" / "agent.pt").exists()
    assert (tmp_path / "agent" / "model.yaml").exists()
    loaded = cls.load(tmp_path / "agent", env=gym.make("Pendulum-v1"))
    assert type(loaded) is cls
    assert loaded.num_timesteps == agent.num_timesteps
    np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], action, atol=1e-5)

    # BaseAgent.load dispatches on the stored algorithm; no env needed to predict.
    generic = BaseAgent.load(tmp_path / "agent")
    assert type(generic) is cls
    np.testing.assert_allclose(generic.predict(obs, deterministic=True)[0], action, atol=1e-5)


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_stochastic_actions_stay_within_env_bounds(algo) -> None:
    cls = ALGOS[algo]
    agent = cls.from_config(model_config(algo), env=gym.make("Pendulum-v1"))
    obs = np.random.default_rng(0).normal(size=(256, OBS_DIM)).astype(np.float32) * 5.0
    for deterministic in (True, False):
        actions, _ = agent.predict(obs, deterministic=deterministic)
        assert actions.shape == (256, ACT_DIM)
        assert np.all(actions >= -2.0 - 1e-5) and np.all(actions <= 2.0 + 1e-5)
    # The [-1, 1] model domain is mapped onto the whole [-2, 2] env range.
    with torch.no_grad():
        for p in agent.model.actor.parameters():
            p.fill_(0.0)
        bias = getattr(agent.model.actor, "mean_head", None) or agent.model.actor
        for m in bias.modules():
            if isinstance(m, torch.nn.Linear):
                m.bias.fill_(5.0)
    high, _ = agent.predict(obs[:4], deterministic=True)
    np.testing.assert_allclose(high, 2.0, atol=1e-3)


def test_predict_accepts_wrapper_dict_obs_and_legacy_tensor_api_warns() -> None:
    agent = SAC.from_config(model_config("sac"), env=gym.make("Pendulum-v1"))
    obs = _obs()
    a_np, _ = agent.predict(obs, deterministic=True)
    a_dict, _ = agent.predict({"state": obs}, deterministic=True)
    np.testing.assert_allclose(a_np, a_dict)

    tensor_obs = {"state_enc": torch.as_tensor(obs[None], dtype=torch.float32)}
    with pytest.warns(DeprecationWarning, match="act"):
        legacy = agent.predict(tensor_obs, deterministic=True)
    assert len(legacy) == 4  # (action, log_prob, value, hidden)
    low_level = agent.act(tensor_obs, deterministic=True)
    torch.testing.assert_close(legacy[0], low_level[0])
    # act() stays in the model's normalised domain; predict() is in env units.
    np.testing.assert_allclose(low_level[0].numpy()[0] * 2.0, a_np, atol=1e-5)


def test_constructor_api_without_target_model_and_env_binding() -> None:
    from srl.core.config import SACConfig
    from srl.registry.builder import ModelBuilder

    model = ModelBuilder.from_dict(model_config("sac"))
    agent = SAC(model, SACConfig(batch_size=8, learning_starts=8), env=gym.make("Pendulum-v1"))
    assert agent.cfg.action_dim == ACT_DIM  # filled in from the env
    assert agent.target_model is not agent.model
    agent.learn(24)
    # Positional legacy form still works.
    legacy = SAC(ModelBuilder.from_dict(model_config("sac")), None, SACConfig(), "cpu")
    assert legacy.cfg.action_dim == 0


def test_learn_without_env_raises_helpful_error() -> None:
    from srl.registry.builder import ModelBuilder

    agent = PPO(ModelBuilder.from_dict(model_config("ppo")))
    with pytest.raises(ValueError, match="needs an environment"):
        agent.learn(10)


def test_from_config_builds_env_from_yaml_and_dispatches_on_algo() -> None:
    cfg = model_config("td3")
    cfg["env_id"] = "Pendulum-v1"
    agent = BaseAgent.from_config(cfg)
    assert isinstance(agent, TD3)
    assert agent.env is not None
    with pytest.raises(TypeError, match="TD3"):
        SAC.from_config(cfg)  # the YAML declares a different algorithm


def test_from_config_rejects_incompatible_heads() -> None:
    from srl.training import ConfigurationError

    cfg = model_config("ppo")
    del cfg["algo"]
    with pytest.raises(ConfigurationError, match="not compatible"):
        SAC.from_config(cfg, env=gym.make("Pendulum-v1"))


def test_shipped_pendulum_sac_config_quickstart(tmp_path) -> None:
    agent = SAC.from_config(
        "configs/envs/pendulum_sac.yaml", env=gym.make("Pendulum-v1"), learning_starts=32
    )
    agent.learn(64)
    action, _ = agent.predict(_obs(), deterministic=True)
    assert action.shape == (1,) and -2.0 <= float(action[0]) <= 2.0
    agent.save(tmp_path / "pendulum")
    again = SAC.load(tmp_path / "pendulum", env=gym.make("Pendulum-v1"))
    np.testing.assert_allclose(again.predict(_obs(), deterministic=True)[0], action, atol=1e-5)


def test_load_cli_style_checkpoint_file(tmp_path) -> None:
    from srl.utils.checkpoint import CheckpointManager

    agent = PPO.from_config(model_config("ppo"), env=gym.make("Pendulum-v1"))
    agent.learn(32)
    path = CheckpointManager(tmp_path).save(agent, step=32, tag="final")
    loaded = PPO.load(path)
    obs = _obs()
    np.testing.assert_allclose(
        loaded.predict(obs, deterministic=True)[0],
        agent.predict(obs, deterministic=True)[0],
        atol=1e-5,
    )


def test_save_with_replay_buffer_and_inplace_load(tmp_path) -> None:
    agent = SAC.from_config(model_config("sac"), env=gym.make("Pendulum-v1"))
    agent.learn(40)
    agent.save(tmp_path / "a", include_replay_buffer=True)
    fresh = SAC.from_config(model_config("sac"), env=gym.make("Pendulum-v1"))
    assert len(fresh.buffer) == 0
    assert fresh.load(tmp_path / "a") is fresh  # instance call restores in place
    assert len(fresh.buffer) == len(agent.buffer) > 0
    with pytest.warns(DeprecationWarning):
        agent.save(tmp_path / "legacy.pt")


def test_a3c_learn_delegates_to_worker_training(tmp_path) -> None:
    cfg = model_config("a3c")
    cfg["train"]["n_envs"] = 1
    agent = A3C.from_config(cfg, env=gym.make("Pendulum-v1"))
    agent.learn(32)
    assert agent.num_timesteps >= 32
    action, _ = agent.predict(_obs(), deterministic=True)
    assert action.shape == (1,) and -2.0 <= float(action[0]) <= 2.0
    agent.save(tmp_path / "a3c")
    assert isinstance(A3C.load(tmp_path / "a3c"), A3C)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────


class _Recorder(Callback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_training_start(self, trainer) -> None:
        self.events.append("training_start")

    def on_step(self, step) -> None:
        self.events.append("step")

    def on_rollout_end(self, step) -> None:
        self.events.append("rollout_end")

    def on_update(self, step, metrics) -> None:
        self.events.append("update")

    def on_eval(self, step, metrics) -> None:
        assert "eval/score_mean" in metrics
        self.events.append("eval")

    def on_training_end(self, trainer) -> None:
        self.events.append("training_end")


def _dedupe(events: list[str]) -> list[str]:
    out: list[str] = []
    for e in events:
        if not out or out[-1] != e:
            out.append(e)
    return out


def test_on_policy_callbacks_fire_in_order() -> None:
    agent = PPO.from_config(model_config("ppo"), env=gym.make("Pendulum-v1"))
    rec = _Recorder()
    agent.learn(
        32, eval_env=gym.make("Pendulum-v1"), eval_freq=16, n_eval_episodes=1, callbacks=rec
    )
    ev = _dedupe(rec.events)
    assert ev[0] == "training_start" and ev[-1] == "training_end"
    # Each 16-step rollout: steps -> rollout_end -> update -> eval.
    assert ev[1:5] == ["step", "rollout_end", "update", "eval"]
    assert rec.events.count("rollout_end") == 2
    assert rec.events.count("step") == 32


def test_off_policy_callbacks_fire_in_order() -> None:
    agent = SAC.from_config(model_config("sac"), env=gym.make("Pendulum-v1"))
    rec = _Recorder()
    agent.learn(
        40, eval_env=gym.make("Pendulum-v1"), eval_freq=20, n_eval_episodes=1, callbacks=rec
    )
    ev = rec.events
    assert ev[0] == "training_start" and ev[-1] == "training_end"
    assert ev.count("step") == 40
    assert "update" in ev and "eval" in ev and "rollout_end" not in ev
    assert ev.index("step") < ev.index("update") < ev.index("eval")


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_early_stopping_actually_stops_training(algo) -> None:
    cls = ALGOS[algo]
    agent = cls.from_config(model_config(algo), env=gym.make("Pendulum-v1"))
    # min_delta so large nothing ever counts as an improvement after the first eval.
    stopper = EarlyStopping(patience=1, min_delta=1e9)
    agent.learn(
        2_000,
        eval_env=gym.make("Pendulum-v1"),
        eval_freq=32,
        n_eval_episodes=1,
        callbacks=[stopper],
    )
    assert stopper.should_stop
    assert stopper.monitor == "eval/score_mean"
    assert agent.num_timesteps < 200, agent.num_timesteps


def test_early_stopping_legacy_monitor_key_is_fixed() -> None:
    with pytest.warns(DeprecationWarning):
        stopper = EarlyStopping(monitor="eval/mean_reward")
    assert stopper.monitor == "eval/score_mean"


def test_eval_callback_feeds_early_stopping_and_saves_best(tmp_path) -> None:
    agent = PPO.from_config(model_config("ppo"), env=gym.make("Pendulum-v1"))
    stopper = EarlyStopping(patience=1, min_delta=1e9)
    evaluator = EvalCallback(
        gym.make("Pendulum-v1"),
        eval_freq=16,
        n_eval_episodes=1,
        best_model_save_path=str(tmp_path / "best"),
    )
    agent.learn(2_000, callbacks=[evaluator, stopper])
    assert stopper.should_stop
    assert agent.num_timesteps < 200
    assert len(evaluator.evaluations) >= 2
    assert (tmp_path / "best" / "agent.pt").exists()


def test_checkpoint_callback_through_learn(tmp_path) -> None:
    agent = PPO.from_config(model_config("ppo"), env=gym.make("Pendulum-v1"))
    cb = CheckpointCallback(tmp_path / "ckpt", save_interval=16)
    agent.learn(48, callbacks=cb)
    saved = sorted((tmp_path / "ckpt").glob("ckpt_*.pt"))
    assert len(saved) == 3
    restored = PPO.load(saved[-1])
    assert restored.num_timesteps == agent.num_timesteps


def test_callback_returning_false_stops() -> None:
    class _StopAfterTen(Callback):
        def on_step(self, step):
            return step < 10

    agent = SAC.from_config(model_config("sac"), env=gym.make("Pendulum-v1"))
    agent.learn(500, callbacks=_StopAfterTen())
    assert agent.num_timesteps == 10


def test_learn_continues_step_counter_when_not_reset() -> None:
    agent = PPO.from_config(model_config("ppo"), env=gym.make("Pendulum-v1"))
    agent.learn(16)
    agent.learn(16, reset_num_timesteps=False)
    assert agent.num_timesteps == 32


def test_no_deprecation_warnings_on_the_new_api(tmp_path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        agent = TD3.from_config(model_config("td3"), env=gym.make("Pendulum-v1"))
        agent.learn(24)
        agent.predict(_obs())
        agent.save(tmp_path / "td3")
        TD3.load(tmp_path / "td3")
