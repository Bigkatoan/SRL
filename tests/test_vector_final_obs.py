"""Regression tests: auto-resetting vector envs must expose the TRUE final
observation (``info["final_obs"]``), and the training loops must store /
bootstrap from it rather than from the next episode's reset observation."""

from __future__ import annotations

import numpy as np
import pytest
from test_envlayer_helpers import (
    COUNTER_EP_LEN,
    CounterEnv,
    mlp_config,
    record_buffer_adds,
    run_cli,
)

from srl.envs import get_final_obs
from srl.envs.async_vector_env import AsyncVectorEnv
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper
from srl.envs.sync_vector_env import SyncVectorEnv


def _counter_fns(n: int, ep_lens=None):
    ep_lens = ep_lens or [COUNTER_EP_LEN] * n
    return [lambda L=L: GymnasiumWrapper(CounterEnv(ep_len=L)) for L in ep_lens]


@pytest.mark.parametrize("vec_cls", [SyncVectorEnv, AsyncVectorEnv])
def test_vector_env_reports_true_final_obs(vec_cls) -> None:
    env = vec_cls(_counter_fns(2, ep_lens=[COUNTER_EP_LEN, COUNTER_EP_LEN + 1]))
    try:
        env.reset(seed=0)
        actions = np.zeros((2, 1), dtype=np.float32)
        for _ in range(COUNTER_EP_LEN - 1):
            _, _, _, trunc, infos = env.step(actions)
            assert not trunc.any()
            assert all("final_obs" not in i for i in infos)
        next_obs, _, _, trunc, infos = env.step(actions)
        assert trunc.tolist() == [True, False]
        # Env 0 ended: next_obs is the reset frame, final_obs the terminal one.
        assert next_obs["state"][0, 0] == 0.0
        assert infos[0]["final_obs"]["state"][0] == COUNTER_EP_LEN
        assert "final_obs" not in infos[1]
        fixed = get_final_obs(next_obs, infos)
        assert fixed["state"][:, 0].tolist() == [COUNTER_EP_LEN, COUNTER_EP_LEN]
        # Caller's next_obs must not be mutated.
        assert next_obs["state"][0, 0] == 0.0
    finally:
        env.close()


def test_get_final_obs_isaac_style_mask() -> None:
    next_obs = {"policy": np.zeros((3, 2), dtype=np.float32)}
    final = {"policy": np.ones((3, 2), dtype=np.float32)}
    info = {"final_obs": final, "_final_obs": np.array([False, True, False])}
    out = get_final_obs(next_obs, info)
    assert out["policy"][:, 0].tolist() == [0.0, 1.0, 0.0]
    assert get_final_obs(next_obs, {}) is next_obs
    assert get_final_obs(next_obs, [{}, {}, {}]) is next_obs


class _FakePartialResetEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self) -> None:
        self.t = np.zeros(2)

    def reset(self, env_ids=None, **_):
        if env_ids is None:
            self.t[:] = 0
        else:
            self.t[np.asarray(env_ids)] = 0
        return {"policy": self.t[:, None].astype(np.float32).copy()}, {}

    def step(self, actions):
        self.t += np.array([1, 0.5])
        trunc = self.t >= 2
        return (
            {"policy": self.t[:, None].astype(np.float32).copy()},
            np.ones(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            trunc,
            {},
        )


def test_isaac_wrapper_uses_standard_final_obs_key() -> None:
    env = IsaacLabWrapper(_FakePartialResetEnv(), supports_partial_reset=True)
    env.reset()
    env.step(np.zeros((2, 1), dtype=np.float32))
    next_obs, _, _, trunc, info = env.step(np.zeros((2, 1), dtype=np.float32))
    assert trunc.tolist() == [True, False]
    assert info["_final_obs"].tolist() == [True, False]
    assert next_obs["policy"][0, 0] == 0.0
    assert get_final_obs(next_obs, info)["policy"][:, 0].tolist() == [2.0, 1.0]


def test_off_policy_stores_true_final_obs_on_truncation(monkeypatch, tmp_path) -> None:
    adds: list[dict] = []
    cfg = mlp_config("sac", total_steps=40, buffer_size=100)
    run_cli(monkeypatch, tmp_path, cfg, on_agent=record_buffer_adds(adds))
    assert len(adds) == 40
    obs_t = np.array([np.asarray(k["obs"]["state"])[0] for k in adds])
    next_t = np.array([np.asarray(k["next_obs"]["state"])[0] for k in adds])
    trunc = np.array([bool(np.asarray(k["truncated"]).reshape(-1)[0]) for k in adds])
    assert trunc.sum() == 40 // COUNTER_EP_LEN
    # Every stored transition advances the counter by exactly one -- in
    # particular truncated ones end at t == ep_len, not at the reset frame.
    np.testing.assert_allclose(next_t, obs_t + 1)
    assert np.all(next_t[trunc] == COUNTER_EP_LEN)


def test_on_policy_bootstraps_from_true_final_obs(monkeypatch, tmp_path) -> None:
    from srl.algorithms.ppo import PPO

    seen_t: list[float] = []
    original_predict = PPO.predict

    def _spy_predict(self, obs, *a, **k):
        first = next(iter(obs.values()))
        seen_t.extend(np.asarray(first.detach().cpu())[..., 0].reshape(-1).tolist())
        return original_predict(self, obs, *a, **k)

    monkeypatch.setattr(PPO, "predict", _spy_predict)
    cfg = mlp_config("ppo", total_steps=32)
    run_cli(monkeypatch, tmp_path, cfg)
    # Acting observations never reach t == ep_len (that frame is replaced
    # by the reset frame); only the truncation bootstrap sees it.
    assert COUNTER_EP_LEN in seen_t
