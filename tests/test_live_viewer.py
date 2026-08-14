"""srl-train --visualize: the live background-viewer feature.

Covers what's safe and meaningful to assert in CI: the CLI flag itself,
the mjlab policy adapter's wiring (remap -> device -> agent.predict), each
backend's graceful-failure behavior when it can't start (a bad task name, an
env that fails to construct), and that `.stop()` actually ends the
background loop (rather than relying on the process dying, which can
segfault a daemon thread mid-GPU-call -- see live_viewer.py's module
docstring). Does NOT start a real background render loop against a real
display or a real Viser web server -- those are "runs forever until told to
stop" by design and are better verified by hand than as a CI unit test.
"""

from __future__ import annotations

import threading
import time

import torch

from srl.cli import train as train_module
from srl.cli.train import _build_parser
from srl.utils import live_viewer as live_viewer_module
from srl.utils.live_viewer import (
    VisualizerHandle,
    _MjlabPolicyAdapter,
    start_gym_visualizer,
    start_mjlab_visualizer,
)


def test_visualize_flag_defaults_false_and_parses() -> None:
    args = _build_parser().parse_args(["--config", "cfg.yaml"])
    assert args.visualize is False

    args = _build_parser().parse_args(["--config", "cfg.yaml", "--visualize"])
    assert args.visualize is True


def test_visualize_backend_defaults_viser_and_parses_native() -> None:
    args = _build_parser().parse_args(["--config", "cfg.yaml"])
    assert args.visualize_backend == "viser"

    args = _build_parser().parse_args(
        ["--config", "cfg.yaml", "--visualize", "--visualize-backend", "native"]
    )
    assert args.visualize_backend == "native"


class _FakeAgent:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.predict_calls: list[dict] = []

    def predict(self, obs, hidden=None, deterministic=False):
        self.predict_calls.append({"obs": obs, "deterministic": deterministic})
        batch = next(iter(obs.values())).shape[0]
        return torch.zeros(batch, 2), None, None, None


def test_mjlab_policy_adapter_remaps_obs_and_calls_agent_predict_deterministically() -> None:
    agent = _FakeAgent()
    remap_calls = []

    def remap_obs_fn(obs):
        remap_calls.append(obs)
        return {"actor_state_enc": obs["policy"]}

    adapter = _MjlabPolicyAdapter(agent, remap_obs_fn)
    raw_obs = {"policy": torch.randn(1, 4)}

    action = adapter(raw_obs)

    assert remap_calls == [raw_obs]
    assert agent.predict_calls[0]["deterministic"] is True
    assert set(agent.predict_calls[0]["obs"].keys()) == {"actor_state_enc"}
    assert action.shape == (1, 2)


def test_start_mjlab_visualizer_unknown_task_fails_gracefully_and_returns_none() -> None:
    agent = _FakeAgent()
    handle = start_mjlab_visualizer(
        agent,
        "Definitely-Not-A-Real-Mjlab-Task-xyz",
        "cpu",
        remap_obs_fn=lambda obs: obs,
    )
    assert handle is None


def test_start_mjlab_visualizer_native_backend_unknown_task_fails_gracefully() -> None:
    """Same graceful-failure contract as the default viser backend --
    backend="native" must not change how construction failures are handled,
    only which viewer class gets constructed once the env exists."""
    agent = _FakeAgent()
    handle = start_mjlab_visualizer(
        agent,
        "Definitely-Not-A-Real-Mjlab-Task-xyz",
        "cpu",
        remap_obs_fn=lambda obs: obs,
        backend="native",
    )
    assert handle is None


def test_maybe_start_visualizer_passes_visualize_backend_through(monkeypatch) -> None:
    captured = {}

    def _fake_start_mjlab_visualizer(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(live_viewer_module, "start_mjlab_visualizer", _fake_start_mjlab_visualizer)

    args = _FakeArgs(visualize_backend="native")
    train_module._maybe_start_visualizer(_FakeAgentWithModel(), args, "cuda")

    assert captured["backend"] == "native"


def test_start_gym_visualizer_env_construction_failure_returns_none() -> None:
    def _broken_env_fn():
        raise RuntimeError("boom")

    agent = _FakeAgent()
    handle = start_gym_visualizer(
        agent,
        _broken_env_fn,
        remap_obs_fn=lambda obs: obs,
        obs_to_tensor_fn=lambda obs, device: obs,
    )
    assert handle is None


class _FakeRenderEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.render_calls = 0
        self.closed = False

    def reset(self):
        self.reset_calls += 1
        return {"state": torch.zeros(4)}, {}

    def step(self, action):
        self.step_calls += 1
        return {"state": torch.zeros(4)}, 0.0, False, False, {}

    def render(self):
        self.render_calls += 1

    def close(self):
        self.closed = True


def test_start_gym_visualizer_steps_predicts_and_renders_then_stop_ends_the_loop() -> None:
    agent = _FakeAgent()
    env = _FakeRenderEnv()

    handle = start_gym_visualizer(
        agent,
        lambda: env,
        remap_obs_fn=lambda obs: obs,
        obs_to_tensor_fn=lambda obs, device: obs,
    )
    assert isinstance(handle, VisualizerHandle)
    assert isinstance(handle.thread, threading.Thread)

    # Let a handful of iterations run, then ask it to stop -- this is the
    # graceful-shutdown path `srl-train`'s main() takes before returning,
    # not "let the daemon thread get killed at process exit" (which can
    # segfault mid-GPU-call for the mjlab backend).
    deadline = time.monotonic() + 5.0
    while env.step_calls < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert env.step_calls >= 3

    handle.stop()
    handle.thread.join(timeout=5.0)

    assert not handle.thread.is_alive()
    assert env.reset_calls == 1
    assert env.render_calls == env.step_calls
    assert len(agent.predict_calls) == env.step_calls
    assert all(call["deterministic"] is True for call in agent.predict_calls)
    assert env.closed is True


class _FailingRenderEnv(_FakeRenderEnv):
    """A render env whose .render() breaks after the first call -- exercises
    the try/except/finally path (graceful error message + env.close()) that
    covers a real dependency-missing failure (e.g. pygame not installed for
    render_mode="human" on classic_control envs, observed when smoke-testing
    this against Pendulum-v1 on a headless box)."""

    def render(self):
        super().render()
        raise RuntimeError("no display available")


def test_start_gym_visualizer_render_failure_stops_the_thread_and_closes_env() -> None:
    agent = _FakeAgent()
    env = _FailingRenderEnv()

    handle = start_gym_visualizer(
        agent,
        lambda: env,
        remap_obs_fn=lambda obs: obs,
        obs_to_tensor_fn=lambda obs, device: obs,
    )
    handle.thread.join(timeout=5.0)

    assert not handle.thread.is_alive()
    assert env.render_calls == 1
    assert env.closed is True


class _FakeModel:
    def __init__(self) -> None:
        self.encoders = {"actor_state_enc": None}
        self.encoder_input_names = {"actor_state_enc": "actor"}


class _FakeAgentWithModel:
    def __init__(self) -> None:
        self.model = _FakeModel()


class _FakeArgs:
    def __init__(self, **kwargs) -> None:
        self.visualize = True
        self.env_type = "mjlab"
        self.env = "mjlab:Some-Task"
        self.eval_freq = 50_000
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_mjlab_visualizer_forces_eval_freq_to_zero_to_avoid_cuda_graph_crash(
    monkeypatch,
) -> None:
    """Regression test: mjlab's env construction does a one-time CUDA graph
    capture (mujoco_warp's Simulation.create_graph()). --visualize's
    background thread steps its own env continuously on the GPU for the rest
    of the run; periodic in-training eval builds a brand-new mjlab env every
    `--eval-freq` steps, and if that overlaps the visualizer's ongoing
    stepping the capture gets corrupted -- confirmed via a real repro
    (training survived 9+ eval-triggered env rebuilds with --visualize off,
    crashed with a hard SIGABRT at the second eval cycle with it on).
    `_maybe_start_visualizer` must force eval off for a successfully-started
    mjlab visualizer to remove that race."""
    fake_handle = VisualizerHandle(thread=threading.Thread(target=lambda: None), stop=lambda: None)
    monkeypatch.setattr(live_viewer_module, "start_mjlab_visualizer", lambda *a, **k: fake_handle)

    args = _FakeArgs(eval_freq=50_000)
    handle = train_module._maybe_start_visualizer(_FakeAgentWithModel(), args, "cuda")

    assert handle is fake_handle
    assert args.eval_freq == 0


def test_mjlab_visualizer_leaves_eval_freq_alone_when_it_fails_to_start(monkeypatch) -> None:
    """If the visualizer never actually started (e.g. mjlab/viser missing),
    there's no ongoing background GPU activity to race against -- eval must
    stay untouched."""
    monkeypatch.setattr(live_viewer_module, "start_mjlab_visualizer", lambda *a, **k: None)

    args = _FakeArgs(eval_freq=50_000)
    handle = train_module._maybe_start_visualizer(_FakeAgentWithModel(), args, "cuda")

    assert handle is None
    assert args.eval_freq == 50_000


def test_gym_visualizer_does_not_touch_eval_freq(monkeypatch) -> None:
    """The CUDA-graph-capture race is mjlab-specific (plain Gymnasium envs
    don't do graph capture at all) -- --visualize on a flat env must not
    disable eval."""
    fake_handle = VisualizerHandle(thread=threading.Thread(target=lambda: None), stop=lambda: None)
    monkeypatch.setattr(live_viewer_module, "start_gym_visualizer", lambda *a, **k: fake_handle)

    args = _FakeArgs(env_type="flat", env="Pendulum-v1", eval_freq=50_000)
    handle = train_module._maybe_start_visualizer(_FakeAgentWithModel(), args, "cpu")

    assert handle is fake_handle
    assert args.eval_freq == 50_000
