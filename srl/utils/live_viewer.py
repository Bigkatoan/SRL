"""Live training-time visualizer (``srl-train --visualize``).

Runs one extra single env, doing deterministic inference with the agent's
LIVE (currently-training) model, rendered in a background daemon thread
while the main training loop keeps running unaffected.

No snapshotting or periodic reload is needed: the policy callables below
read ``agent.model``/``agent.predict`` directly, and the main thread updates
that same object in place, so the viewer is always showing the current
weights by construction. A possibly-torn read mid-``optimizer.step()`` (or
``agent.predict()``'s own ``model.eval()`` racing the training loop's
``model.train()``) is a non-issue for a visual monitor -- this mirrors the
relaxed-consistency tradeoff ``AsyncOffPolicyRunner`` already accepts for
its background trainer thread ("a maximum staleness of one gradient step",
per its own docstring). No lock is taken for the same reason: the cost of
occasionally rendering a half-updated frame is nothing, and correctness of
training itself never depends on anything this module does.

Two backends, matching the two ways SRL builds envs:

* mjlab: reuses mjlab's own live viewer, pointed at a fresh single-env
  instance of the same task -- either ``ViserPlayViewer`` (browser-based,
  the default) or ``NativeMujocoViewer`` (a real desktop GLFW window via
  ``mujoco.viewer.launch_passive``, selected with ``--visualize-backend
  native``). Both inherit mjlab's ``BaseViewer`` and share the same
  ``run()``/``_interrupted`` stop protocol, so `start_mjlab_visualizer`
  only needs to pick which class to construct -- everything else below is
  backend-agnostic. `NativeMujocoViewer` needs an actual display
  (``DISPLAY``/GPU-attached desktop or X11 forwarding) reachable from the
  training process; `ViserPlayViewer` only needs a browser that can reach
  the printed URL, which is why it's the default for headless/remote
  training boxes.
* Plain Gymnasium-style envs (``flat``/``goal``/``racecar`` env types): a
  simple step-and-render loop against an env constructed with a renderable
  ``render_mode``.

isaaclab is deliberately not supported here: a process hosts exactly one
Isaac Sim render context, shared by every isaaclab env in it, so a second
"just for viewing" env can't be added alongside headless training envs the
way it can for mjlab (independent per-env MuJoCo state) or Gymnasium
(independent env instances, no shared global renderer).

Both backends return a `VisualizerHandle` (or None if they failed to start)
rather than a bare `threading.Thread`. Call `.stop()` before the process
exits (`srl-train`'s `main()` does, right before it returns) -- a `daemon=True`
thread still doing GPU work (mjlab's physics step, warp kernels) gets killed
abruptly at interpreter shutdown if it's mid-call when the process exits,
which can segfault/core-dump on the way out. `.stop()` asks the loop to wind
down on its own terms first; the caller should still `thread.join(timeout=...)`
afterwards since `.stop()` itself doesn't block.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class VisualizerHandle:
    thread: threading.Thread
    stop: Callable[[], None]


class _MjlabPolicyAdapter:
    """Adapts an SRL agent to mjlab's ``PolicyProtocol``: obs (the raw dict
    of per-group torch tensors ``env.get_observations()`` returns) -> action
    tensor, routed through the same obs-remap rules `srl-train` uses
    everywhere else.
    """

    def __init__(self, agent: Any, remap_obs_fn: Callable[[dict], dict]) -> None:
        self._agent = agent
        self._remap_obs_fn = remap_obs_fn

    def __call__(self, obs: dict):
        import torch

        remapped = self._remap_obs_fn(obs)
        obs_t = {k: v.to(self._agent.device) for k, v in remapped.items()}
        with torch.no_grad():
            action, _, _, _ = self._agent.predict(obs_t, deterministic=True)
        return action

    def reset(self) -> None:
        pass


def start_mjlab_visualizer(
    agent: Any,
    task_name: str,
    device: str,
    *,
    remap_obs_fn: Callable[[dict], dict],
    backend: str = "viser",
) -> VisualizerHandle | None:
    """Launch mjlab's live viewer against a fresh single-env instance of
    *task_name*.

    *backend* selects which of mjlab's own viewer classes to construct:
    ``"viser"`` (default, browser-based, works headless) or ``"native"``
    (a real desktop GLFW window via ``mujoco.viewer.launch_passive`` --
    needs a display reachable from the training process).

    Returns a `VisualizerHandle`, or None if mjlab/the requested backend
    aren't importable or the viewer failed to start -- visualization is a
    nice-to-have and must never take training down with it.
    """
    if backend == "native" and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        # GLFW's failure to init here is NOT a catchable Python exception --
        # confirmed via a real repro: mujoco.viewer.launch_passive() on a
        # headless box (no $DISPLAY) prints "ERROR: could not initialize
        # GLFW" and hard-exits the whole process (not just this thread),
        # bypassing every try/except below. Check for a display up front and
        # skip entirely rather than risk that crash -- this is the one
        # failure mode this function cannot degrade gracefully from once
        # `launch_passive()` is actually called.
        print(
            "[srl-train] --visualize: --visualize-backend native needs a display "
            "($DISPLAY is unset) -- skipping rather than risk a hard crash "
            "(mujoco.viewer.launch_passive's GLFW failure isn't catchable here). "
            "Use --visualize-backend viser for headless/remote training, or run "
            "with a real display / X11 forwarding attached.",
            file=sys.stderr,
        )
        return None

    try:
        import mjlab  # noqa: F401  (side effect: discovers mjlab.tasks packages)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        if backend == "native":
            from mjlab.viewer import NativeMujocoViewer as ViewerCls
        else:
            from mjlab.viewer import ViserPlayViewer as ViewerCls
    except ImportError as exc:
        print(
            f"[srl-train] --visualize: mjlab/{backend} not importable ({exc}); skipping.",
            file=sys.stderr,
        )
        return None

    try:
        env_cfg = load_env_cfg(task_name)
        env_cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(env_cfg, device=device)
        policy = _MjlabPolicyAdapter(agent, remap_obs_fn)
        viewer = ViewerCls(env, policy)
    except Exception as exc:
        print(
            f"[srl-train] --visualize: failed to start the mjlab {backend} viewer "
            f"({exc!r}); skipping.",
            file=sys.stderr,
        )
        return None

    def _run() -> None:
        try:
            # catch_sigint=False: signal.signal() only works on the main
            # thread; BaseViewer.run() already degrades gracefully if it
            # can't install a handler, but there's no reason to try here.
            viewer.run(catch_sigint=False)
        except Exception as exc:  # pragma: no cover -- background thread, best effort
            print(
                f"[srl-train] --visualize: viewer stopped with an error: {exc!r}", file=sys.stderr
            )

    def _stop() -> None:
        # Neither viewer class has a public stop() -- ViserPlayViewer's
        # is_running() always returns True by design ("Viser runs until
        # process is killed"), and NativeMujocoViewer's tracks its own
        # window instead (closes when the user closes it, not on demand).
        # Both inherit the same BaseViewer._interrupted flag from their
        # own SIGINT handler, so setting it directly is the one stop
        # mechanism that works for both.
        try:
            viewer._interrupted = True
        except Exception:
            pass

    thread = threading.Thread(target=_run, name="srl-visualizer", daemon=True)
    thread.start()
    if backend == "native":
        print(
            "[srl-train] --visualize: mjlab native viewer starting in the "
            "background (a desktop window should appear shortly -- needs a "
            "display reachable from this process)."
        )
    else:
        print(
            "[srl-train] --visualize: mjlab viewer starting in the background "
            "(see the console output above for the URL to open in a browser)."
        )
    return VisualizerHandle(thread=thread, stop=_stop)


def start_gym_visualizer(
    agent: Any,
    make_render_env_fn: Callable[[], Any],
    *,
    remap_obs_fn: Callable[[dict], dict],
    obs_to_tensor_fn: Callable[[dict, Any], dict],
) -> VisualizerHandle | None:
    """Launch a simple render loop for a plain Gymnasium-style env (``flat``/
    ``goal``/``racecar`` env types): step *make_render_env_fn()*'s env with
    the live agent's deterministic action every step and call ``.render()``.

    The env must already be constructed with a renderable ``render_mode``
    (e.g. ``"human"``) -- this function does not set one itself.
    """
    try:
        env = make_render_env_fn()
    except Exception as exc:
        print(
            f"[srl-train] --visualize: failed to create the render env ({exc!r}); skipping.",
            file=sys.stderr,
        )
        return None

    stop_event = threading.Event()

    def _run() -> None:
        import torch

        try:
            obs, _ = env.reset()
            while not stop_event.is_set():
                obs_remapped = remap_obs_fn(obs)
                obs_t = obs_to_tensor_fn(obs_remapped, agent.device)
                with torch.no_grad():
                    action, _, _, _ = agent.predict(obs_t, deterministic=True)
                action_np = action.detach().cpu().numpy()
                if action_np.ndim > 1 and action_np.shape[0] == 1:
                    action_np = action_np.squeeze(0)
                obs, _, terminated, truncated, _ = env.step(action_np)
                env.render()
                if terminated or truncated:
                    obs, _ = env.reset()
        except Exception as exc:  # pragma: no cover -- background thread, best effort
            print(
                f"[srl-train] --visualize: viewer stopped with an error: {exc!r}", file=sys.stderr
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

    thread = threading.Thread(target=_run, name="srl-visualizer", daemon=True)
    thread.start()
    print("[srl-train] --visualize: rendering one live env in the background.")
    return VisualizerHandle(thread=thread, stop=stop_event.set)
