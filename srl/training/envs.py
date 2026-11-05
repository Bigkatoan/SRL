"""Environment construction for training: env-id -> SRL-interface env.

Everything that turns "an env id from a YAML/CLI flag" or "a raw Gymnasium
env handed to ``agent.learn()``" into the SRL wrapper interface lives here
(moved out of ``srl/cli/train.py`` so the Python API and the CLI build envs
the exact same way).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "bootstrap_isaac_sim",
    "build_training_env",
    "env_action_dim",
    "is_batched_env",
    "make_env",
    "normalize_env_name",
    "resolve_env_spec",
    "wrap_env",
]


# `gymnasium_robotics.register_robotics_envs()` unconditionally re-registers
# every Fetch/AntMaze/... variant on each call, and gymnasium.register() warns
# loudly ("Overriding environment X already in registry") on every id that's
# already there. `make_env` runs once for the training env and again for
# every eval cycle (the evaluator builds its own env the same way) --
# without the once-per-process guard below, a single run re-registers
# everything, and re-prints every one of gymnasium_robotics's ~400 warnings,
# on every eval.
#
# On the currently pinned version, importing gymnasium_robotics already
# registers everything as an import-time side effect, making the explicit
# call redundant even the FIRST time -- but this is calling a private-ish
# behavioral detail, not a documented contract, and the >=1.2.0 range this
# project supports may not all auto-register on import. So: keep calling it
# (for older-version compatibility), but only once per process, and swallow
# the specific "already in registry" noise it's expected to produce -- this
# is not a real problem to surface to a user, on the first call or any other.
_robotics_envs_registered = False


def _register_robotics_envs_once() -> None:
    global _robotics_envs_registered
    if _robotics_envs_registered:
        return
    import warnings

    import gymnasium_robotics

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*[Oo]verriding environment .* already in registry.*"
        )
        gymnasium_robotics.register_robotics_envs()
    _robotics_envs_registered = True


def _env_class_supports_partial_reset(env_class) -> bool:
    """Whether ``env_class.reset`` accepts an ``env_ids`` kwarg.

    Checked against the CLASS (before construction), not an instance --
    ``env_cfg.auto_reset`` has to be decided before the env object exists,
    so this has to be knowable up front. Used to gate two things together:
    whether it's safe to disable the env's own auto-reset (only safe if the
    caller can do the equivalent partial reset itself afterward) and
    whether ``IsaacLabWrapper`` should actually attempt to do so.

    mjlab's ``ManagerBasedRlEnv.reset`` takes ``env_ids`` (verified against
    its real source: ``def reset(self, *, seed=None, env_ids=None,
    options=None)``). Real Isaac Lab is not installed anywhere this was
    tested, so this checks it the same generic way rather than assuming --
    if a future Isaac Lab version's signature doesn't match, this correctly
    falls back to the old, unfixed-but-not-broken behavior instead of
    guessing wrong and crashing on the first partial reset.
    """
    import inspect

    try:
        params = inspect.signature(env_class.reset).parameters
    except (TypeError, ValueError):
        return False
    return "env_ids" in params


def make_env(env_name: str, device: str, n_envs: int, env_type: str = "flat", render: bool = False):
    """Build an env for training/eval, or (``render=True``) for the
    ``--visualize`` live viewer's plain-Gymnasium path (flat/goal/racecar
    only -- isaaclab/mjlab visualization goes through
    ``srl.utils.live_viewer`` instead, since they need a differently-built
    raw env; ``render`` is a no-op for those two branches here).
    """
    import gymnasium as gym

    from srl.envs.action_rescale import maybe_rescale_action
    from srl.envs.goal_env_wrapper import GoalEnvWrapper
    from srl.envs.gymnasium_wrapper import GymnasiumWrapper
    from srl.envs.racecar_wrapper import RacecarWrapper

    normalized_env_type = (env_type or "flat").strip().lower()
    normalized_env_name = normalize_env_name(env_name, normalized_env_type)
    make_kwargs = {"render_mode": "human"} if render else {}

    if normalized_env_type == "isaaclab" or normalized_env_name.startswith("isaaclab:"):
        from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

        task_name = normalized_env_name.split(":", 1)[1]
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv

        env_cfg = isaaclab_tasks.utils.parse_env_cfg(task_name, device=device, num_envs=n_envs)
        # See `_env_class_supports_partial_reset` / `IsaacLabWrapper`'s own
        # docstring: only disable auto-reset when we've confirmed this env
        # class can do the equivalent partial reset itself, so a real Isaac
        # Lab env whose `reset()` signature doesn't happen to match mjlab's
        # degrades to the old (unfixed, but not broken) behavior instead of
        # crashing on the first `env.reset(env_ids=...)` call.
        supports_partial_reset = _env_class_supports_partial_reset(ManagerBasedRLEnv) and hasattr(
            env_cfg, "auto_reset"
        )
        if supports_partial_reset:
            env_cfg.auto_reset = False
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        return IsaacLabWrapper(base_env, supports_partial_reset=supports_partial_reset)

    if normalized_env_type == "mjlab" or normalized_env_name.startswith("mjlab:"):
        # mjlab (github.com/mujocolab/mjlab) is a MuJoCo-Warp GPU-batched
        # env stack that mirrors Isaac Lab's ManagerBasedRLEnv API closely
        # enough that IsaacLabWrapper works unchanged -- no real Isaac Sim/
        # isaaclab install needed, which is the whole point of this branch:
        # same "one process, num_envs already batched on GPU" shape, without
        # the multi-GB Omniverse runtime. Task lookup goes through mjlab's
        # own registry (mjlab.tasks.registry), populated by auto-importing
        # every package registered under the "mjlab.tasks" entry-point group
        # -- the same mechanism `import mjlab` itself runs at import time, so
        # any task package installed alongside mjlab (e.g. a project's own
        # `your_pkg.tasks` registered via
        # `[project.entry-points."mjlab.tasks"]` in its pyproject.toml) is
        # already discoverable here with no extra wiring.
        from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

        task_name = normalized_env_name.split(":", 1)[1]
        import mjlab  # noqa: F401  (side effect: discovers installed mjlab.tasks packages)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(task_name)
        env_cfg.scene.num_envs = n_envs
        # `auto_reset=False` + `IsaacLabWrapper`'s own manual per-env reset
        # is what makes `info["true_final_obs"]` (the TRUE terminal
        # observation, not a freshly-reset unrelated episode's first frame)
        # available to the on/off-policy trainers /
        # `AsyncOffPolicyRunner` for correct time-limit bootstrapping. See
        # `IsaacLabWrapper`'s docstring for the full mechanism and why this
        # was wrong before (silently corrupting exactly the transitions a
        # policy good enough to survive to the episode time limit produces
        # -- i.e. getting WORSE the better training goes).
        supports_partial_reset = _env_class_supports_partial_reset(ManagerBasedRlEnv) and hasattr(
            env_cfg, "auto_reset"
        )
        if supports_partial_reset:
            env_cfg.auto_reset = False
        base_env = ManagerBasedRlEnv(env_cfg, device=device)
        return IsaacLabWrapper(base_env, supports_partial_reset=supports_partial_reset)

    # Plain-Gymnasium env types: the policy (tanh/clipped actor heads),
    # replay buffers and warmup all work in a normalised [-1, 1] action
    # domain; `maybe_rescale_action` maps it onto the env's real finite
    # [low, high] bounds right at `env.step()` (no-op for spaces that are
    # already [-1, 1], unbounded, or not continuous). isaaclab/mjlab above
    # are left untouched: their action spaces are unbounded and their
    # action terms do their own scaling.
    if normalized_env_type == "goal":
        _register_robotics_envs_once()
        return maybe_rescale_action(GoalEnvWrapper(gym.make(normalized_env_name, **make_kwargs)))

    if normalized_env_type == "racecar":
        import racecar_gym  # noqa: F401

        return maybe_rescale_action(RacecarWrapper(gym.make(normalized_env_name, **make_kwargs)))

    base = gym.make(normalized_env_name, **make_kwargs)
    return maybe_rescale_action(GymnasiumWrapper(base))


# ──────────────────────────────────────────────────────────────────────────────
# Env-id / env-type resolution
# ──────────────────────────────────────────────────────────────────────────────


def resolve_env_name(cli_env: str | None, raw_cfg: dict) -> str:
    env_name = cli_env or raw_cfg.get("env_id") or (raw_cfg.get("train") or {}).get("env_id")
    if not env_name:
        raise ValueError("Environment id is required. Pass --env or set env_id in the YAML config.")
    return env_name


def resolve_env_type(raw_cfg: dict) -> str:
    env_type = raw_cfg.get("env_type") or (raw_cfg.get("train") or {}).get("env_type") or "flat"
    return str(env_type).strip().lower()


def normalize_env_name(env_name: str, env_type: str) -> str:
    if env_type not in ("isaaclab", "mjlab"):
        return env_name
    prefix = f"{env_type}:"
    if env_name.startswith(prefix):
        return env_name
    return f"{prefix}{env_name}"


def resolve_env_spec(cli_env: str | None, raw_cfg: dict) -> tuple[str, str]:
    """``(normalized env id, env_type)`` from a CLI ``--env`` and a YAML dict."""
    env_type = resolve_env_type(raw_cfg)
    env_name = resolve_env_name(cli_env, raw_cfg)
    return normalize_env_name(env_name, env_type), env_type


def is_isaac_like(env_name: str | None, env_type: str | None) -> bool:
    """isaaclab/mjlab: one process, ``num_envs`` already batched on device."""
    env_type = str(env_type or "")
    env_name = str(env_name or "")
    return (
        env_type in ("isaaclab", "mjlab")
        or env_name.startswith("isaaclab:")
        or env_name.startswith("mjlab:")
    )


def is_batched_env(env: Any, env_type: str | None = None, env_name: str | None = None) -> bool:
    """Whether *env* returns batched observations and auto-resets sub-envs.

    True for isaaclab/mjlab (always batched, even at ``num_envs=1``) and for
    any SRL vector env with more than one sub-env. A single plain
    ``GymnasiumWrapper`` env is unbatched and must be reset by the caller.
    """
    if is_isaac_like(env_name, env_type):
        return True
    try:
        from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

        if isinstance(env, IsaacLabWrapper):
            return True
    except Exception:  # pragma: no cover - optional deps
        pass
    return getattr(env, "num_envs", 1) > 1


def env_action_dim(env) -> int:
    """The per-env action dimensionality, fed to SAC/DDPG/TD3's config.

    `env.act_space` is the BATCHED action space on isaaclab/mjlab envs
    (shape (num_envs, action_dim), matching gymnasium's VectorEnv
    convention) -- see IsaacLabWrapper's own docstring, which is exactly why
    it also exposes `single_act_space` (falling back to `act_space` itself
    for envs that don't distinguish the two, e.g. plain Gymnasium). Using
    `act_space` directly here computed a wildly wrong `action_dim` on every
    mjlab/isaaclab off-policy run (e.g. 128 instead of 2 at num_envs=64) --
    confirmed by a real run collapsing SAC's target_entropy to -128 and
    alpha to ~0 within a few thousand steps. DDPG/TD3 read the same
    `action_dim` to size their exploration-noise generators, so they were
    equally affected (wrong-shaped noise, not just SAC's entropy target).
    """
    single_act_space = getattr(env, "single_act_space", None) or env.act_space
    return int(np.prod(getattr(single_act_space, "shape", ()) or (1,)))


# ──────────────────────────────────────────────────────────────────────────────
# Raw env -> SRL interface
# ──────────────────────────────────────────────────────────────────────────────


def _is_srl_env(env: Any) -> bool:
    return hasattr(env, "obs_space") and hasattr(env, "act_space")


def _is_goal_space(space: Any) -> bool:
    keys = getattr(space, "spaces", None)
    return isinstance(keys, dict) and {"observation", "achieved_goal", "desired_goal"} <= set(keys)


def wrap_env(env: Any, *, device: str = "cpu", env_type: str | None = None) -> Any:
    """Return *env* in SRL's wrapper interface.

    * ``None`` -> ``None``.
    * ``str`` -> :func:`make_env` (a Gymnasium id, ``isaaclab:<task>`` or
      ``mjlab:<task>``; *env_type* selects e.g. the ``goal`` wrapper).
    * An env that already speaks the SRL interface (``obs_space``/
      ``act_space``: every ``srl.envs`` wrapper and vector env) -> unchanged.
    * A raw ``gymnasium.Env`` -> :class:`~srl.envs.GoalEnvWrapper` (GoalEnv
      Dict observation) or :class:`~srl.envs.GymnasiumWrapper`, then
      :class:`~srl.envs.RescaleActionWrapper` when its Box action bounds are
      not already ``[-1, 1]`` -- exactly what ``srl-train`` does.
    """
    if env is None:
        return None
    if isinstance(env, str):
        return make_env(env, device, 1, env_type or "flat")
    if _is_srl_env(env):
        return env
    import gymnasium as gym

    from srl.envs.action_rescale import maybe_rescale_action

    if isinstance(env, gym.vector.VectorEnv):
        raise TypeError(
            "gymnasium.vector.VectorEnv is not supported directly; pass a "
            "srl.envs.SyncVectorEnv / AsyncVectorEnv built from per-env factories "
            "(e.g. SyncVectorEnv([lambda: gym.make(id)] * n)) instead."
        )
    if not isinstance(env, gym.Env):
        raise TypeError(f"Expected a gymnasium.Env or an SRL env wrapper, got {type(env)!r}")
    if _is_goal_space(env.observation_space):
        from srl.envs.goal_env_wrapper import GoalEnvWrapper

        return maybe_rescale_action(GoalEnvWrapper(env))
    from srl.envs.gymnasium_wrapper import GymnasiumWrapper

    return maybe_rescale_action(GymnasiumWrapper(env))


def env_factory_for(env: Any, env_type: str = "flat", device: str = "cpu") -> Callable | None:
    """Best-effort picklable factory that builds a fresh copy of *env*.

    Used for evaluation envs and A3C workers when only an env *instance* was
    given. Returns ``None`` when the env has no registered Gymnasium id.
    """
    from functools import partial

    if isinstance(env, str):
        return partial(make_env, env, device, 1, env_type)
    probe = env
    while probe is not None:
        spec = getattr(probe, "spec", None)
        env_id = getattr(spec, "id", None)
        if env_id:
            return partial(make_env, env_id, device, 1, env_type)
        probe = getattr(probe, "env", None)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Training env (vectorization + seeding) and Isaac Sim bootstrap
# ──────────────────────────────────────────────────────────────────────────────


def build_training_env(
    env_name: str,
    *,
    device: str = "cpu",
    n_envs: int = 1,
    env_type: str = "flat",
    vec_mode: str = "auto",
    seed: int = 0,
    env_fn: Callable[..., Any] | None = None,
):
    """Build the (possibly vectorized) training env exactly like ``srl-train``.

    isaaclab/mjlab vectorize internally; otherwise ``n_envs > 1`` wraps
    ``n_envs`` copies in :class:`~srl.envs.SyncVectorEnv` /
    :class:`~srl.envs.AsyncVectorEnv` (``vec_mode``: ``auto`` -> async).
    """
    from srl.envs.async_vector_env import AsyncVectorEnv
    from srl.envs.sync_vector_env import SyncVectorEnv
    from srl.utils.seeding import seed_env_spaces

    factory = env_fn or make_env

    def _make(seed_offset: int = 0):
        built = factory(env_name, device, n_envs, env_type)
        # Per-env seed for the env's own spaces (`act_space.sample()` etc.);
        # each vector sub-env gets `seed + index`, matching how the vector
        # envs offset `reset(seed=...)`.
        seed_env_spaces(built, seed + seed_offset)
        return built

    if is_isaac_like(env_name, env_type) or n_envs == 1:
        return _make()
    env_fns = [lambda i=i: _make(i) for i in range(n_envs)]
    if vec_mode == "sync":
        env = SyncVectorEnv(env_fns)
    else:
        env = AsyncVectorEnv(env_fns)
    # The vector env's own (parent-side) action space is what off-policy
    # warmup samples from -- for AsyncVectorEnv it's a fresh unpickled
    # copy from worker 0, so seed it here (reproducible warmup).
    seed_env_spaces(env, seed)
    return env


def bootstrap_isaac_sim() -> None:
    """Start Isaac Sim's (singleton) app once, before any ``omni.*`` import.

    Uses IsaacLab's AppLauncher (not raw SimulationApp) so it loads the
    headless-rendering kit file, which sets ``/isaaclab/cameras_enabled``
    (required by TiledCamera) and activates ``omni.replicator.core``. Never
    raises: a failure is reported on stderr.
    """
    try:
        import atexit

        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=True).app
        atexit.register(app.close)
        # Set the asset-root BEFORE any isaaclab sub-module is imported; the
        # constant NUCLEUS_ASSET_ROOT_DIR is evaluated at **module load time**
        # in isaaclab/utils/assets.py.  If the carb setting is None at that
        # point the path becomes "None/..." and USD loading fails.
        import carb as _carb

        settings = _carb.settings.get_settings()
        if not settings.get("/persistent/isaac/asset_root/cloud"):
            settings.set(
                "/persistent/isaac/asset_root/cloud",
                "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
            )
        print("[srl-train] Isaac Sim initialized (headless)")
    except Exception as exc:
        print(f"[srl-train] WARNING: Isaac Sim could not be initialized: {exc}", file=sys.stderr)
