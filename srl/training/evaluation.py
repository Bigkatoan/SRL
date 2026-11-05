"""Evaluation: deterministic eval episodes, eval scheduling, best checkpoints."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from srl.training._obs import agent_act, obs_to_tensors, remap_obs_to_encoders

__all__ = ["Evaluator", "evaluate_agent", "next_eval_step", "seed_best_tracker"]


def next_eval_step(start_step: int, eval_freq: int) -> int | None:
    if eval_freq <= 0:
        return None
    if start_step <= 0:
        return eval_freq
    return int(math.floor(start_step / eval_freq) + 1) * eval_freq


def seed_best_tracker(best_cm, best_tracker, device: str) -> None:
    """On `--resume`, seed `best_tracker.best_score` from a `best_*`
    checkpoint this run already produced before the restart, if any.

    Without this, a resumed run's very first eval would always look like
    "the best so far" to a freshly-constructed (`best_score=None`) tracker
    and immediately overwrite an existing `best_*` checkpoint even when the
    new score is worse than what's already saved -- silently discarding a
    genuinely-better pre-resume peak the whole feature exists to protect.
    """
    import torch

    candidates = sorted(best_cm.save_dir.glob("best_*.pt")) + sorted(
        best_cm.save_dir.glob("best_*.safetensors")
    )
    if not candidates:
        return
    latest = candidates[-1]
    if latest.suffix == ".safetensors":
        meta_path = latest.with_suffix(".meta.pt")
        payload = (
            torch.load(meta_path, map_location=device, weights_only=False)
            if meta_path.exists()
            else {}
        )
    else:
        payload = torch.load(latest, map_location=device, weights_only=False)
    best_tracker.seed_from_checkpoint(payload, monitor="eval/score_mean")
    if best_tracker.best_score is not None:
        best_tracker.best_path = latest


def evaluate_agent(
    agent, eval_env, *, env_type: str, seed: int, episodes: int, render: bool
) -> dict[str, float]:
    """Run `episodes` deterministic eval episodes against `eval_env`.

    `eval_env` is owned by the caller (constructed once, reused across every
    eval cycle in a run, closed once at the end) -- see :class:`Evaluator`.
    This function never constructs or closes it: rebuilding an
    isaaclab/mjlab env every eval cycle is both wasteful (a full GPU-batched
    simulator instance) and produces a wall of
    repeated construction-banner logging (Warp module loads, `Setting seed`,
    the ObservationManager/TerminationManager/... tables) on every single
    eval, which reads exactly like the environment is "repeatedly
    reinitializing" -- confirmed by a real test pass on JAVIS's mjlab task.
    """
    import numpy as np

    episode_scores: list[float] = []
    episode_lengths: list[int] = []
    success_values: list[float] = []
    encoder_names = list(agent.model.encoders.keys())

    for episode_index in range(max(int(episodes), 1)):
        obs, _ = eval_env.reset(seed=seed + episode_index)
        done = False
        truncated = False
        score = 0.0
        length = 0
        episode_success: float | None = None

        while not (done or truncated):
            obs_remapped = remap_obs_to_encoders(
                obs,
                encoder_names,
                encoder_input_names=getattr(agent.model, "encoder_input_names", None),
            )
            obs_t = obs_to_tensors(obs_remapped, agent.device, force_batch=True)
            action, _, _, _ = agent_act(agent, obs_t, deterministic=True)
            action_np = action.detach().cpu().numpy()
            # isaaclab/mjlab envs always expect a batched (num_envs,
            # action_dim) action, even at num_envs=1 -- IsaacLabWrapper
            # never unbatches, unlike a plain gymnasium.Env. Squeezing
            # here for those env types produces a 1-D action and
            # `action.shape[1]` in the env's action manager raises
            # IndexError.
            if (
                env_type not in ("isaaclab", "mjlab")
                and action_np.ndim > 1
                and action_np.shape[0] == 1
            ):
                action_np = action_np.squeeze(0)
            next_obs, reward, done, truncated, info = eval_env.step(action_np)
            score += float(np.asarray(reward).reshape(-1)[0])
            length += 1
            obs = next_obs
            if render and hasattr(eval_env, "render"):
                try:
                    eval_env.render()
                except Exception:
                    pass
            for key in ("is_success", "success"):
                if isinstance(info, dict) and key in info:
                    try:
                        episode_success = float(np.asarray(info[key]).reshape(-1)[0])
                    except Exception:
                        pass
                    break
        episode_scores.append(score)
        episode_lengths.append(length)
        # SB3/baselines convention: an episode's success is `is_success` at
        # its FINAL step -- not the fraction of steps that happened to report
        # success (which under-reports e.g. FetchReach, where the goal is only
        # reached part-way through the episode).
        if episode_success is not None:
            success_values.append(episode_success)

    metrics = {
        "eval/score_mean": float(sum(episode_scores) / len(episode_scores)),
        "eval/score_max": float(max(episode_scores)),
        "eval/episode_length_mean": float(sum(episode_lengths) / len(episode_lengths)),
        "eval/episodes": float(len(episode_scores)),
    }
    if success_values:
        metrics["eval/success_mean"] = float(sum(success_values) / len(success_values))
    return metrics


class Evaluator:
    """Owns the eval env, the eval schedule and best-checkpoint tracking.

    Parameters
    ----------
    env:
        An SRL-interface eval env (owned by the caller, closed only if this
        object built it), or ``None`` to build one lazily with *env_fn* on the
        first evaluation that actually fires -- rebuilding an isaaclab/mjlab
        env every eval cycle is both wasteful and produces a wall of repeated
        construction-banner logging, so one env is reused for the whole run.
    env_fn:
        Zero-arg factory for the eval env (used when *env* is ``None``).
    eval_freq:
        Env steps between periodic evaluations; ``<= 0`` disables them.
    n_episodes:
        Deterministic episodes per evaluation.
    seed:
        Episode ``i`` resets with ``seed + i``.
    env_type:
        ``"isaaclab"``/``"mjlab"`` keep the action batched at ``num_envs=1``.
    best_cm / best_tracker:
        When both are given, a ``best_*`` checkpoint is saved via *best_cm*
        whenever ``eval/score_mean`` improves on everything *best_tracker*
        has seen this run (``--save-best``).
    """

    def __init__(
        self,
        env: Any = None,
        *,
        env_fn: Callable[[], Any] | None = None,
        eval_freq: int = 0,
        n_episodes: int = 10,
        seed: int = 10_000,
        env_type: str | None = None,
        render: bool = False,
        best_cm: Any = None,
        best_tracker: Any = None,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.env_fn = env_fn
        self._owns_env = env is None
        self.eval_freq = int(eval_freq or 0)
        self.n_episodes = int(n_episodes)
        self.seed = int(seed)
        self.env_type = str(env_type or "flat")
        self.render = bool(render)
        self.best_cm = best_cm
        self.best_tracker = best_tracker
        self.verbose = verbose
        self.next_eval_step: int | None = next_eval_step(0, self.eval_freq)
        #: The real step eval last fired at (``None`` if never). The "should
        #: the final eval run" decision compares against this directly:
        #: on-policy rollout stepping can overshoot past the nominal
        #: eval_freq-aligned threshold (rollout_steps is a ceil-division), so
        #: the step periodic eval actually fires at is not always an exact
        #: multiple of eval_freq, even though next_eval_step's schedule is.
        self.last_eval_step: int | None = None

    @property
    def enabled(self) -> bool:
        return self.eval_freq > 0

    def reset_schedule(self, start_step: int) -> None:
        self.next_eval_step = next_eval_step(start_step, self.eval_freq)

    def _get_env(self):
        if self.env is None:
            if self.env_fn is None:
                raise RuntimeError("Evaluator has neither an eval env nor an env_fn to build one.")
            self.env = self.env_fn()
        return self.env

    def evaluate(
        self, agent, step: int, *, logger=None, total_steps: int | None = None
    ) -> dict[str, float]:
        """Run one evaluation now, log it and update the best checkpoint."""
        eval_metrics = evaluate_agent(
            agent,
            self._get_env(),
            env_type=self.env_type,
            seed=self.seed,
            episodes=self.n_episodes,
            render=self.render,
        )
        if logger is not None:
            logger.record_metrics(
                eval_metrics, step=step, total_steps=total_steps, prefix=None, console=False
            )
        success_part = ""
        if "eval/success_mean" in eval_metrics:
            # Only present for goal-conditioned envs (env_type: goal) -- the
            # primary metric for exactly that env type, so it belongs in the
            # live console line, not just in metrics.jsonl/TensorBoard.
            success_part = f" | success_mean={eval_metrics['eval/success_mean']:.4f}"
        if self.verbose:
            print(
                f"[eval] step {step} | score_mean={eval_metrics['eval/score_mean']:.4f}"
                f"{success_part} | episodes={int(eval_metrics['eval/episodes'])}",
                flush=True,
            )
        if self.best_tracker is not None and self.best_cm is not None:
            saved_path = self.best_tracker.update(
                eval_metrics["eval/score_mean"],
                agent,
                cm=self.best_cm,
                step=step,
                metrics=eval_metrics,
            )
            if saved_path is not None and self.verbose:
                print(
                    f"[eval] step {step} | new best score_mean="
                    f"{eval_metrics['eval/score_mean']:.4f} -> saved {saved_path}",
                    flush=True,
                )
        self.last_eval_step = step
        return eval_metrics

    def maybe_evaluate(
        self, agent, step: int, *, logger=None, total_steps: int | None = None
    ) -> dict[str, float] | None:
        """Evaluate iff *step* reached the next scheduled eval step."""
        if self.next_eval_step is None or step < self.next_eval_step:
            return None
        metrics = self.evaluate(agent, step, logger=logger, total_steps=total_steps)
        self.next_eval_step = self.next_eval_step + self.eval_freq
        return metrics

    def finalize(
        self, agent, step: int, *, logger=None, total_steps: int | None = None
    ) -> dict[str, float] | None:
        """Final eval at the end of training, unless one already ran at *step*."""
        if self.next_eval_step is None or self.last_eval_step == step:
            return None
        return self.evaluate(agent, step, logger=logger, total_steps=total_steps)

    def close(self) -> None:
        if self.env is not None and self._owns_env:
            try:
                self.env.close()
            finally:
                self.env = None
