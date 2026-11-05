"""Off-policy collect/update loop (SAC, TD3, DDPG), sync and async.

Moved verbatim (modulo naming) from ``srl/cli/train.py``'s
``_run_off_policy`` so its Phase-0-verified behaviour is preserved: random
warmup in the normalised action domain, the TRUE terminal observation for
auto-reset vector envs, separate terminated/truncated flags, HER episode
collection, per-episode exploration-noise reset, ``gradient_steps``/UTD
bookkeeping, and the ``AsyncOffPolicyRunner`` fast path.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from srl.envs.vector_utils import get_final_obs
from srl.training._obs import (
    agent_act,
    encoder_layout,
    obs_to_tensors,
    remap_obs_to_encoders,
    split_vector_transition,
)
from srl.training.algo_config import her_goal_obs_parts
from srl.training.base import Trainer
from srl.utils.spaces import sample_action_space

__all__ = ["OffPolicyTrainer", "warn_if_no_updates_will_run"]


def warn_if_no_updates_will_run(agent, total_steps: int, start_step: int) -> None:
    """Warn when the step budget won't reach `learning_starts`/`update_after`.

    Off-policy algorithms only take their first gradient step once at least
    that many env steps have been collected. A smoke-test run with a small
    step budget below that threshold completes "successfully" with zero
    gradient updates and byte-identical eval scores across every checkpoint
    -- easy to misdiagnose as a training bug rather than what it actually is:
    no learning happened at all in this run.
    """
    update_after = getattr(agent.cfg, "update_after", None)
    source = "update_after"
    if update_after is None:
        update_after = getattr(agent.cfg, "learning_starts", None)
        source = "learning_starts"
    if update_after is None:
        return
    remaining_steps = int(total_steps) - int(start_step)
    if remaining_steps < int(update_after):
        print(
            f"[srl-train] Warning: --steps leaves only {remaining_steps} env steps to "
            f"collect, but this algorithm's {source} is {int(update_after)} -- no "
            "gradient update will run in this call. Increase --steps (or the "
            f"config's train.total_steps) past {source}, or lower it, to actually train.",
            file=sys.stderr,
        )


def _schedule(cfg) -> tuple[int, int, int, int]:
    """``(random_steps, update_after, update_every, gradient_steps)``.

    Single place that interprets the off-policy schedule fields (to be
    replaced by ``srl.config.normalize_offpolicy_schedule`` once merged).
    """
    random_steps = getattr(cfg, "start_steps", None)
    if random_steps is None:
        random_steps = getattr(cfg, "learning_starts", 10_000)
    update_after = getattr(cfg, "update_after", None)
    if update_after is None:
        update_after = getattr(cfg, "learning_starts", 10_000)
    update_every = getattr(cfg, "update_every", None)
    if update_every is None:
        update_every = getattr(cfg, "train_freq", 1)
    gradient_steps = max(int(getattr(cfg, "gradient_steps", 1)), 1)
    return (
        max(int(random_steps), 0),
        max(int(update_after), 0),
        max(int(update_every), 1),
        (gradient_steps),
    )


class OffPolicyTrainer(Trainer):
    """Step the env, store transitions, ``agent.update()`` every ``train_freq``.

    Extra parameter (on top of :class:`~srl.training.base.Trainer`'s):

    runner_cfg:
        An :class:`~srl.core.config.AsyncRunnerConfig` with ``use_async`` or
        ``use_gpu_buffer`` set switches to
        :class:`~srl.runners.AsyncOffPolicyRunner` (``on_step`` is not called
        on that path).
    """

    def __init__(self, agent, env, *, runner_cfg: Any = None, **kwargs) -> None:
        super().__init__(agent, env, **kwargs)
        self.runner_cfg = runner_cfg

    # ------------------------------------------------------------------
    def _train(self) -> None:
        warn_if_no_updates_will_run(self.agent, self.total_steps, self.start_step)
        runner_cfg = self.runner_cfg
        if runner_cfg is not None and (runner_cfg.use_async or runner_cfg.use_gpu_buffer):
            self._train_async(runner_cfg)
        else:
            self._train_sync()

    # ------------------------------------------------------------------
    def _train_async(self, runner_cfg) -> None:
        from srl import runners

        agent = self.agent
        gradient_steps = _schedule(agent.cfg)[3]
        # The async runner has always treated an explicit 0 for these as
        # "unset" (falls back to learning_starts) -- keep its `or` semantics.
        random_steps = int(
            getattr(agent.cfg, "start_steps", None) or getattr(agent.cfg, "learning_starts", 10_000)
        )
        update_after = int(
            getattr(agent.cfg, "update_after", None)
            or getattr(agent.cfg, "learning_starts", 10_000)
        )
        update_every = int(
            getattr(agent.cfg, "update_every", None) or getattr(agent.cfg, "train_freq", 1)
        )

        def _log_fn(step: int, metrics: dict) -> None:
            self.step = max(self.step, int(step))
            if not self._record_update(step, metrics):
                runner.request_stop()

        # `AsyncOffPolicyRunner` has no notion of eval scheduling or eval-env
        # construction -- it calls `eval_fn(step)` every collector-loop
        # iteration and leaves both to the caller (see its docstring).
        def _eval_fn(step: int) -> None:
            self.step = max(self.step, int(step))
            if not self._maybe_evaluate(step):
                runner.request_stop()

        # Without this, `train/score_mean`/`train/len` never show up on this
        # path: the runner owns its own collector loop.
        def _episode_fn(reward, done, truncated, step: int, info) -> None:
            self.logger.update_episodes(reward, done, truncated, step=step, info=info)

        eval_enabled = self.evaluator is not None and self.evaluator.enabled
        runner = runners.AsyncOffPolicyRunner(
            agent=agent,
            env=self.env,
            total_steps=self.total_steps - self.start_step,
            runner_cfg=runner_cfg,
            log_fn=_log_fn,
            device=str(agent.device),
            random_steps=random_steps,
            update_after=update_after,
            update_every=update_every,
            gradient_steps=gradient_steps,
            eval_fn=_eval_fn if eval_enabled else None,
            episode_fn=_episode_fn,
            start_step=self.start_step,
        )
        try:
            runner.run()
            self.step = max(self.step, self.total_steps)
        finally:
            # Force a final eval at the nominal last step if periodic eval
            # didn't already land exactly there, then close the eval env
            # exactly once -- in `finally` so a mid-run exception still
            # closes whatever eval env got built.
            try:
                self._final_evaluate(self.total_steps)
            finally:
                self._close_evaluator()

    # ------------------------------------------------------------------
    def _train_sync(self) -> None:
        from srl.core.her_replay_buffer import HERReplayBuffer

        agent, env = self.agent, self.env
        random_steps, update_after, update_every, gradient_steps = _schedule(agent.cfg)
        obs, reset_info = env.reset(seed=self.seed)
        encoder_names, encoder_input_names = encoder_layout(agent)
        # HER collection: episodes go in via add_transition() with the goal
        # vectors that GoalEnvWrapper stashes in info["goal_obs"], not via
        # buffer.add().
        her_buffer = agent.buffer if isinstance(agent.buffer, HERReplayBuffer) else None
        goal_obs = reset_info.get("goal_obs") if her_buffer is not None else None
        vectorized_env = self.vectorized_env
        step_increment = getattr(env, "num_envs", 1) if vectorized_env else 1
        since_last_update = 0
        reset_noise = getattr(agent, "reset_noise", None)

        try:
            while self.step < self.total_steps and not self.stop_training:
                remaining_steps = max(self.total_steps - self.step, 0)
                active_envs = min(step_increment, remaining_steps) if vectorized_env else 1
                obs_remapped = remap_obs_to_encoders(
                    obs, encoder_names, encoder_input_names=encoder_input_names
                )
                obs_t = obs_to_tensors(obs_remapped, agent.device, force_batch=not vectorized_env)
                if self.step < random_steps:
                    if vectorized_env:
                        # `env.act_space` on isaaclab/mjlab envs is already the
                        # BATCHED (num_envs, action_dim) space -- sample
                        # `single_act_space` (one env's space) per env, else
                        # stacking N samples gives (N, num_envs, action_dim).
                        per_env_space = getattr(env, "single_act_space", env.act_space)
                        action_np = np.stack(
                            [
                                sample_action_space(per_env_space)
                                for _ in range(getattr(env, "num_envs", 1))
                            ],
                            axis=0,
                        )
                    else:
                        action_np = sample_action_space(env.act_space)
                else:
                    action, _, _, _ = agent_act(agent, obs_t)
                    action_np = action.cpu().numpy()
                    if not vectorized_env and action_np.ndim > 1 and action_np.shape[0] == 1:
                        action_np = action_np.squeeze(0)

                next_obs, reward, done, trunc, info = env.step(action_np)
                self.step += active_envs
                since_last_update += active_envs
                log_reward, log_done, log_trunc, log_info = reward, done, trunc, info
                if vectorized_env and active_envs < step_increment:
                    log_reward = np.asarray(reward)[:active_envs]
                    log_done = np.asarray(done)[:active_envs]
                    log_trunc = np.asarray(trunc)[:active_envs]
                    log_info = list(info)[:active_envs]
                self.logger.update_episodes(
                    log_reward, log_done, log_trunc, step=self.step, info=log_info
                )
                # Per-episode exploration-noise reset (DDPG's OU process).
                if reset_noise is not None:
                    ended = np.logical_or(
                        np.atleast_1d(np.asarray(done)), np.atleast_1d(np.asarray(trunc))
                    )
                    if ended.any():
                        reset_noise(ended)
                if vectorized_env:
                    # `next_obs` is right for continuing the rollout, but
                    # wrong to STORE for any sub-env that just ended: for
                    # isaaclab/mjlab and Sync/AsyncVectorEnv it's a freshly
                    # reset episode's first frame, not the true terminal state
                    # the Q-target needs. `get_final_obs` splices every ended
                    # sub-env's true terminal observation back in.
                    store_next_obs = get_final_obs(next_obs, info)
                    transitions = split_vector_transition(
                        obs, store_next_obs, action_np, reward, done, trunc
                    )[:active_envs]
                    for env_index, (
                        obs_i,
                        next_obs_i,
                        action_i,
                        reward_i,
                        done_i,
                        trunc_i,
                    ) in enumerate(transitions):
                        agent.buffer.add(
                            obs=obs_i,
                            action=action_i,
                            reward=np.array([reward_i], dtype=np.float32),
                            done=np.array([done_i], dtype=bool),
                            truncated=np.array([trunc_i], dtype=bool),
                            next_obs=next_obs_i,
                            env_idx=env_index,
                        )
                elif her_buffer is not None:
                    next_goal_obs = info.get("goal_obs")
                    if next_goal_obs is None:
                        raise RuntimeError(
                            "HER is enabled but the environment did not provide "
                            "info['goal_obs']; expected a GoalEnvWrapper-wrapped env."
                        )
                    cur_obs_vec, cur_ag, cur_dg = her_goal_obs_parts(goal_obs)
                    next_obs_vec, next_ag, _ = her_goal_obs_parts(next_goal_obs)
                    her_buffer.add_transition(
                        obs=cur_obs_vec,
                        achieved_goal=cur_ag,
                        desired_goal=cur_dg,
                        action=np.asarray(action_np, dtype=np.float32).ravel(),
                        next_obs=next_obs_vec,
                        next_achieved_goal=next_ag,
                        # Real termination only; `truncated` closes the
                        # episode without fabricating a terminal state.
                        done=bool(done),
                        truncated=bool(trunc),
                    )
                    goal_obs = next_goal_obs
                else:
                    agent.buffer.add(
                        obs=obs,
                        action=action_np,
                        reward=np.array([reward], dtype=np.float32),
                        done=np.array([done], dtype=bool),
                        truncated=np.array([trunc], dtype=bool),
                        next_obs=next_obs,
                    )
                obs = next_obs
                if not vectorized_env and (done or trunc):
                    obs, reset_info = env.reset()
                    if her_buffer is not None:
                        goal_obs = reset_info.get("goal_obs")

                if not self._continue(self.callbacks.on_step(self.step)):
                    break

                if self.step >= update_after and since_last_update >= update_every:
                    update_span = since_last_update
                    metrics_list = []
                    for _ in range(gradient_steps):
                        metrics = agent.update()
                        if metrics:
                            metrics_list.append(metrics)
                    since_last_update = 0
                    if metrics_list:
                        sums: dict[str, float] = {}
                        counts: dict[str, int] = {}
                        for metric in metrics_list:
                            for key, value in metric.items():
                                sums[key] = sums.get(key, 0.0) + float(value)
                                counts[key] = counts.get(key, 0) + 1
                        merged = {key: sums[key] / counts[key] for key in sums}
                        merged["train/utd_ratio"] = gradient_steps / max(update_span, 1)
                        if her_buffer is not None:
                            # Makes it observable that HER is actually
                            # accumulating episodes.
                            merged["her/episodes"] = float(len(her_buffer))
                            merged["her/transitions"] = float(her_buffer.num_transitions)
                        if not self._record_update(self.step, merged):
                            break
                if not self._maybe_evaluate(self.step):
                    break

            self._final_evaluate(self.step)
        finally:
            self._close_evaluator()
