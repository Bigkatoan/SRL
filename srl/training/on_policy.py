"""On-policy rollout/update loop (PPO, A2C).

Moved verbatim (modulo naming) from ``srl/cli/train.py``'s
``_run_on_policy`` so its Phase-0-verified behaviour is preserved: action
clipping at the env boundary, time-limit bootstrap from the TRUE final obs,
GAE chain breaks at any episode end, per-episode reset of a single
unvectorized env, optional obs/reward normalisation.
"""

from __future__ import annotations

import math

import numpy as np

from srl.envs.vector_utils import get_final_obs
from srl.training._obs import agent_act, encoder_layout, obs_to_tensors, remap_obs_to_encoders
from srl.training.base import Trainer

__all__ = ["OnPolicyTrainer"]


class OnPolicyTrainer(Trainer):
    """Collect ``n_steps`` per env, compute GAE, ``agent.update()``, repeat.

    See :class:`~srl.training.base.Trainer` for the constructor arguments.
    """

    def _train(self) -> None:
        agent, env = self.agent, self.env
        n_steps = agent.cfg.n_steps
        num_envs = max(getattr(agent.cfg, "num_envs", 1), 1)
        obs, _ = env.reset(seed=self.seed)
        # Gaussian policies sample unbounded actions: clip them to the env's
        # action bounds at env.step() only (the unclipped sample is what the
        # buffer stores for the log-prob) -- SB3/CleanRL ClipAction semantics.
        # Deterministic act() clips to the same bounds itself.
        if hasattr(agent, "set_action_bounds_from_env"):
            agent.set_action_bounds_from_env(env)
        clip_action = getattr(agent, "clip_action", None)
        # isaaclab/mjlab and SyncVectorEnv auto-reset finished sub-envs
        # themselves; a single plain GymnasiumWrapper (n_envs=1) does NOT, so
        # this loop must reset it on every episode end -- otherwise every step
        # after the first episode is taken in a dead env.
        vectorized_env = self.vectorized_env
        # PPO's optional VecNormalize-style normalisation (PPOConfig.
        # normalize_obs / normalize_reward). Obs stats are updated on every
        # collected observation BEFORE act() (which applies them itself), and
        # the buffer stores the normalised obs so update() trains on exactly
        # what the policy saw. Rewards are scaled before the truncation
        # bootstrap below so V (learned on the scaled reward) is added on the
        # same scale.
        update_obs_norm = getattr(agent, "update_obs_normalizer", None)
        normalize_obs = getattr(agent, "normalize_obs", None)
        normalize_reward = getattr(agent, "normalize_reward", None)
        encoder_names, encoder_input_names = encoder_layout(agent)
        gamma = float(getattr(agent.cfg, "gamma", 0.99))

        def _to_model_input(raw_obs):
            remapped = remap_obs_to_encoders(
                raw_obs, encoder_names, encoder_input_names=encoder_input_names
            )
            return remapped, obs_to_tensors(remapped, agent.device, force_batch=False)

        try:
            while self.step < self.total_steps and not self.stop_training:
                remaining_steps = max(self.total_steps - self.step, 0)
                rollout_steps = min(n_steps, max(1, math.ceil(remaining_steps / num_envs)))
                for _ in range(rollout_steps):
                    obs_remapped = remap_obs_to_encoders(
                        obs, encoder_names, encoder_input_names=encoder_input_names
                    )
                    if update_obs_norm is not None:
                        update_obs_norm(obs_remapped)
                    obs_t = obs_to_tensors(obs_remapped, agent.device, force_batch=False)
                    action, log_prob, value, _ = agent_act(agent, obs_t)
                    obs_to_store = normalize_obs(obs_remapped) if normalize_obs is not None else obs
                    action_np = action.cpu().numpy()
                    env_action = clip_action(action_np) if clip_action is not None else action_np
                    next_obs, reward, done, trunc, info = env.step(env_action)
                    self.logger.update_episodes(reward, done, trunc, step=self.step, info=info)

                    # Correct time-limit truncation handling. `next_obs` is
                    # what to ACT on next, but is NOT what a value bootstrap
                    # at a truncation should use: for isaaclab/mjlab it's a
                    # different, freshly-reset episode's first frame.
                    # `get_final_obs` (info["final_obs"], populated by every
                    # SRL vector env; for a single-env GymnasiumWrapper,
                    # `next_obs` IS already the true final obs since it never
                    # auto-resets) is the real terminal state to bootstrap
                    # from. Only pay for the extra forward pass on steps
                    # where something actually timed out.
                    reward_arr = np.asarray(reward, dtype=np.float32)
                    done_arr = np.asarray(done, dtype=bool)
                    trunc_arr = np.asarray(trunc, dtype=bool)
                    timeout_mask = trunc_arr & ~done_arr
                    if normalize_reward is not None:
                        reward_arr = normalize_reward(reward_arr, done_arr | trunc_arr)
                    if timeout_mask.any():
                        _, tf_t = _to_model_input(get_final_obs(next_obs, info))
                        _, _, bootstrap_value, _ = agent_act(agent, tf_t)
                        if bootstrap_value is not None:
                            bv = bootstrap_value.detach().cpu().numpy().reshape(-1)
                            reward_arr = reward_arr + gamma * bv * timeout_mask.astype(np.float32)

                    agent.buffer.add(
                        obs=obs_to_store,
                        action=action_np,
                        reward=reward_arr,
                        # `terminated OR truncated`, not `terminated` alone --
                        # compute_returns_and_advantages() uses this to break
                        # the GAE chain at ANY episode boundary (the
                        # `timeout_mask` correction above already adds back
                        # what continuing would have been worth).
                        done=np.asarray(done_arr | trunc_arr),
                        log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                        value=value.cpu().numpy() if value is not None else None,
                    )
                    obs = next_obs
                    if not vectorized_env and bool(np.any(done_arr | trunc_arr)):
                        obs, _ = env.reset()
                    self.step += getattr(agent.cfg, "num_envs", 1)
                    # A stop requested mid-rollout is honoured after the
                    # update below (the buffer must not be left half-full).
                    self._continue(self.callbacks.on_step(self.step))

                _, last_t = _to_model_input(obs)
                _, _, last_val, _ = agent_act(agent, last_t)
                agent.buffer.compute_returns_and_advantages(
                    last_value=last_val.cpu().numpy() if last_val is not None else 0.0
                )
                self._continue(self.callbacks.on_rollout_end(self.step))
                metrics = agent.update()
                if not self._record_update(self.step, metrics):
                    break
                if not self._maybe_evaluate(self.step):
                    break

            self._final_evaluate(self.step)
        finally:
            self._close_evaluator()
