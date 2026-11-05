"""A3C (Asynchronous Advantage Actor-Critic) — multi-process on-policy.

This implementation uses Python's ``multiprocessing`` to launch worker
processes that collect trajectories and compute gradients locally.
Gradients are pushed to a shared model via ``apply_async``.

Usage::

    agent = A3C.from_config("my_a3c.yaml", env=gym.make("Pendulum-v1"))
    agent.learn(1_000_000)          # worker envs are rebuilt from the env's id

    # or with an explicit (picklable) worker env factory:
    agent.learn(1_000_000, env_fn=functools.partial(srl.training.make_env, "Pendulum-v1"))
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from collections.abc import Callable

import torch
import torch.nn as nn

from srl.core.base_agent import OnPolicyAgent
from srl.core.config import A3CConfig
from srl.losses.rl_losses import a2c_policy_loss, a2c_value_loss, entropy_loss


def _worker_fn(
    rank: int,
    shared_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    env_fn: Callable,
    config: A3CConfig,
    counter: mp.Value,
    lock: mp.Lock,
    stop_event: mp.Event,
    stats_queue: mp.Queue | None = None,
) -> None:
    """Worker process: collect a trajectory, compute gradients, apply to shared model."""
    import numpy as np
    import torch

    from srl.algorithms._onpolicy_common import env_action_bounds
    from srl.core.rollout_buffer import RolloutBuffer

    # Derive every worker's seed from the run's global seed (previously
    # `rank` alone, so --seed had no effect and every run was identical).
    worker_seed = int(getattr(config, "seed", 0) or 0) + rank
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    env = env_fn()
    local_model = _clone_model(shared_model)
    bounds = env_action_bounds(env)

    buffer = RolloutBuffer(
        capacity=config.n_steps,
        num_envs=1,
        gamma=config.gamma,
        lam=config.gae_lambda,
    )

    try:
        obs, _ = env.reset(seed=worker_seed)
    except TypeError:  # minimal envs whose reset() takes no seed
        obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0

    while not stop_event.is_set():
        # Sync local weights from shared model
        local_model.load_state_dict(shared_model.state_dict())

        buffer.reset()

        for _ in range(config.n_steps):
            obs_t = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                result = local_model(obs_t)
            actor_out = result["actor_out"]
            if isinstance(actor_out, dict):
                action = actor_out.get("action")
                log_prob = actor_out.get("log_prob")
            elif isinstance(actor_out, tuple):
                action, log_prob = actor_out
            else:
                action, log_prob = actor_out, None
            value = result["value"]

            action_np = action.squeeze(0).numpy()
            # Clip only what reaches the env; the buffer keeps the raw sample
            # so its stored log-prob stays consistent (SB3/CleanRL ClipAction).
            env_action = np.clip(action_np, *bounds) if bounds is not None else action_np
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            terminated, truncated = bool(terminated), bool(truncated)
            done = terminated or truncated
            reward_f = float(reward)
            episode_return += reward_f
            episode_length += 1

            # Time-limit truncation: the episode is cut, not finished, so add
            # back the value of the true final state (the env does not
            # auto-reset, so next_obs IS that state) before the GAE chain is
            # broken at this step.
            train_reward = reward_f
            if truncated and not terminated:
                next_t = {
                    k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0)
                    for k, v in next_obs.items()
                }
                with torch.no_grad():
                    boot = local_model(next_t)["value"]
                if boot is not None:
                    train_reward += config.gamma * float(boot.squeeze().item())

            buffer.add(
                obs=obs,
                action=action_np,
                reward=train_reward,
                done=done,
                log_prob=log_prob.squeeze(0).numpy() if log_prob is not None else None,
                value=value.squeeze(0).numpy() if value is not None else None,
            )
            obs = next_obs
            if done:
                if stats_queue is not None:
                    stats_queue.put(
                        {
                            "type": "episode",
                            "step": counter.value,
                            "score": episode_return,
                            "episode_length": episode_length,
                            "worker": rank,
                        }
                    )
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()

        # Compute last value for bootstrap
        obs_t = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            last_val = local_model(obs_t)["value"]
        last_val_f = last_val.squeeze().item() if last_val is not None else 0.0
        buffer.compute_returns_and_advantages(last_value=last_val_f)

        # Compute gradients locally
        local_model.train()
        losses = []
        metric_lists: dict[str, list[float]] = {
            "policy": [],
            "value": [],
            "entropy": [],
            "total": [],
        }
        for mini in buffer.get_batches(config.batch_size):
            obs_b = {k: v for k, v in mini.obs.items()}
            # actor_action=mini.actions: re-evaluate the action actually taken,
            # not a freshly resampled one -- see PPO.update()'s comment on the
            # same mechanism (AgentModel.forward's evaluate_actions() path).
            result = local_model(obs_b, actor_action=mini.actions)
            actor_out = result["actor_out"]
            if isinstance(actor_out, dict):
                log_prob = actor_out.get("log_prob")
                ent = actor_out.get("entropy")
                if ent is None:
                    dist = actor_out.get("dist")
                    ent = dist.entropy() if dist is not None else torch.zeros(1)
            elif isinstance(actor_out, tuple):
                _, log_prob = actor_out
                ent = torch.zeros(1)
            else:
                log_prob = actor_out
                ent = torch.zeros(1)

            adv = mini.advantages
            pol_loss = a2c_policy_loss(log_prob, adv)
            val_loss = a2c_value_loss(result["value"].squeeze(-1), mini.returns)
            ent_loss = entropy_loss(ent)
            total = pol_loss + config.vf_coef * val_loss + config.entropy_coef * ent_loss
            losses.append(total)
            metric_lists["policy"].append(float(pol_loss.item()))
            metric_lists["value"].append(float(val_loss.item()))
            metric_lists["entropy"].append(float(ent_loss.item()))
            metric_lists["total"].append(float(total.item()))

        if losses:
            loss = sum(losses) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), config.max_grad_norm)

            # Copy gradients to shared model
            with lock:
                for sp, lp in zip(shared_model.parameters(), local_model.parameters(), strict=True):
                    if lp.grad is not None:
                        sp.grad = lp.grad.clone()
                optimizer.step()

            with lock:
                counter.value += config.n_steps
            if stats_queue is not None:
                stats_queue.put(
                    {
                        "type": "metrics",
                        "step": counter.value,
                        "metrics": {k: sum(v) / len(v) for k, v in metric_lists.items() if v},
                    }
                )

    env.close()


def _clone_model(src: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(src)


class A3C(OnPolicyAgent):
    """Launch N worker processes and train asynchronously.

    Each worker builds its own env from a picklable zero-arg factory
    (``env_fn``); :meth:`learn` derives one from the bound env's Gymnasium id
    when no ``env_fn`` is given.
    """

    algo_name = "a3c"
    config_cls = A3CConfig

    def __init__(
        self,
        model: nn.Module,
        config: A3CConfig | None = None,
        device: str | torch.device = "cpu",
        *,
        env=None,
        env_fn: Callable | None = None,
    ) -> None:
        if torch.device(device).type != "cpu":
            # Workers are CPU processes sharing the model's memory; the model
            # has always stayed on the CPU (only the device label differed).
            device = "cpu"
        super().__init__(model, config, device, env=env)
        self.model.share_memory()
        self.env_fn = env_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _fit_config_to_env(self, env) -> None:
        # n_workers is the parallelism knob for A3C, not the env's num_envs.
        return None

    def learn(
        self,
        total_timesteps: int,
        env=None,
        eval_env=None,
        callbacks=None,
        log_dir: str | None = None,
        progress: bool = False,
        *,
        env_fn: Callable | None = None,
        eval_freq: int | None = None,
        n_eval_episodes: int = 5,
        seed: int | None = None,
        reset_num_timesteps: bool = True,
        logger=None,
        evaluator=None,
        log_interval: int = 1_000,
        env_type: str | None = None,
    ) -> A3C:
        """Train with ``cfg.n_workers`` worker processes (delegates to :meth:`train`).

        Same arguments as :meth:`OnPolicyAgent.learn`, plus *env_fn* (a
        picklable zero-arg env factory for the workers). The workers report
        to the logger; ``on_step``/``on_rollout_end``/``on_update`` callbacks
        are not called (training happens in other processes), but
        ``on_training_start``/``on_eval``/``on_training_end`` are, and
        evaluation runs once at the end.
        """
        from srl.training.callbacks import CallbackList
        from srl.training.envs import env_factory_for

        if env is not None:
            if callable(env) and not hasattr(env, "reset"):
                env_fn, env = env, None
            else:
                self.set_env(env)
        env_fn = env_fn or self.env_fn
        if env_fn is None and self.env is not None:
            env_fn = env_factory_for(self.env, env_type or self._env_type or "flat")
        if env_fn is None:
            raise ValueError(
                "A3C.learn() needs a picklable env factory for its worker processes: pass "
                "env_fn=... (or an env created with gym.make so its id can be reused)."
            )
        setup = self._build_learn_setup(
            total_timesteps,
            eval_env=eval_env,
            callbacks=callbacks,
            log_dir=log_dir,
            progress=progress,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
            logger=logger,
            evaluator=evaluator,
            reset_num_timesteps=reset_num_timesteps,
        )
        cbs = CallbackList(callbacks)
        trainer = _A3CTrainerView(self, setup)
        cbs.on_training_start(trainer)
        try:
            self.train(
                total_timesteps=setup["total_steps"],
                env_fn=env_fn,
                logger=setup["logger"],
                log_interval=log_interval,
                seed=seed,
                start_step=setup["start_step"],
            )
            trainer.step = self.num_timesteps
            evaluator = setup["evaluator"]
            if evaluator is not None:
                try:
                    metrics = evaluator.finalize(
                        self, trainer.step, logger=setup["logger"], total_steps=trainer.total_steps
                    )
                    if metrics is not None:
                        cbs.on_eval(trainer.step, metrics)
                finally:
                    evaluator.close()
        finally:
            cbs.on_training_end(trainer)
        return self

    def train(
        self,
        total_timesteps: int,
        env_fn: Callable,
        logger=None,
        log_interval: int = 1_000,
        seed: int | None = None,
        start_step: int = 0,
    ) -> None:
        """Run the workers until the shared step counter reaches *total_timesteps*."""
        if seed is not None:
            self.cfg.seed = int(seed)
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", int(start_step))
        lock = ctx.Lock()
        stop_event = ctx.Event()
        stats_queue = ctx.Queue() if logger is not None else None

        workers = []
        for rank in range(self.cfg.n_workers):
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    rank,
                    self.model,
                    self.optimizer,
                    env_fn,
                    self.cfg,
                    counter,
                    lock,
                    stop_event,
                    stats_queue,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        last_logged_step = 0
        while counter.value < total_timesteps:
            if logger is not None and stats_queue is not None:
                while True:
                    try:
                        event = stats_queue.get_nowait()
                    except queue.Empty:
                        break
                    if event.get("type") == "episode":
                        logger.record_episode(
                            step=int(event.get("step", counter.value)),
                            score=float(event["score"]),
                            length=int(event["episode_length"]),
                        )
                    elif event.get("type") == "metrics":
                        logger.record_metrics(
                            event.get("metrics", {}),
                            step=int(event.get("step", counter.value)),
                            total_steps=total_timesteps,
                            prefix="a3c",
                        )
                if counter.value - last_logged_step >= log_interval:
                    logger.record_metrics(
                        {
                            "global_steps": float(counter.value),
                            "workers": float(self.cfg.n_workers),
                        },
                        step=int(counter.value),
                        total_steps=total_timesteps,
                        prefix="a3c",
                    )
                    last_logged_step = counter.value
            time.sleep(0.2)

        stop_event.set()
        for p in workers:
            p.join(timeout=30)
        if logger is not None and stats_queue is not None:
            while True:
                try:
                    event = stats_queue.get_nowait()
                except queue.Empty:
                    break
                if event.get("type") == "episode":
                    logger.record_episode(
                        step=int(event.get("step", counter.value)),
                        score=float(event["score"]),
                        length=int(event["episode_length"]),
                    )
                elif event.get("type") == "metrics":
                    logger.record_metrics(
                        event.get("metrics", {}),
                        step=int(event.get("step", counter.value)),
                        total_steps=total_timesteps,
                        prefix="a3c",
                    )
        self._global_step = counter.value
        self.num_timesteps = int(counter.value)

    def update(self) -> dict[str, float]:
        """No-op: A3C updates happen inside the worker processes."""
        return {}

    def _algo_payload(self) -> dict[str, object]:
        return self._base_payload()

    def _load_algo_payload(self, payload: dict[str, object]) -> None:
        self._load_base_payload(payload)


class _A3CTrainerView:
    """Minimal trainer-shaped object handed to callbacks by :meth:`A3C.learn`."""

    def __init__(self, agent: A3C, setup: dict) -> None:
        self.agent = agent
        self.env = agent.env
        self.logger = setup["logger"]
        self.total_steps = setup["total_steps"]
        self.start_step = setup["start_step"]
        self.step = setup["start_step"]
        self.evaluator = setup["evaluator"]
        self.env_type = agent._env_type
        self.stop_training = False
