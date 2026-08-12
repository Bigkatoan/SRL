"""A3C (Asynchronous Advantage Actor-Critic) — multi-process on-policy.

This implementation uses Python's ``multiprocessing`` to launch worker
processes that collect trajectories and compute gradients locally.
Gradients are pushed to a shared model via ``apply_async``.

Usage::

    shared_model = ModelBuilder.from_yaml("configs/ant.yaml")
    shared_model.share_memory()

    a3c = A3C(shared_model, config=A3CConfig(num_workers=4))
    a3c.train(total_timesteps=1_000_000, env_fn=lambda: GymnasiumWrapper(gym.make("Ant-v4")))
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from collections.abc import Callable

import torch
import torch.nn as nn

from srl.core.base_agent import BaseAgent
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
    import torch

    from srl.core.rollout_buffer import RolloutBuffer

    torch.manual_seed(rank)
    env = env_fn()
    local_model = _clone_model(shared_model)

    buffer = RolloutBuffer(
        capacity=config.n_steps,
        num_envs=1,
        gamma=config.gamma,
        lam=config.gae_lambda,
    )

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
            next_obs, reward, done, _, _ = env.step(action_np)
            reward_f = float(reward)
            episode_return += reward_f
            episode_length += 1

            buffer.add(
                obs=obs,
                action=action_np,
                reward=reward_f,
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
            result = local_model(obs_b)
            actor_out = result["actor_out"]
            if isinstance(actor_out, dict):
                log_prob = actor_out.get("log_prob")
            elif isinstance(actor_out, tuple):
                _, log_prob = actor_out
            else:
                log_prob = actor_out

            adv = mini.advantages
            pol_loss = a2c_policy_loss(log_prob, adv)
            val_loss = a2c_value_loss(result["value"].squeeze(-1), mini.returns)
            ent = torch.zeros(1)
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


class A3C(BaseAgent):
    """Launch N worker processes and train asynchronously."""

    def __init__(
        self,
        model: nn.Module,
        config: A3CConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.cfg = config or A3CConfig()
        self._device = torch.device(device)
        self.model.share_memory()
        self._global_step = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def train(
        self, total_timesteps: int, env_fn: Callable, logger=None, log_interval: int = 1_000
    ) -> None:
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
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

    def predict(self, obs, hidden=None, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
        actor_out = result["actor_out"]
        if isinstance(actor_out, dict):
            action = (
                actor_out.get("mean", actor_out.get("action"))
                if deterministic
                else actor_out.get("action")
            )
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            action, log_prob = actor_out
        else:
            action, log_prob = actor_out, None
        return action, log_prob, result["value"], result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError("Use A3C.train(total_timesteps, env_fn=...)")

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        torch.save(self.checkpoint_payload(), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.load_checkpoint_payload(ckpt)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_state": self.model.state_dict(),
            "algo_step": self._global_step,
        }

    def load_checkpoint_payload(self, payload: dict[str, object]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
