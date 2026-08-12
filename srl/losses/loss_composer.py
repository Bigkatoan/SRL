"""LossComposer — weighted sum of named loss terms with optional schedules."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


class LossComposer:
    """Combine multiple loss terms with configurable weights.

    Usage::

        composer = LossComposer()
        composer.add("policy", weight=1.0)
        composer.add("value", weight=0.5)
        composer.add("entropy", weight=0.01)
        composer.add("reconstruction", weight=0.1, schedule="cosine", total_steps=1_000_000)

        total, info = composer.compute(
            policy=ppo_clip_loss(...),
            value=ppo_value_loss(...),
            entropy=entropy_loss(...),
            reconstruction=reconstruction_loss(...),
            step=current_step,
        )
    """

    def __init__(self) -> None:
        self._terms: dict[str, dict] = {}

    def add(
        self,
        name: str,
        weight: float = 1.0,
        schedule: str = "constant",
        total_steps: int = 1_000_000,
        min_weight: float = 0.0,
        custom_fn: Callable[[int], float] | None = None,
    ) -> LossComposer:
        self._terms[name] = {
            "weight": weight,
            "schedule": schedule,
            "total_steps": total_steps,
            "min_weight": min_weight,
            "custom_fn": custom_fn,
        }
        return self

    def _effective_weight(self, name: str, step: int) -> float:
        t = self._terms[name]
        base = t["weight"]
        sched = t["schedule"]
        total = t["total_steps"]
        min_w = t["min_weight"]
        fn = t.get("custom_fn")

        if fn is not None:
            return fn(step)
        if sched == "constant":
            return base
        progress = min(step / max(total, 1), 1.0)
        if sched == "linear_decay":
            return base * (1.0 - progress) + min_w * progress
        if sched == "cosine":
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_w + (base - min_w) * cosine_val
        return base

    def compute(
        self, step: int = 0, **losses: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted sum.

        Parameters
        ----------
        step:
            Current training step (used by schedules).
        **losses:
            Keyword arguments matching registered term names.

        Returns
        -------
        (total_loss, info_dict)
        """
        total: torch.Tensor | None = None
        info: dict[str, float] = {}

        for name, tensor in losses.items():
            w = self._effective_weight(name, step) if name in self._terms else 1.0
            term = w * tensor
            info[name] = tensor.item()
            info[f"{name}_weight"] = w
            total = term if total is None else total + term

        if total is None:
            raise ValueError("No loss terms provided.")

        info["total"] = total.item()
        return total, info

    @classmethod
    def from_loss_configs(cls, loss_configs: list) -> LossComposer:
        """Build from a list of :class:`~srl.registry.config_schema.LossConfig`."""
        composer = cls()
        for lc in loss_configs:
            composer.add(
                name=lc.name,
                weight=lc.weight,
                schedule=lc.schedule,
            )
        return composer
