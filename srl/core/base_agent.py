"""BaseAgent — abstract interface every SRL algorithm implements."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np


class BaseAgent(abc.ABC):
    """Abstract base for all SRL agents.

    Subclasses implement :meth:`learn`, :meth:`predict`, and the internal
    ``_train_step`` loop.  Save/load delegate to :class:`~srl.utils.checkpoint.CheckpointManager`.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def learn(self, total_steps: int, callback=None) -> BaseAgent:
        """Run the main training loop for *total_steps* environment steps."""

    @abc.abstractmethod
    def predict(
        self,
        observation: np.ndarray | dict,
        deterministic: bool = False,
        state: tuple | None = None,
    ) -> tuple[np.ndarray, tuple | None]:
        """Return *(action, next_recurrent_state)* for a single observation.

        ``state`` carries LSTM hidden states ``(h, c)`` for recurrent policies.
        Pass ``None`` for non-recurrent policies.
        """

    def save(self, path: str | Path) -> None:
        """Save checkpoint via :class:`~srl.utils.checkpoint.CheckpointManager`."""
        from srl.utils.checkpoint import CheckpointManager

        path = Path(path)
        CheckpointManager(path.parent).save(self, tag=path.stem)

    def load(self, path: str | Path) -> BaseAgent:
        """Load checkpoint in-place and return *self*."""
        from srl.utils.checkpoint import CheckpointManager

        CheckpointManager(Path(path).parent).load(self, path)
        return self

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> None:  # noqa: B027 (intentional no-op hook)
        """Called after every environment step. Override for custom logic."""
        pass

    def _on_episode_end(self) -> None:  # noqa: B027 -- see _on_step
        """Called at episode boundaries. Override for custom logic."""
        pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def device(self):
        return self._device

    def _as_tensor(self, arr: np.ndarray | Any):
        """Convert numpy array to a tensor on this agent's device."""
        import torch

        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float().to(self._device)
        return arr.to(self._device)
