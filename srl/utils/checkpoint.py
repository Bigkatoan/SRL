"""CheckpointManager — save and load model or agent checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG SRL's training loops actually draw from.

    Without this, `--resume` restores model/optimizer/step state exactly but
    silently continues with a *different* exploration-noise and
    minibatch-shuffle sequence than an uninterrupted run would have had --
    the optimizer state and step count line up, but nothing downstream of
    them is actually reproducible across a resume.
    """
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    # torch.set_rng_state()/set_rng_state_all() both require CPU ByteTensors
    # regardless of which device the *model* weights are being restored to --
    # but CheckpointManager.load() calls torch.load(..., map_location=device),
    # which relocates every tensor in the payload (these included) onto that
    # device. On a CUDA resume, `.to(torch.uint8)` alone fixed the dtype but
    # left the tensor on cuda, and set_rng_state() rejected it outright
    # ("RNG state must be a torch.ByteTensor") -- --resume was unusable on
    # GPU. `.cpu()` first, independent of map_location.
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"].cpu().to(torch.uint8))
    if "torch_cuda_random" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in state["torch_cuda_random"]])


class CheckpointManager:
    """Save and load agent checkpoints.

    Uses ``safetensors`` when available, falls back to ``torch.save``.

    Parameters
    ----------
    save_dir:
        Directory where checkpoints are stored.
    max_keep:
        Keep only the last *max_keep* checkpoints.
    """

    def __init__(self, save_dir: str | Path, max_keep: int = 5) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._saved: list[Path] = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        metrics: dict[str, Any] | None = None,
        tag: str = "ckpt",
    ) -> Path:
        ckpt_path = self.save_dir / f"{tag}_{step:010d}.pt"
        payload = self._build_payload(model=model, optimizer=optimizer, step=step, metrics=metrics)

        if self._can_use_safetensors(model, payload, optimizer):
            try:
                from safetensors.torch import save_file as st_save

                st_path = ckpt_path.with_suffix(".safetensors")
                st_save(payload["model_state"], str(st_path))
                meta_path = ckpt_path.with_suffix(".meta.pt")
                meta_payload = {k: v for k, v in payload.items() if k != "model_state"}
                torch.save(meta_payload, meta_path)
                self._record(st_path)
                return st_path
            except (ImportError, Exception):
                pass

        torch.save(payload, ckpt_path)
        self._record(ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        model: Any,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file as st_load

            state = st_load(str(path), device=str(device))

            meta_path = path.with_suffix(".meta.pt")
            meta: dict[str, Any] = {}
            if meta_path.exists():
                meta = torch.load(meta_path, map_location=device, weights_only=False)
            payload = {"model_state": state, **meta}
            self._load_payload(model=model, payload=payload, optimizer=optimizer)
            return payload
        else:
            payload = torch.load(path, map_location=device, weights_only=False)
            self._load_payload(model=model, payload=payload, optimizer=optimizer)
            return payload

    def latest(self) -> Path | None:
        """Return the most recently saved checkpoint path."""
        if self._saved:
            return self._saved[-1]
        # Scan directory
        candidates = sorted(self.save_dir.glob("ckpt_*.pt")) + sorted(
            self.save_dir.glob("ckpt_*.safetensors")
        )
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, path: Path) -> None:
        self._saved.append(path)
        while len(self._saved) > self.max_keep:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink(missing_ok=True)
            # Remove companion meta file if present
            meta = old.with_suffix(".meta.pt")
            if meta.exists():
                meta.unlink(missing_ok=True)

    def _build_payload(
        self,
        *,
        model: Any,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if hasattr(model, "checkpoint_payload"):
            payload = dict(model.checkpoint_payload())
        elif isinstance(model, nn.Module):
            payload = {"model_state": model.state_dict()}
        else:
            raise TypeError(
                "CheckpointManager.save expects an nn.Module or object with checkpoint_payload()."
            )

        payload["step"] = step
        payload["metrics"] = metrics or {}
        payload["rng_state"] = _capture_rng_state()
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        return payload

    def _load_payload(
        self,
        *,
        model: Any,
        payload: dict[str, Any],
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        # Checkpoints saved before RNG state was tracked have no such key --
        # nothing to restore, resume just won't be bit-exact for those.
        if "rng_state" in payload:
            _restore_rng_state(payload["rng_state"])
        if hasattr(model, "load_checkpoint_payload"):
            model.load_checkpoint_payload(payload)
            return
        if not isinstance(model, nn.Module):
            raise TypeError(
                "CheckpointManager.load expects an nn.Module or object "
                "with load_checkpoint_payload()."
            )
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])

    def _can_use_safetensors(
        self, model: Any, payload: dict[str, Any], optimizer: torch.optim.Optimizer | None
    ) -> bool:
        return (
            isinstance(model, nn.Module)
            and optimizer is None
            and set(payload.keys()) <= {"model_state", "step", "metrics", "rng_state"}
        )


class BestCheckpointTracker:
    """Tracks the best value of a monitored eval metric across a run and
    saves a ``best_*`` checkpoint whenever a new best is seen.

    This exists because the *final* checkpoint a run produces is not always
    the *best* one -- PPO/A2C/SAC/... runs can (and on some tasks reliably
    do) peak partway through and then regress for the remainder of training
    (entropy collapse driving std/exploration to near-zero is one concrete
    way this happens on-policy). Without something tracking the best score
    independently, that peak is unrecoverable once training moves past it:
    only ``final_*``/periodic ``ckpt_*`` checkpoints exist, and by the time
    a run ends the periodic ones have long since rotated past whatever step
    the real peak was at.

    Deliberately keeps its own ``CheckpointManager`` (passed in per call, not
    owned) with ``max_keep=1`` rather than sharing the periodic-checkpoint
    manager's save/eviction FIFO: `CheckpointManager._record` evicts its
    oldest save once more than ``max_keep`` accumulate, tag-agnostically --
    sharing a manager between a periodic ``ckpt_*``/``final_*`` writer and
    this tracker would let ordinary periodic saves evict an old "best" file
    long before anything better has been found to replace it, silently
    defeating the entire point of this class.

    Parameters
    ----------
    mode:
        ``"max"`` (default) if higher monitored-metric values are better,
        ``"min"`` if lower is better.
    """

    def __init__(self, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.mode = mode
        self.best_score: float | None = None
        self.best_path: Path | None = None
        self.best_step: int | None = None

    def is_better(self, score: float) -> bool:
        """Whether *score* would improve on the best seen so far."""
        if self.best_score is None:
            return True
        return score > self.best_score if self.mode == "max" else score < self.best_score

    def seed_from_checkpoint(self, payload: dict[str, Any], monitor: str) -> None:
        """Seed ``best_score`` from a previously-saved best checkpoint's
        stored metrics -- used on ``--resume`` so a resumed run doesn't
        treat its first eval as automatically "best" and overwrite a
        genuinely-better pre-resume checkpoint with a worse one.
        """
        metrics = payload.get("metrics") or {}
        if monitor in metrics:
            self.best_score = float(metrics[monitor])
            self.best_step = int(payload.get("step", 0))

    def update(
        self,
        score: float,
        model: Any,
        *,
        cm: CheckpointManager,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        """Record *score*; if it's a new best, save via *cm* and return the
        checkpoint path. Returns None when *score* did not improve.
        """
        if not self.is_better(score):
            return None
        self.best_score = score
        self.best_step = step
        self.best_path = cm.save(model, optimizer=optimizer, step=step, metrics=metrics, tag="best")
        return self.best_path
