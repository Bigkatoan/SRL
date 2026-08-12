"""CheckpointManager — save and load model or agent checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


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
            and set(payload.keys()) <= {"model_state", "step", "metrics"}
        )
