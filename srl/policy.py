"""Lightweight inference: :func:`load_policy` -> :class:`Policy`.

A :class:`Policy` is everything needed to turn a raw observation into an
action in env units -- the model, the observation normalizer statistics and
the action rescale map -- and nothing else (no env, optimizer, replay buffer
or algorithm object)::

    import srl

    policy = srl.load_policy("runs/pendulum")            # agent.save() directory
    policy = srl.load_policy("checkpoints/sac_pendulum")  # srl-train checkpoint dir
    action = policy(obs)                                  # numpy in, env units out

``policy.predict(obs)`` returns exactly what ``agent.predict(obs,
deterministic=True)[0]`` returns for the agent that was saved.

Command line::

    python -m srl.policy info   checkpoints/sac_pendulum
    python -m srl.policy export checkpoints/sac_pendulum --format torchscript -o policy.ts
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = ["Policy", "load_policy", "main"]

_ON_POLICY = ("ppo", "a2c", "a3c")
_DETERMINISTIC_ONLY = ("td3", "ddpg")


# ──────────────────────────────────────────────────────────────────────────────
# Observation normalization (same arithmetic as the agents, per encoder key)
# ──────────────────────────────────────────────────────────────────────────────


def _off_policy_stats(state: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``(mean, std)`` per key of a :class:`~srl.algorithms._common.ObsNormalizer` state."""
    eps = float(state.get("eps", 1e-2))
    stats = {}
    counts = state.get("count") or {}
    for key, mean in (state.get("mean") or {}).items():
        m2 = torch.as_tensor(state["m2"][key], dtype=torch.float64)
        var = (m2 / max(float(counts.get(key, 1.0)), 1.0)).to(torch.float32)
        std = torch.sqrt(var).clamp_min(eps)
        stats[key] = (torch.as_tensor(mean, dtype=torch.float64).to(torch.float32), std)
    return stats


def _ppo_stats(state: dict[str, Any], eps: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``(mean, std)`` per key of PPO's ``RunningNormalizer`` states."""
    stats = {}
    for key, norm in (state.get("obs") or {}).items():
        mean = np.asarray(norm["mean"], dtype=np.float64)
        var = np.asarray(norm["var"], dtype=np.float64)
        stats[key] = (
            torch.as_tensor(mean, dtype=torch.float32),
            torch.as_tensor(np.sqrt(var + eps), dtype=torch.float32),
        )
    return stats


class _ObsNormalization:
    """``clip((x - mean) / std, -clip, clip)`` on flat float observations, per key."""

    def __init__(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]], clip: float) -> None:
        self.stats = stats
        self.clip = float(clip)

    def to(self, device: torch.device) -> _ObsNormalization:
        self.stats = {k: (m.to(device), s.to(device)) for k, (m, s) in self.stats.items()}
        return self

    def __call__(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(obs)
        for key, value in obs.items():
            stat = self.stats.get(key)
            if stat is None or value.dim() != 2 or not value.is_floating_point():
                continue
            mean, std = stat
            out[key] = torch.clamp((value - mean) / std, -self.clip, self.clip)
        return out


def _algo_config_dict(ckpt: Any) -> dict[str, Any]:
    exp = ckpt.experiment
    if exp is not None and exp.algo_config is not None:
        import dataclasses

        return {
            f.name: getattr(exp.algo_config, f.name) for f in dataclasses.fields(exp.algo_config)
        }
    return dict(ckpt.agent_meta.get("algo_config") or {})


def _obs_normalization(ckpt: Any) -> _ObsNormalization | None:
    payload = ckpt.payload
    cfg = _algo_config_dict(ckpt)
    off = payload.get("obs_normalizer_state")
    if off is not None and off.get("mean"):
        return _ObsNormalization(_off_policy_stats(off), float(off.get("clip", 5.0)))
    ppo = payload.get("normalizers")
    if ppo and ppo.get("obs") and cfg.get("normalize_obs", True):
        eps = float(ppo.get("norm_epsilon", cfg.get("norm_epsilon", 1e-8)))
        clip = float(ppo.get("clip_obs", cfg.get("clip_obs", 10.0)))
        return _ObsNormalization(_ppo_stats(ppo, eps), clip)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────


class Policy:
    """Inference-only policy: raw observation -> action in env units.

    Build one with :func:`load_policy`. Calling it (or :meth:`predict`)
    applies, in order: the observation-to-encoder routing ``agent.predict``
    uses, the saved observation normalizer, the actor (deterministic: the
    distribution's mode), the training-time action clip and the
    ``[-1, 1]`` -> env-units rescale.

    Recurrent policies keep their hidden state between calls; :meth:`reset`
    clears it (call it at episode start).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        algo: str,
        obs_key: str = "state",
        obs_shape: tuple | None = None,
        action_rescale: dict | None = None,
        action_clip: dict | None = None,
        obs_normalization: _ObsNormalization | None = None,
        device: str | torch.device = "cpu",
        obs_spec: dict | None = None,
        action_spec: dict | None = None,
        experiment: Any = None,
        meta: dict | None = None,
        source: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model: Any = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.algo = str(algo or "").lower()
        self.obs_key = obs_key
        self.obs_shape = tuple(obs_shape) if obs_shape is not None else None
        self.action_rescale = action_rescale
        self._clip = None
        if action_clip is not None:
            self._clip = (
                np.asarray(action_clip["low"], dtype=np.float32),
                np.asarray(action_clip["high"], dtype=np.float32),
            )
        self.obs_normalization = obs_normalization.to(self.device) if obs_normalization else None
        self.obs_spec = obs_spec or {}
        self.action_spec = action_spec or {}
        #: The validated experiment config (``config.yaml``), when the checkpoint has one.
        self.experiment = experiment
        self.meta = meta or {}
        self.source = source
        self._hidden: Any = None
        self._warned_stochastic = False

    # -- construction ---------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt: Any, device: str | torch.device = "cpu") -> Policy:
        """Build from a :class:`srl.core.agent_io.Checkpoint`."""
        from srl.core.agent_io import obs_spec_from_model_config

        model = ckpt.build_model(device)
        ckpt.load_weights(model)
        am = ckpt.agent_meta
        obs_spec = ckpt.meta.get("obs_spec")
        if not obs_spec:
            try:
                model_cfg = ckpt.model_config
            except Exception:  # noqa: BLE001 -- informational only
                model_cfg = None
            obs_spec = obs_spec_from_model_config(
                model_cfg, am.get("obs_key", "state"), am.get("obs_shape")
            )
        bounds = ckpt.payload.get("action_bounds")
        action_spec = ckpt.meta.get("action_spec") or {
            "rescale": am.get("env_action_rescale"),
            "clip": bounds,
        }
        return cls(
            model,
            algo=ckpt.algo or "",
            obs_key=am.get("obs_key", "state"),
            obs_shape=am.get("obs_shape"),
            action_rescale=am.get("env_action_rescale"),
            action_clip=bounds,
            obs_normalization=_obs_normalization(ckpt),
            device=device,
            obs_spec=obs_spec,
            action_spec=action_spec,
            experiment=ckpt.experiment,
            meta=ckpt.meta,
            source=ckpt.path,
        )

    # -- inference --------------------------------------------------------
    def reset(self) -> None:
        """Forget the recurrent hidden state (no-op for feed-forward policies)."""
        self._hidden = None

    def to_tensors(self, observation: Any) -> tuple[dict[str, torch.Tensor], bool]:
        """Raw observation -> ``(encoder-keyed batched tensors, batched)``."""
        from srl.core.base_agent import observation_to_tensors

        return observation_to_tensors(
            observation,
            model=self.model,
            obs_key=self.obs_key,
            obs_shape=self.obs_shape,
            device=self.device,
        )

    def act_tensors(
        self, obs: dict[str, torch.Tensor], hidden: Any = None, deterministic: bool = True
    ) -> tuple[torch.Tensor, Any]:
        """Encoder-keyed batched tensors -> ``(action in the model's domain, new_hidden)``."""
        if self.obs_normalization is not None:
            obs = self.obs_normalization(obs)
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden, compute_critic=False)
            action = self._select(result["actor_out"], deterministic)
        return action, result.get("new_hidden")

    def _select(self, actor_out: Any, deterministic: bool) -> Any:
        if self.algo == "sac":
            from srl.algorithms.sac import sac_select_action

            return sac_select_action(actor_out, deterministic)[0]
        if self.algo in _DETERMINISTIC_ONLY:
            if not deterministic and not self._warned_stochastic:
                warnings.warn(
                    f"{self.algo.upper()} policies are deterministic; exploration noise is a "
                    "training-time detail and is not applied by Policy",
                    stacklevel=4,
                )
                self._warned_stochastic = True
            return actor_out.get("action") if isinstance(actor_out, dict) else actor_out
        from srl.algorithms._onpolicy_common import select_action

        return select_action(actor_out, deterministic)[0]

    def predict(self, observation: Any, deterministic: bool = True) -> np.ndarray:
        """Action (numpy, env units) for one observation or a batch of them.

        *observation* is what the env returns: an array (one observation or
        a leading batch dim) or a dict of arrays keyed by obs/encoder name.
        """
        obs_t, batched = self.to_tensors(observation)
        action, self._hidden = self.act_tensors(obs_t, self._hidden, deterministic)
        action_np = action.detach().cpu().numpy()
        if self._clip is not None:
            action_np = np.clip(action_np, self._clip[0], self._clip[1])
        from srl.core.base_agent import to_env_action

        action_np = to_env_action(action_np, self.action_rescale)
        return action_np if batched else action_np[0]

    __call__ = predict

    # -- metadata -----------------------------------------------------------
    @property
    def is_recurrent(self) -> bool:
        return any(enc.get("recurrent") for enc in (self.obs_spec.get("encoders") or {}).values())

    @property
    def input_keys(self) -> list[str]:
        """The observation keys :meth:`export_torchscript`'s module takes, in order."""
        if self.obs_shape is not None:
            return [self.obs_key]
        encoders = self.obs_spec.get("encoders") or {}
        actor_encoders = self.model.encoder_names_for_head("actor") or list(encoders)
        return [name for name in actor_encoders if name in encoders]

    def _example_inputs(self) -> tuple[torch.Tensor, ...]:
        encoders = self.obs_spec.get("encoders") or {}
        shapes = []
        for key in self.input_keys:
            shape = self.obs_shape if key == self.obs_key and self.obs_shape else None
            if shape is None:
                shape = (encoders.get(key) or {}).get("shape")
            if not shape:
                raise ValueError(
                    f"cannot export: the input shape of {key!r} is unknown; pass example_inputs="
                )
            shapes.append(tuple(shape))
        return tuple(torch.zeros((1, *s), device=self.device) for s in shapes)

    # -- export ---------------------------------------------------------------
    def export_torchscript(
        self, path: str | Path, example_inputs: tuple[torch.Tensor, ...] | None = None
    ) -> Path:
        """Trace the full deterministic policy (normalize -> actor -> clip -> rescale).

        The saved module takes one batched ``float32`` tensor per entry of
        :attr:`input_keys` (images already ``(C, H, W)``) and returns the
        action in env units; load it with ``torch.jit.load(path)`` -- no SRL
        import needed. Recurrent policies are not supported.
        """
        if self.is_recurrent:
            raise NotImplementedError("TorchScript export of recurrent policies is not supported")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        module = _ExportedPolicy(self).eval()
        example = example_inputs if example_inputs is not None else self._example_inputs()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            traced = torch.jit.trace(module, example, check_trace=False)
        traced.save(str(path))
        return path

    def __repr__(self) -> str:
        src = f" from {self.source}" if self.source is not None else ""
        return f"Policy(algo={self.algo!r}, inputs={self.input_keys}, device={self.device}{src})"


class _ExportedPolicy(nn.Module):
    """``Policy`` as a pure tensor module (what :meth:`Policy.export_torchscript` traces)."""

    clip_low: torch.Tensor
    clip_high: torch.Tensor
    r_low: torch.Tensor
    r_high: torch.Tensor
    r_finite: torch.Tensor

    def __init__(self, policy: Policy) -> None:
        super().__init__()
        self.policy = policy
        self.model = policy.model
        self.keys = policy.input_keys
        clip = policy._clip
        self.register_buffer("clip_low", torch.as_tensor(clip[0]) if clip else torch.empty(0))
        self.register_buffer("clip_high", torch.as_tensor(clip[1]) if clip else torch.empty(0))
        rescale = policy.action_rescale
        if rescale is not None:
            self.register_buffer(
                "r_low", torch.as_tensor(np.asarray(rescale["low"]), dtype=torch.float32)
            )
            self.register_buffer(
                "r_high", torch.as_tensor(np.asarray(rescale["high"]), dtype=torch.float32)
            )
            self.register_buffer(
                "r_finite", torch.as_tensor(np.asarray(rescale["finite"], dtype=bool))
            )
        self.has_rescale = rescale is not None

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        from srl.training._obs import remap_obs_to_encoders

        obs = dict(zip(self.keys, inputs, strict=True))
        obs = remap_obs_to_encoders(
            obs, list(self.model.encoders.keys()), getattr(self.model, "encoder_input_names", None)
        )
        action, _ = self.policy.act_tensors(obs, None, True)
        if self.clip_low.numel():
            action = torch.max(torch.min(action, self.clip_high), self.clip_low)
        if self.has_rescale:
            scaled = self.r_low + (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.r_high - self.r_low)
            action = torch.where(self.r_finite, scaled, action)
        return action


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def resolve_policy_path(path: str | Path, checkpoint: str = "auto") -> Path:
    """An agent directory / checkpoint file for *path*.

    *path* may be an ``agent.save()`` directory, a legacy ``.pt`` checkpoint,
    or an ``srl-train`` checkpoint directory (``checkpoints/<run>/``), in
    which case *checkpoint* picks ``"best"``, ``"final"``, ``"latest"`` or
    ``"auto"`` (best, else final, else latest).
    """
    from srl.core.agent_io import find_checkpoint, is_agent_dir

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No SRL policy at {path}")
    if path.is_dir() and not is_agent_dir(path):
        return Path(find_checkpoint(path, checkpoint))
    return path


def load_policy(
    path: str | Path, device: str | torch.device = "cpu", *, checkpoint: str = "auto"
) -> Policy:
    """Load a :class:`Policy` for inference (see :func:`resolve_policy_path` for *path*)."""
    from srl.core.agent_io import read_checkpoint

    target = resolve_policy_path(path, checkpoint)
    ckpt = read_checkpoint(target, map_location=device, inference_only=True)
    return Policy.from_checkpoint(ckpt, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: python -m srl.policy {info,export}
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m srl.policy", description="Inspect or export a saved SRL policy."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="print the checkpoint's metadata")
    info.add_argument("path")
    info.add_argument("--checkpoint", default="auto", choices=["auto", "best", "final", "latest"])
    export = sub.add_parser("export", help="export the deterministic policy")
    export.add_argument("path")
    export.add_argument("--format", default="torchscript", choices=["torchscript"])
    export.add_argument("-o", "--output", default=None, help="default: <checkpoint>/policy.ts")
    export.add_argument("--checkpoint", default="auto", choices=["auto", "best", "final", "latest"])
    args = parser.parse_args(argv)

    policy = load_policy(args.path, checkpoint=args.checkpoint)
    if args.command == "info":
        summary = {
            "source": str(policy.source),
            "algo": policy.algo,
            "inputs": policy.input_keys,
            "obs_spec": policy.obs_spec,
            "action_spec": policy.action_spec,
            "format_version": policy.meta.get("format_version", 0),
            "num_timesteps": policy.meta.get("num_timesteps"),
        }
        print(json.dumps(summary, indent=2, default=str))
        return 0
    source = Path(policy.source) if policy.source is not None else Path(".")
    default = (source if source.is_dir() else source.parent) / "policy.ts"
    out = policy.export_torchscript(args.output or default)
    print(f"[srl.policy] wrote {out} (inputs: {policy.input_keys})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
