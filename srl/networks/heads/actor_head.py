"""Actor heads for continuous action spaces."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from srl.networks.distributions import DiagonalGaussian, SquashedGaussian
from srl.networks.layers.mlp_builder import build_mlp


class DeterministicActorHead(nn.Module):
    """Deterministic policy for DDPG: μ(s) ∈ [-1, 1]^action_dim."""

    type_name = "deterministic"

    def __init__(self, input_dim: int, action_dim: int, layer_configs: list, **kw):
        super().__init__()
        mlp_kw = {
            k: v
            for k, v in kw.items()
            if k
            in (
                "default_activation",
                "default_norm",
                "default_dropout",
                "norm_order",
                "weight_init",
            )
        }
        self.net, hid = build_mlp(layer_configs, input_dim, **mlp_kw)
        self.out = nn.Sequential(nn.Linear(hid, action_dim), nn.Tanh())

    def forward(self, z: torch.Tensor, **_) -> dict:
        action = self.out(self.net(z))
        return {"action": action, "log_prob": None}

    def get_action(self, z: torch.Tensor, deterministic: bool = True):
        return self.forward(z)["action"], None, None


class GaussianActorHead(nn.Module):
    """Gaussian policy for PPO / A2C / A3C."""

    type_name = "gaussian"

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        layer_configs: list,
        state_dependent_std: bool = True,
        **kw,
    ):
        super().__init__()
        mlp_kw = {
            k: v
            for k, v in kw.items()
            if k
            in (
                "default_activation",
                "default_norm",
                "default_dropout",
                "norm_order",
                "weight_init",
            )
        }
        log_std_init = float(kw.get("log_std_init", 0.0))
        log_std_min = float(kw.get("log_std_min", -20.0))
        log_std_max = float(kw.get("log_std_max", 2.0))
        self.net, hid = build_mlp(layer_configs, input_dim, **mlp_kw)
        self.mean_head = nn.Linear(hid, action_dim)
        if state_dependent_std:
            self.log_std_head: nn.Module = nn.Linear(hid, action_dim)
        else:
            self.log_std_head = None
            self.log_std_param = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.state_dependent_std = state_dependent_std
        self.dist = DiagonalGaussian(
            action_dim,
            log_std_init=log_std_init,
            state_dependent_std=state_dependent_std,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    def forward(self, z: torch.Tensor, deterministic: bool = False, **_) -> dict:
        h = self.net(z)
        mean = self.mean_head(h)
        if self.state_dependent_std:
            log_std = self.log_std_head(h)
        else:
            log_std = self.log_std_param.expand_as(mean)
        dist = self.dist(mean, log_std)
        action = dist.mode() if deterministic else dist.rsample()
        return {"action": action, "log_prob": dist.log_prob(action), "dist": dist, "mean": mean}

    def get_action(self, z, deterministic=False):
        out = self.forward(z, deterministic)
        return out["action"], out["log_prob"], out.get("dist")

    def evaluate_actions(self, z: torch.Tensor, actions: torch.Tensor):
        h = self.net(z)
        mean = self.mean_head(h)
        if self.state_dependent_std:
            log_std = self.log_std_head(h)
        else:
            log_std = self.log_std_param.expand_as(mean)
        dist = self.dist(mean, log_std)
        return dist.log_prob(actions), dist.entropy()


class SquashedGaussianActorHead(nn.Module):
    """SAC actor: tanh-squashed Gaussian."""

    type_name = "squashed_gaussian"

    def __init__(self, input_dim: int, action_dim: int, layer_configs: list, **kw):
        super().__init__()
        mlp_kw = {
            k: v
            for k, v in kw.items()
            if k
            in (
                "default_activation",
                "default_norm",
                "default_dropout",
                "norm_order",
                "weight_init",
            )
        }
        log_std_min = float(kw.get("log_std_min", -20.0))
        log_std_max = float(kw.get("log_std_max", 2.0))
        self.net, hid = build_mlp(layer_configs, input_dim, **mlp_kw)
        self.mean_head = nn.Linear(hid, action_dim)
        self.log_std_head = nn.Linear(hid, action_dim)
        self.dist = SquashedGaussian(action_dim, log_std_min=log_std_min, log_std_max=log_std_max)

    def forward(self, z: torch.Tensor, deterministic: bool = False, **_) -> dict:
        h = self.net(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        dist = self.dist(mean, log_std)
        if deterministic:
            action = dist.mode()
            log_prob = dist.log_prob(action)
        else:
            action, log_prob = dist.rsample_and_log_prob()
        return {"action": action, "log_prob": log_prob, "dist": dist, "mean": mean}

    def get_action(self, z, deterministic=False):
        out = self.forward(z, deterministic)
        return out["action"], out["log_prob"], out.get("dist")

    def evaluate_actions(self, z: torch.Tensor, actions: torch.Tensor):
        h = self.net(z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        dist = self.dist(mean, log_std)
        return dist.log_prob(actions), dist.entropy()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_ACTOR_HEADS = {
    "deterministic": DeterministicActorHead,
    "gaussian": GaussianActorHead,
    "squashed_gaussian": SquashedGaussianActorHead,
}


def build_actor_head(
    head_type: str,
    input_dim: int,
    action_dim: int,
    layer_configs: list,
    **kwargs: Any,
) -> nn.Module:
    if head_type not in _ACTOR_HEADS:
        raise ValueError(f"Unknown actor head type '{head_type}'. Options: {sorted(_ACTOR_HEADS)}")
    return _ACTOR_HEADS[head_type](input_dim, action_dim, layer_configs, **kwargs)
