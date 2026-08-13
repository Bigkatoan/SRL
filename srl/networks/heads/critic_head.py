"""Critic / Value heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from srl.networks.layers.mlp_builder import build_mlp


class ValueHead(nn.Module):
    """V(s) — on-policy value function (PPO, A2C)."""

    type_name = "value"

    def __init__(self, input_dim: int, layer_configs: list, **kw):
        super().__init__()
        self.net, hid = build_mlp(layer_configs, input_dim, **kw)
        self.out = nn.Linear(hid, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(z)).squeeze(-1)


class QFunctionHead(nn.Module):
    """Q(s, a) — off-policy critic (DDPG)."""

    type_name = "q_function"

    def __init__(self, input_dim: int, action_dim: int, layer_configs: list, **kw):
        super().__init__()
        self._action_dim = action_dim
        self.net, hid = build_mlp(layer_configs, input_dim + action_dim, **kw)
        self.out = nn.Linear(hid, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if action is None:
            action = z.new_zeros(z.shape[0], self._action_dim)
        sa = torch.cat([z, action], dim=-1)
        return self.out(self.net(sa)).squeeze(-1)


class TwinQHead(nn.Module):
    """Two independent Q-networks for SAC (reduces overestimation bias)."""

    type_name = "twin_q"

    def __init__(self, input_dim: int, action_dim: int, layer_configs: list, **kw):
        super().__init__()
        self.q1 = QFunctionHead(input_dim, action_dim, layer_configs, **kw)
        self.q2 = QFunctionHead(input_dim, action_dim, layer_configs, **kw)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(z, action), self.q2(z, action)

    def q_min(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(z, action)
        return torch.min(q1, q2)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_CRITIC_HEADS = {
    "value": ValueHead,
    "q_function": QFunctionHead,
    "q": QFunctionHead,  # alias -- srl/registry/builder.py, srl/cli/train.py's
    # DDPG-compatible-heads check, and config_schema.py's type comment all
    # already treat "q" as a valid critic head type; it just wasn't actually
    # registered here, so `critic: {type: q}` passed every validation check
    # and then crashed at build time with "Unknown critic head type 'q'".
    "twin_q": TwinQHead,
}


def build_critic_head(
    head_type: str,
    input_dim: int,
    layer_configs: list,
    action_dim: int = 0,
    **kwargs: Any,
) -> nn.Module:
    if head_type not in _CRITIC_HEADS:
        raise ValueError(
            f"Unknown critic head type '{head_type}'. Options: {sorted(_CRITIC_HEADS)}"
        )
    cls = _CRITIC_HEADS[head_type]
    if head_type == "value":
        return cls(input_dim, layer_configs, **kwargs)
    return cls(input_dim, action_dim, layer_configs, **kwargs)
