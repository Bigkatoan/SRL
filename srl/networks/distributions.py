"""Action distributions for continuous control.

DiagonalGaussian : standard Gaussian used by PPO / A2C / A3C
SquashedGaussian : tanh-squashed Gaussian used by SAC
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class DiagonalGaussian(nn.Module):
    """Diagonal Gaussian action distribution.

    Parameters
    ----------
    action_dim:
        Dimensionality of the action vector.
    log_std_init:
        Initial value of the log standard deviation parameter.
    state_dependent_std:
        If True, log_std is output by the network (passed in forward).
        If False, log_std is a learned global parameter.
    """

    def __init__(
        self,
        action_dim: int,
        log_std_init: float = 0.0,
        state_dependent_std: bool = True,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        if not state_dependent_std:
            self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor | None = None,
    ) -> DiagonalGaussian._Dist:
        if not self.state_dependent_std:
            log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return self._Dist(mean, std)

    class _Dist:
        def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
            self.mean = mean
            self.std = std
            self._dist = torch.distributions.Normal(mean, std)

        def sample(self) -> torch.Tensor:
            return self._dist.sample()

        def rsample(self) -> torch.Tensor:
            return self._dist.rsample()

        def log_prob(self, action: torch.Tensor) -> torch.Tensor:
            return self._dist.log_prob(action).sum(-1)

        def entropy(self) -> torch.Tensor:
            return self._dist.entropy().sum(-1)

        def mode(self) -> torch.Tensor:
            return self.mean


class SquashedGaussian(nn.Module):
    """Tanh-squashed Gaussian used by SAC.

    log_prob corrected for tanh change-of-variables:
        log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))

    Numerically stable version:
        - 2(log 2 - u - softplus(-2u))
    """

    def __init__(
        self, action_dim: int, log_std_min: float = LOG_STD_MIN, log_std_max: float = LOG_STD_MAX
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> SquashedGaussian._Dist:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return self._Dist(mean, std)

    class _Dist:
        def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
            self.mean = mean
            self.std = std
            self._normal = torch.distributions.Normal(mean, std)

        def rsample_and_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
            u = self._normal.rsample()
            action = torch.tanh(u)
            log_prob = self._normal.log_prob(u).sum(-1)
            # Tanh change-of-variables correction (numerically stable)
            log_prob -= (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
            return action, log_prob

        def log_prob(self, action: torch.Tensor) -> torch.Tensor:
            # Invert tanh: u = atanh(a), clipped for stability
            u = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
            log_prob = self._normal.log_prob(u).sum(-1)
            log_prob -= (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
            return log_prob

        def entropy(self) -> torch.Tensor:
            return self._normal.entropy().sum(-1)

        def mode(self) -> torch.Tensor:
            return torch.tanh(self.mean)

        def sample(self) -> torch.Tensor:
            with torch.no_grad():
                u = self._normal.sample()
                return torch.tanh(u)
