"""LSTMEncoder — wraps any base encoder with an LSTM for partial observability."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Recurrent encoder: base_encoder → LSTM(hidden_size) → latent.

    Parameters
    ----------
    base_encoder:
        Any encoder with a ``latent_dim`` property.
    hidden_size:
        LSTM hidden state size.
    num_layers:
        Number of LSTM layers.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        hidden_size: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.base = base_encoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=base_encoder.latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self._latent_dim = hidden_size

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        obs:
            ``[B, T, *obs_shape]`` for sequences, or ``[B, *obs_shape]`` for single step.
        hidden:
            LSTM state ``(h, c)``.  Pass ``None`` to initialise to zeros.

        Returns
        -------
        latent : Tensor   [B, T, hidden_size] or [B, hidden_size]
        (h, c) : tuple    New LSTM state.
        """
        single_step = (
            obs.ndim == self.base.input_ndim
            if hasattr(self.base, "input_ndim")
            else (obs.ndim <= 3)
        )
        if single_step:
            obs = obs.unsqueeze(1)  # add time dim

        B, T = obs.shape[:2]
        flat = self.base(obs.view(B * T, *obs.shape[2:]))  # (B*T, latent)
        seq = flat.view(B, T, -1)
        out, (h, c) = self.lstm(seq, hidden)

        if single_step:
            return out[:, 0, :], (h, c)
        return out, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> tuple:
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()
