"""Convolutional autoencoder representation module."""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.losses.aux_losses import reconstruction_loss


class AutoencoderModule(nn.Module):
    """Combine a CNN encoder and a ConvDecoder for pixel reconstruction.

    Parameters
    ----------
    encoder:
        CNN encoder (returns latent).
    decoder:
        :class:`~srl.networks.heads.aux_head.ConvDecoderHead`.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def compute_loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass + reconstruction loss.

        Returns
        -------
        (latent, ae_loss)
        """
        obs_f = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs
        latent = self.encoder(obs_f)
        recon = self.decoder(latent)
        loss = reconstruction_loss(recon, obs_f)
        return latent, loss
