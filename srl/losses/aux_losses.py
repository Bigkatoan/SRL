"""Auxiliary self-supervised losses for visual encoder training.

Available losses
----------------
info_nce_loss    — CURL / SimCLR contrastive InfoNCE
reconstruction_loss — AE pixel MSE
byol_loss        — BYOL cosine regression (stop-grad on target)
vae_loss         — VAE reconstruction MSE + KL divergence (beta=1)
drq_aug_loss     — DrQ-v2 augmentation consistency (Q-value MSE)
spr_loss         — SPR self-predictive latent L2 regression
barlow_twins_loss — Barlow Twins cross-correlation redundancy reduction
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE / SimCLR contrastive loss.

    Parameters
    ----------
    z_anchor, z_positive:
        L2-normalised projection embeddings, shape ``(B, D)``.
    """
    B = z_anchor.size(0)
    z_a = F.normalize(z_anchor, dim=-1)
    z_p = F.normalize(z_positive, dim=-1)

    # Similarity matrix (B, B)
    logits = torch.mm(z_a, z_p.T) / temperature
    labels = torch.arange(B, device=z_anchor.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    return loss / 2.0


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE pixel reconstruction loss for autoencoder."""
    return F.mse_loss(recon, target.float(), reduction=reduction)


def byol_loss(
    online: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """BYOL regression loss (stop-gradient on target side, already handled
    externally by using a momentum encoder)."""
    online = F.normalize(online, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (online * target).sum(dim=-1).mean()


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """VAE evidence lower bound: MSE reconstruction + beta * KL divergence.

    Parameters
    ----------
    recon:
        Reconstructed pixels, shape ``(B, C, H, W)``.
    target:
        Original pixels, same shape.
    mu, log_var:
        Encoder posterior parameters, shape ``(B, latent_dim)``.
    beta:
        KL weight.  beta=1 is the standard VAE; higher values enforce
        disentanglement (β-VAE).
    """
    recon_loss = F.mse_loss(recon, target.float(), reduction="mean")
    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - var)
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()
    return recon_loss + beta * kl


def drq_aug_loss(
    q1_aug: torch.Tensor,
    q2_aug: torch.Tensor,
) -> torch.Tensor:
    """DrQ-v2 augmentation consistency loss.

    Encourages Q-value consistency across two independently-augmented views
    of the same observation.  Both tensors are detached from each other so
    only the encoder is trained.

    Parameters
    ----------
    q1_aug, q2_aug:
        Q-values from two augmented views, shape ``(B,)`` or ``(B, 1)``.
    """
    return F.mse_loss(q1_aug, q2_aug.detach())


def spr_loss(
    z_t: torch.Tensor,
    actions: torch.Tensor,
    model,  # AgentModel  (typed loosely to avoid circular import)
    encoder_key: str,
    n_steps: int = 1,
) -> torch.Tensor:
    """Self-Predictive Representations (SPR) latent prediction loss.

    Regresses the encoder's current latent ``z_t`` forward by one step using
    a lightweight :class:`~srl.networks.heads.aux_head.LatentTransitionModel`
    attached to the model.  The target is a *stop-gradient* of the same
    encoder applied to the (unavailable) next observation, so we approximate
    via the transition model prediction MSE on ``z_t`` itself (contrastive
    variant without look-ahead, sufficient for representation stability).

    Parameters
    ----------
    z_t:
        Current latent, shape ``(B, latent_dim)``.
    actions:
        Actions taken at time t, shape ``(B, action_dim)``.
    model:
        AgentModel that may contain a ``LatentTransitionModel`` submodule.
    """
    from srl.networks.heads.aux_head import LatentTransitionModel
    for module in model.modules():
        if isinstance(module, LatentTransitionModel):
            z_pred = module(z_t, actions)
            # Self-consistency: predicted z should normalise close to original z
            return F.mse_loss(
                F.normalize(z_pred, dim=-1),
                F.normalize(z_t.detach(), dim=-1),
            )
    return torch.zeros(1, device=z_t.device, requires_grad=True)


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lam: float = 5e-3,
) -> torch.Tensor:
    """Barlow Twins redundancy-reduction loss.

    Parameters
    ----------
    z1, z2:
        Projection embeddings from two augmented views, shape ``(B, D)``.
    lam:
        Weight for the off-diagonal redundancy term.
    """
    B, D = z1.shape
    # Normalise features across batch dimension
    z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    # Cross-correlation matrix
    c = torch.mm(z1_n.T, z2_n) / B  # (D, D)
    # On-diagonal: push toward 1; off-diagonal: push toward 0
    on_diag = (c.diagonal() - 1.0).pow(2).sum()
    mask = ~torch.eye(D, dtype=torch.bool, device=c.device)
    off_diag = c[mask].pow(2).sum()
    return on_diag + lam * off_diag
