"""TextEncoder — tiny character-level CNN.  No pretrained weights."""

from __future__ import annotations

import torch
import torch.nn as nn


class CharCNNTextEncoder(nn.Module):
    """Encode short text goals via a character-level CNN.

    Input: integer token ids, shape ``[B, seq_len]`` where each token is a
    character index in [0, vocab_size).

    Architecture:
        Embedding(vocab_size, embed_dim)
        → Transpose to [B, embed_dim, seq_len]
        → 3 × Conv1d + ReLU
        → AdaptiveMaxPool1d(1)
        → Linear(channels, latent_dim)

    The encoder is compact (~50K params) and trained from scratch alongside RL.
    """

    def __init__(
        self,
        vocab_size: int = 128,  # ASCII printable chars
        embed_dim: int = 32,
        latent_dim: int = 64,
        channels: int = 128,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(channels, latent_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @staticmethod
    def tokenize(text: str, max_len: int = 64) -> list[int]:
        """Convert string to ASCII char codes, padded/truncated to max_len."""
        ids = [min(ord(c), 127) for c in text[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, seq_len] int64."""
        x = self.embed(token_ids).transpose(1, 2)  # [B, embed, L]
        x = self.pool(self.conv(x)).squeeze(-1)  # [B, channels]
        return self.proj(x)  # [B, latent_dim]
