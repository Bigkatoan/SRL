"""AgentModel — DAG-based nn.Module that wires encoders and heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from srl.registry.flow_graph import FlowGraph


class AgentModel(nn.Module):
    """A dynamic neural network built from a :class:`~srl.registry.flow_graph.FlowGraph`.

    Parameters
    ----------
    encoders:
        Mapping from node name → encoder module. Encoder modules should
        accept ``(obs, hidden_state)`` or ``(obs,)`` and return a latent
        tensor ``(B, latent_dim)`` (and optionally a new hidden state).
    flow_graph:
        The :class:`~srl.registry.flow_graph.FlowGraph` describing data flow.
    actor:
        Actor head module (receives concatenated upstream latents).
    critic:
        Critic / value head module.
    aux_modules:
        Optional auxiliary heads (autoencoder decoder, projection heads, …).
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        flow_graph: FlowGraph,
        actor: nn.Module | None = None,
        critic: nn.Module | None = None,
        aux_modules: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.flow_graph = flow_graph

        self.encoders = nn.ModuleDict(encoders)
        self.actor = actor
        self.critic = critic
        self.aux_modules = nn.ModuleDict(aux_modules or {})

        # Register actor/critic as sub-modules
        if actor is not None:
            self.add_module("actor", actor)
        if critic is not None:
            self.add_module("critic", critic)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run the full forward pass.

        Parameters
        ----------
        obs_dict:
            Dict mapping encoder name → raw observation tensor.
            If a key is missing the encoder is skipped (useful for partial
            inference during deployment).
        hidden_states:
            Optional LSTM hidden states per encoder name.
        action:
            Action tensor — passed to Q-function heads when present.

        Returns
        -------
        dict with keys: latents, actor_out, value, new_hidden
        """
        hidden_states = hidden_states or {}
        latents: dict[str, torch.Tensor] = {}
        new_hidden: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Auto-remap: if obs_dict has different keys than encoder names, try to map them
        # E.g., if obs has {'state'} and encoder expects {'state_enc'}, auto-map for simplicity
        _obs_dict = obs_dict
        if len(obs_dict) == 1 and len(self.encoders) == 1:
            obs_key = list(obs_dict.keys())[0]
            enc_name = list(self.encoders.keys())[0]
            if obs_key != enc_name:
                _obs_dict = {enc_name: obs_dict[obs_key]}

        for node_name in self.flow_graph.execution_order:
            inputs = self.flow_graph.get_inputs(node_name)

            if node_name in self.encoders:
                enc = self.encoders[node_name]
                obs = _obs_dict.get(node_name)
                if obs is None:
                    # Upstream encoder — use concatenated upstream latents
                    if inputs:
                        latents[node_name] = _concat_latents(inputs, latents)
                    continue

                hs = hidden_states.get(node_name)
                out = _run_encoder(enc, obs, hs)
                if isinstance(out, tuple):
                    latents[node_name], new_hidden[node_name] = out
                else:
                    latents[node_name] = out

            elif self.actor is not None and node_name == getattr(self.actor, "name", "actor"):
                # Handled below
                pass
            elif self.critic is not None and node_name == getattr(self.critic, "name", "critic"):
                pass

        # Compute actor head input
        actor_out = None
        if self.actor is not None:
            actor_name = _get_module_name(self.actor, "actor")
            actor_inputs = self.flow_graph.get_inputs(actor_name)
            if actor_inputs:
                actor_latent = _concat_latents(actor_inputs, latents)
            elif latents:
                actor_latent = torch.cat(list(latents.values()), dim=-1)
            else:
                raise RuntimeError("No latents available for actor head.")
            actor_out = self.actor(actor_latent)

        # Compute critic head input
        value_out = None
        if self.critic is not None:
            critic_name = _get_module_name(self.critic, "critic")
            critic_inputs = self.flow_graph.get_inputs(critic_name)
            if critic_inputs:
                critic_latent = _concat_latents(critic_inputs, latents)
            elif latents:
                critic_latent = torch.cat(list(latents.values()), dim=-1)
            else:
                raise RuntimeError("No latents available for critic head.")

            if action is not None:
                value_out = self.critic(critic_latent, action)
            else:
                value_out = self.critic(critic_latent)

        return {
            "latents": latents,
            "actor_out": actor_out,
            "value": value_out,
            "new_hidden": new_hidden,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def encode(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_states: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Run only the encoder passes, return (latents, new_hidden)."""
        hidden_states = hidden_states or {}
        latents: dict[str, torch.Tensor] = {}
        new_hidden: dict[str, Any] = {}

        for name, enc in self.encoders.items():
            obs = obs_dict.get(name)
            if obs is None:
                continue
            hs = hidden_states.get(name)
            out = _run_encoder(enc, obs, hs)
            if isinstance(out, tuple):
                latents[name], new_hidden[name] = out
            else:
                latents[name] = out
        return latents, new_hidden

    def act(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_states: dict[str, Any] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Encode → actor → return (action, new_hidden)."""
        result = self.forward(obs_dict, hidden_states=hidden_states)
        actor_out = result["actor_out"]
        if actor_out is None:
            raise RuntimeError("No actor head configured.")

        if deterministic:
            if isinstance(actor_out, torch.Tensor):
                action = actor_out
            elif hasattr(actor_out, "mean"):
                action = actor_out.mean
            else:
                action, _ = actor_out
        else:
            if isinstance(actor_out, torch.Tensor):
                action = actor_out
            elif hasattr(actor_out, "rsample"):
                action = actor_out.rsample()
            else:
                action, _ = actor_out

        return action, result["new_hidden"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_encoder(encoder: nn.Module, obs: torch.Tensor, hidden=None):
    """Call encoder with optional hidden state."""
    try:
        if hidden is not None:
            return encoder(obs, hidden)
        return encoder(obs)
    except TypeError:
        return encoder(obs)


def _concat_latents(names: list[str], latents: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = [latents[n] for n in names]
    return torch.cat(tensors, dim=-1)


def _get_module_name(module: nn.Module, fallback: str) -> str:
    return getattr(module, "name", fallback)
