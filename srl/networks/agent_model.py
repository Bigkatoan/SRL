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
        encoder_input_names: dict[str, str | None] | None = None,
    ) -> None:
        super().__init__()
        self.flow_graph = flow_graph

        self.encoders = nn.ModuleDict(encoders)
        self.encoder_input_names = dict(encoder_input_names or {})
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
        *,
        detach_encoders: bool = False,
        actor_action: torch.Tensor | None = None,
        compute_actor: bool = True,
        compute_critic: bool = True,
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
            Action tensor — passed to Q-function critic heads when present
            (DDPG/TD3/SAC-style Q(s, a)). Deliberately a separate parameter
            from `actor_action` below: a `ValueHead` (PPO/A2C's on-policy
            critic) only accepts `(z)`, so this is only ever forwarded to
            critic types that declared an `action_dim` (Q-function/TwinQ).
        actor_action:
            Action tensor to re-evaluate under the CURRENT actor distribution
            for on-policy algorithms (PPO/A2C-style importance-sampling
            ratio). When set and the actor head implements
            `evaluate_actions(z, action)`, `actor_out` is `{"action":
            actor_action, "log_prob": ..., "entropy": ...}` computed against
            *this* action instead of a freshly sampled one -- without this,
            `actor_out["log_prob"]` would be the log-prob of an unrelated new
            sample, which silently breaks the PPO ratio (`new_log_prob -
            old_log_prob` is meaningless unless both refer to the same
            action).
        compute_actor, compute_critic:
            Set either to False when a caller only wants the other head's
            output (e.g. an off-policy `update()` computing next_action from
            the actor alone, or a Q(s, a) critic pass that has no use for a
            fresh actor sample). Both default to True, which reproduces the
            exact behaviour of every existing caller -- this is purely
            opt-in. When one is False, encoders that feed *only* the skipped
            head are not run either (see `encoder_names_for_head`), which is
            where the real saving comes from for architectures with separate
            actor/critic encoders (e.g. an asymmetric actor-critic with a
            privileged critic observation). For a single shared encoder that
            implicitly feeds both heads (no explicit flow-graph routing),
            `encoder_names_for_head` conservatively reports every encoder as
            needed regardless of which head is requested, so nothing is
            skipped there -- only the unwanted head's own module call (and,
            for the critic with no `action` given, its dummy zero-action
            forward) is avoided. Raises if both are False, since that would
            make the call a no-op that still pays for every encoder.

        Returns
        -------
        dict with keys: latents, actor_out, value, new_hidden
        """
        if not compute_actor and not compute_critic:
            raise ValueError(
                "AgentModel.forward(): compute_actor and compute_critic cannot both be "
                "False -- there would be nothing to compute."
            )

        hidden_states = hidden_states or {}
        latents: dict[str, torch.Tensor] = {}
        new_hidden: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Auto-remap: if obs_dict has different keys than encoder names, try to map them
        # E.g., if obs has {'state'} and encoder expects {'state_enc'}, auto-map for simplicity
        _obs_dict = self._remap_obs_dict(obs_dict)

        # When only one head is requested, skip encoders that exclusively
        # feed the other one. `None` means "compute every encoder" (the
        # default, both-heads-requested case) so the loop below is an exact
        # no-op change from before.
        needed_encoders: set[str] | None = None
        if not (compute_actor and compute_critic):
            needed_encoders = set()
            if compute_actor and self.actor is not None:
                actor_name = _get_module_name(self.actor, "actor")
                needed_encoders.update(self.encoder_names_for_head(actor_name))
            if compute_critic and self.critic is not None:
                critic_name = _get_module_name(self.critic, "critic")
                needed_encoders.update(self.encoder_names_for_head(critic_name))

        for node_name in self.flow_graph.execution_order:
            inputs = self.flow_graph.get_inputs(node_name)

            if node_name in self.encoders:
                if needed_encoders is not None and node_name not in needed_encoders:
                    continue
                enc = self.encoders[node_name]
                obs = _obs_dict.get(node_name)
                if obs is None:
                    # Upstream encoder — use concatenated upstream latents
                    if inputs:
                        latents[node_name] = _concat_latents(inputs, latents)
                    continue

                hs = hidden_states.get(node_name)
                out = _run_encoder(enc, obs, hs)
                # out may be tensor or (latent, hidden)
                if isinstance(out, tuple):
                    latent, hs_new = out
                    if detach_encoders:
                        latent = latent.detach()
                    latents[node_name] = latent
                    new_hidden[node_name] = hs_new
                else:
                    if detach_encoders:
                        out = out.detach()
                    latents[node_name] = out

            elif self.actor is not None and node_name == getattr(self.actor, "name", "actor"):
                # Handled below
                pass
            elif self.critic is not None and node_name == getattr(self.critic, "name", "critic"):
                pass

        # Compute actor head input
        actor_out = None
        if self.actor is not None and compute_actor:
            actor_name = _get_module_name(self.actor, "actor")
            actor_inputs = self.flow_graph.get_inputs(actor_name)
            if actor_inputs:
                actor_latent = _concat_latents(actor_inputs, latents)
            elif latents:
                actor_latent = torch.cat(list(latents.values()), dim=-1)
            else:
                raise RuntimeError("No latents available for actor head.")
            if actor_action is not None and hasattr(self.actor, "evaluate_actions"):
                eval_log_prob, eval_entropy = self.actor.evaluate_actions(
                    actor_latent, actor_action
                )
                actor_out = {
                    "action": actor_action,
                    "log_prob": eval_log_prob,
                    "entropy": eval_entropy,
                }
            else:
                actor_out = self.actor(actor_latent)

        # Compute critic head input
        value_out = None
        if self.critic is not None and compute_critic:
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
        *,
        detach_encoders: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Run only the encoder passes, return (latents, new_hidden)."""
        hidden_states = hidden_states or {}
        latents: dict[str, torch.Tensor] = {}
        new_hidden: dict[str, Any] = {}

        remapped_obs = self._remap_obs_dict(obs_dict)
        for name, enc in self.encoders.items():
            obs = remapped_obs.get(name)
            if obs is None:
                continue
            hs = hidden_states.get(name)
            out = _run_encoder(enc, obs, hs)
            if isinstance(out, tuple):
                latent, hs_new = out
                if detach_encoders:
                    latent = latent.detach()
                latents[name] = latent
                new_hidden[name] = hs_new
            else:
                if detach_encoders:
                    out = out.detach()
                latents[name] = out
        return latents, new_hidden

    def encoder_names_for_head(self, head_name: str) -> list[str]:
        """Return encoder names that feed a head, following flow edges recursively."""
        if not self.encoders:
            return []

        actor_name = _get_module_name(self.actor, "actor") if self.actor is not None else None
        critic_name = _get_module_name(self.critic, "critic") if self.critic is not None else None
        if head_name not in {actor_name, critic_name}:
            return []

        inputs = self.flow_graph.get_inputs(head_name)
        if not inputs:
            return list(self.encoders.keys())

        encoder_names: list[str] = []
        seen: set[str] = set()

        def visit(node_name: str) -> None:
            if node_name in seen:
                return
            seen.add(node_name)
            if node_name in self.encoders:
                encoder_names.append(node_name)
            for upstream in self.flow_graph.get_inputs(node_name):
                visit(upstream)

        for input_name in inputs:
            visit(input_name)
        return encoder_names

    def _remap_obs_dict(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map observation dict keys to encoder names.

        Multi-modal (image + vector) routing rules
        ------------------------------------------
        The obs_dict passed to forward() must have keys that match encoder
        names for each encoder to receive its correct input.

        Rule 0 — EXPLICIT NAME: encoder has input_name set → route by that obs key.
             e.g. obs={"joint_states": vec}, encoder joint_enc input_name="joint_states"
                  → {"joint_enc": vec}  ✓

        Rule 1 — EXACT MATCH: any key already equals an encoder name → passthrough.
                 e.g. obs={"cnn_enc": img, "mlp_enc": vec},  encoders: cnn_enc, mlp_enc  ✓

        Rule 2 — SINGLE → SINGLE: one obs key, one encoder → rename obs key.
                 e.g. obs={"policy": img},  encoder: policy_enc  →  {"policy_enc": img}  ✓

        Rule 3 — N → N (same count): zip obs values to encoder names by order.
                 e.g. obs={"policy": img, "priv": vec},  encoders: cnn_enc, mlp_enc
                      →  {"cnn_enc": img, "mlp_enc": vec}  ✓  (order must match)

        Rule 4 — PASSTHROUGH: anything else (partial matches, count mismatch).

        Validation
        ----------
        - If an encoder declares input_name and that obs key is missing → KeyError.
        - If explicit routing leaves obs keys unused → warnings.warn.

        Idempotency
        -----------
        Callers that already know the encoder layout (e.g. srl/cli/train.py's
        rollout loops, which remap once before calling agent.predict()/model())
        may pass in an obs_dict whose keys already match every encoder name.
        Remapping it a second time here would be wrong, not just redundant: Rule
        0's explicit input_name lookup searches for the ORIGINAL raw obs key,
        which no longer exists once the first pass has already renamed it away
        -- every explicit-input_name config would KeyError on its second forward
        pass. Short-circuit when nothing is left to do.
        """
        from srl.utils.obs_remap import apply_obs_remap

        encoder_names = list(self.encoders.keys())
        if set(obs_dict.keys()) == set(encoder_names):
            return obs_dict

        return apply_obs_remap(
            obs_dict,
            encoder_names,
            self.encoder_input_names,
        )

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
