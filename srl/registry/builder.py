"""ModelBuilder — assembles an AgentModel from a YAML or dict config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from srl.registry.config_schema import AgentModelConfig, EncoderConfig, HeadConfig
from srl.registry.flow_graph import FlowGraph
from srl.registry.registry import EncoderRegistry, HeadRegistry


def _build_encoder(cfg: EncoderConfig):
    """Instantiate a single encoder from its config."""
    enc_type = cfg.type.lower()

    if enc_type == "mlp":
        from srl.networks.encoders.mlp_encoder import MLPEncoder
        enc = MLPEncoder(
            input_dim=cfg.input_dim,  # type: ignore[arg-type]
            layer_configs=cfg.layers,
            latent_dim=cfg.latent_dim,
        )
    elif enc_type == "cnn":
        from srl.networks.encoders.cnn_encoder import CNNEncoder
        enc = CNNEncoder(
            input_shape=tuple(cfg.input_shape),  # type: ignore[arg-type]
            layer_configs=cfg.layers,
            latent_dim=cfg.latent_dim,
        )
    elif enc_type == "lstm":
        from srl.networks.encoders.mlp_encoder import MLPEncoder
        from srl.networks.encoders.recurrent import LSTMEncoder
        base = MLPEncoder(
            input_dim=cfg.input_dim,  # type: ignore[arg-type]
            layer_configs=cfg.layers,
            latent_dim=cfg.latent_dim,
        )
        enc = LSTMEncoder(base_encoder=base, hidden_size=cfg.lstm_hidden)
    elif enc_type == "text":
        from srl.networks.encoders.text_encoder import CharCNNTextEncoder
        enc = CharCNNTextEncoder(latent_dim=cfg.latent_dim)
    else:
        # Try registry
        enc_cls = EncoderRegistry.get(enc_type)
        enc = enc_cls(cfg)

    # Optional: wrap with MomentumEncoder
    if cfg.use_momentum:
        from srl.networks.encoders.momentum_encoder import MomentumEncoder
        enc = MomentumEncoder(encoder=enc, tau=cfg.momentum_tau)

    # Optional: wrap with recurrent if requested separately (e.g. for CNN)
    if cfg.recurrent and enc_type not in ("lstm",):
        from srl.networks.encoders.recurrent import LSTMEncoder
        enc = LSTMEncoder(base_encoder=enc, hidden_size=cfg.lstm_hidden)

    return enc


def _get_encoder_latent_dim(cfg: EncoderConfig) -> int:
    """Return the output latent dimension of an encoder config."""
    if cfg.recurrent or cfg.type.lower() == "lstm":
        return cfg.lstm_hidden
    return cfg.latent_dim


def _build_head(cfg: HeadConfig, input_dim: int):
    """Instantiate an actor or critic head from its config."""
    head_type = cfg.type.lower()

    if head_type in ("gaussian", "squashed_gaussian", "deterministic"):
        from srl.networks.heads.actor_head import build_actor_head
        return build_actor_head(
            head_type=head_type,
            input_dim=input_dim,
            action_dim=cfg.action_dim,  # type: ignore[arg-type]
            layer_configs=cfg.layers,
            log_std_init=cfg.log_std_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )
    elif head_type in ("value", "twin_q", "q", "q_function"):
        from srl.networks.heads.critic_head import build_critic_head
        return build_critic_head(
            head_type=head_type,
            input_dim=input_dim,
            layer_configs=cfg.layers,
            action_dim=cfg.action_dim,
        )
    else:
        head_cls = HeadRegistry.get(head_type)
        return head_cls(cfg, input_dim)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ModelBuilder:
    """Build an :class:`~srl.networks.agent_model.AgentModel` from config.

    Usage::

        model = ModelBuilder.from_yaml("configs/visual_ppo.yaml")
        # or
        model = ModelBuilder.from_dict(cfg_dict)
    """

    @staticmethod
    def from_yaml(path: str | Path) -> "Any":  # returns AgentModel
        with open(path, "r") as fh:
            d = yaml.safe_load(fh)
        return ModelBuilder.from_dict(d)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Any":  # returns AgentModel
        from srl.networks.agent_model import AgentModel

        cfg = AgentModelConfig.from_dict(d)
        return ModelBuilder._build(cfg)

    @staticmethod
    def _build(cfg: AgentModelConfig) -> "Any":
        from srl.networks.agent_model import AgentModel

        # 1. Build encoders (shared instances for duplicate names)
        encoder_map: dict[str, Any] = {}
        latent_dims: dict[str, int] = {}
        for enc_cfg in cfg.encoders:
            if enc_cfg.name not in encoder_map:
                encoder_map[enc_cfg.name] = _build_encoder(enc_cfg)
                latent_dims[enc_cfg.name] = _get_encoder_latent_dim(enc_cfg)

        # 2. Build flow graph
        node_names: list[str] = list(encoder_map.keys())
        if cfg.actor:
            node_names.append(cfg.actor.name)
        if cfg.critic:
            node_names.append(cfg.critic.name)

        flow_graph = FlowGraph(flow_specs=cfg.flows, node_names=node_names)

        # 3. Compute input dims for actor/critic heads
        if cfg.actor:
            actor_input_dim = flow_graph.resolve_input_dim(cfg.actor.name, latent_dims)
            # Fallback: use latent_dim of first encoder (no flow defined)
            if actor_input_dim == 0 and encoder_map:
                first_enc = list(latent_dims.keys())[0]
                actor_input_dim = latent_dims[first_enc]
            actor = _build_head(cfg.actor, actor_input_dim)
        else:
            actor = None

        if cfg.critic:
            critic_input_dim = flow_graph.resolve_input_dim(cfg.critic.name, latent_dims)
            if critic_input_dim == 0 and encoder_map:
                first_enc = list(latent_dims.keys())[0]
                critic_input_dim = latent_dims[first_enc]
            critic = _build_head(cfg.critic, critic_input_dim)
        else:
            critic = None

        # 4. Aux heads
        aux_encoders: dict[str, Any] = {}
        for enc_cfg in cfg.encoders:
            if enc_cfg.aux_type is not None:
                if enc_cfg.aux_type == "autoencoder":
                    from srl.networks.heads.aux_head import ConvDecoderHead
                    if enc_cfg.input_shape is not None:
                        in_channels = enc_cfg.input_shape[0]
                        aux_encoders[f"{enc_cfg.name}_aux"] = ConvDecoderHead(
                            latent_dim=enc_cfg.latent_dim,
                            out_channels=in_channels,
                        )
                elif enc_cfg.aux_type in ("contrastive", "byol"):
                    from srl.networks.heads.aux_head import ProjectionHead
                    aux_encoders[f"{enc_cfg.name}_aux"] = ProjectionHead(
                        input_dim=enc_cfg.latent_dim,
                        proj_dim=enc_cfg.aux_latent_dim,
                    )

        # 5. Build and return AgentModel
        return AgentModel(
            encoders=encoder_map,
            flow_graph=flow_graph,
            actor=actor,
            critic=critic,
            aux_modules=aux_encoders if aux_encoders else None,
        )
