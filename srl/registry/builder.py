"""ModelBuilder — assembles an AgentModel from a YAML or dict config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from srl.config._validate import ConfigError
from srl.registry.config_schema import AgentModelConfig, EncoderConfig, HeadConfig
from srl.registry.flow_graph import FlowGraph
from srl.registry.registry import EncoderRegistry, HeadRegistry


def _validate_config(cfg: AgentModelConfig) -> None:
    """Fail fast with an actionable message instead of a cryptic torch error
    (e.g. "mat1 and mat2 shapes cannot be multiplied", or a bare TypeError
    from nn.Linear(None, ...)) deep inside the first forward pass.

    Type names and per-type requirements come from the registries, so
    built-in and plugin types are checked identically.
    """
    errors: list[str] = []
    seen_encoders: dict[str, EncoderConfig] = {}
    for i, enc_cfg in enumerate(cfg.encoders):
        path = f"encoders[{i}] ('{enc_cfg.name}')"
        prior = seen_encoders.get(enc_cfg.name)
        if prior is not None and (
            prior.type != enc_cfg.type
            or prior.latent_dim != enc_cfg.latent_dim
            or prior.input_dim != enc_cfg.input_dim
            or prior.input_shape != enc_cfg.input_shape
        ):
            errors.append(
                f"Encoder '{enc_cfg.name}' is declared more than once with different "
                "configs. ModelBuilder builds only the first declaration and silently "
                "ignores later ones with the same name -- this is almost certainly a "
                "copy-paste bug. Give the duplicate a distinct name if it's meant to "
                "be a second, separate encoder."
            )
        seen_encoders.setdefault(enc_cfg.name, enc_cfg)

        try:
            entry = EncoderRegistry.entry(enc_cfg.type)
        except (ConfigError, ImportError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        for attr in entry.requires:
            if getattr(enc_cfg, attr, None) is None:
                errors.append(
                    f"{path}: type '{enc_cfg.type}' requires '{attr}', which is 'auto' or "
                    "missing and was not resolved. Set it explicitly, or build with the env "
                    "spaces (ModelBuilder.from_dict(d, obs_space=..., act_space=...) / "
                    "srl.config.resolve_auto_dims)."
                )

    for role, head_cfg in (("actor", cfg.actor), ("critic", cfg.critic)):
        if head_cfg is None:
            continue
        try:
            entry = HeadRegistry.entry(head_cfg.type)
        except (ConfigError, ImportError) as exc:
            errors.append(f"{role}: {exc}")
            continue
        if entry.requires_action_dim and not head_cfg.action_dim:
            errors.append(
                f"{role}.action_dim is required for {role} type '{head_cfg.type}' but is "
                "'auto' or missing and was not resolved. Set it explicitly, or build with "
                "the env spaces (act_space=...)."
            )
        for attr in entry.requires:
            if getattr(head_cfg, attr, None) is None:
                errors.append(f"{role}: type '{head_cfg.type}' requires '{attr}'")
    if errors:
        raise ConfigError(errors)


def _build_encoder(cfg: EncoderConfig):
    """Instantiate a single encoder from its config via the encoder registry."""
    enc = EncoderRegistry.get(cfg.type)(cfg)

    # Optional: wrap with MomentumEncoder
    if cfg.use_momentum:
        from srl.networks.encoders.momentum_encoder import MomentumEncoder

        enc = MomentumEncoder(encoder=enc, tau=cfg.momentum_tau)

    # Optional: wrap with recurrent if requested separately (e.g. for CNN)
    if cfg.recurrent and not _is_recurrent_type(cfg.type):
        from srl.networks.encoders.recurrent import LSTMEncoder

        enc = LSTMEncoder(base_encoder=enc, hidden_size=cfg.lstm_hidden)

    return enc


def _is_recurrent_type(enc_type: str) -> bool:
    return bool(getattr(EncoderRegistry.get(enc_type), "srl_is_recurrent", False))


def _get_encoder_latent_dim(cfg: EncoderConfig) -> int:
    """Return the output latent dimension of an encoder config."""
    if cfg.recurrent or _is_recurrent_type(cfg.type):
        return cfg.lstm_hidden
    return cfg.latent_dim


def _build_head(cfg: HeadConfig, input_dim: int):
    """Instantiate an actor or critic head from its config via the head registry."""
    return HeadRegistry.get(cfg.type)(cfg, input_dim)


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
    def from_yaml(
        path: str | Path, *, obs_space: Any = None, act_space: Any = None
    ) -> Any:  # returns AgentModel
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        return ModelBuilder.from_dict(d, obs_space=obs_space, act_space=act_space)

    @staticmethod
    def from_dict(
        d: dict[str, Any], *, obs_space: Any = None, act_space: Any = None
    ) -> Any:  # returns AgentModel
        """Strictly parse ``d`` and build the model.

        ``obs_space`` / ``act_space`` (optional) resolve ``input_dim: auto``,
        ``input_shape: auto`` and ``action_dim: auto`` and cross-check explicit
        values -- see :func:`srl.config.resolve_auto_dims`.
        """
        return ModelBuilder.from_config(
            AgentModelConfig.from_dict(d), obs_space=obs_space, act_space=act_space
        )

    @staticmethod
    def from_config(
        cfg: AgentModelConfig, *, obs_space: Any = None, act_space: Any = None
    ) -> Any:  # returns AgentModel
        """Build from an already-parsed :class:`AgentModelConfig`
        (e.g. ``load_experiment(...).model``)."""
        if obs_space is not None or act_space is not None:
            from srl.config.auto_dims import resolve_auto_dims

            cfg = resolve_auto_dims(cfg, obs_space, act_space)
        _validate_config(cfg)
        return ModelBuilder._build(cfg)

    @staticmethod
    def _build(cfg: AgentModelConfig) -> Any:
        from srl.networks.agent_model import AgentModel

        # 1. Build encoders (shared instances for duplicate names)
        encoder_map: dict[str, Any] = {}
        latent_dims: dict[str, int] = {}
        encoder_input_names: dict[str, str | None] = {}
        for enc_cfg in cfg.encoders:
            encoder_input_names[enc_cfg.name] = enc_cfg.input_name
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
                        # Reconstruct the same (C, H, W) the encoder read in --
                        # ConvDecoderHead's job is to decode back to it.
                        aux_encoders[f"{enc_cfg.name}_aux"] = ConvDecoderHead(
                            latent_dim=enc_cfg.latent_dim,
                            output_shape=tuple(enc_cfg.input_shape),
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
            encoder_input_names=encoder_input_names,
        )
