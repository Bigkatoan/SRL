"""Config schema dataclasses for YAML-driven model building.

Parsing is strict: an unknown key anywhere in the model section (a typo such
as ``latnet_dim``) raises :class:`~srl.config.ConfigError` with a "did you
mean" suggestion instead of being silently dropped. Type-specific keys of a
registered encoder/head type (e.g. ``vocab_size`` for ``type: text``) are
accepted when the type declares them (``extra_keys``) and collected into
``cfg.extra``; arbitrary keyword arguments for a custom type can always be
passed explicitly under ``extra:``.

``input_dim`` / ``input_shape`` (encoders) and ``action_dim`` (heads) may be
written as ``auto`` or omitted; they are then ``None`` here and filled from the
environment's spaces by :func:`srl.config.resolve_auto_dims`.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from srl.config._validate import (
    AT_LEAST_ONE,
    ConfigDeprecationWarning,
    ConfigError,
    from_mapping,
    suggest,
    validate_constraints,
    yaml_fields,
)

AUX_TYPES = ("autoencoder", "contrastive", "byol")
AuxType = Literal["autoencoder", "contrastive", "byol"]

# Keys that live at the top level of an experiment YAML but are not part of
# the model section; AgentModelConfig.from_dict() accepts and ignores them.
EXPERIMENT_KEYS = frozenset(
    {"schema_version", "algo", "env_id", "env_type", "include_goal", "train"}
)
# Accepted with a deprecation warning, then ignored.
DEPRECATED_MODEL_KEYS = {
    "losses": (
        "the top-level 'losses:' block was never read -- each algorithm builds "
        "its own loss terms from its train: hyperparameters (e.g. vf_coef, "
        "entropy_coef, aux_weight). Remove it."
    ),
}


class _Validated:
    def __post_init__(self) -> None:
        validate_constraints(self)


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


# ---------------------------------------------------------------------------
# Layer-level config (shared by MLP / CNN layer lists)
# ---------------------------------------------------------------------------


@dataclass
class LayerConfig:
    """A single layer description inside an encoder or head.

    Documentation of the dict layer form; the networks parse ``layers``
    entries themselves (ints, ``[out, k, s]`` lists or dicts with these keys).
    """

    out_features: int | None = None  # MLP
    out_channels: int | None = None  # CNN
    kernel: int = 3  # the key cnn_builder reads (was misnamed kernel_size here)
    stride: int = 1
    padding: int = 0
    activation: str = "relu"
    norm: str = "none"
    dropout: float = 0.0
    dropout_type: str = "standard"
    pooling: str = "none"
    pooling_kernel: int = 2
    residual: bool = False
    depthwise: bool = False
    norm_order: str = "post"  # "pre" | "post"
    weight_init: str = "none"


# ---------------------------------------------------------------------------
# Encoder / head configs
# ---------------------------------------------------------------------------


def _split_extra(
    d: dict[str, Any], cls: type, entry: Any, path: str, errors: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a raw mapping into (dataclass kwargs, type-specific extras)."""
    known = set(yaml_fields(cls)) - {"extra"}
    allowed_extra = set(getattr(entry, "extra_keys", ()) or ())
    allow_any = bool(getattr(entry, "allow_extra", False))
    explicit = d.pop("extra", None)
    extra: dict[str, Any] = {}
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            errors.append(f"{path}.extra: expected a mapping, got {type(explicit).__name__}")
        else:
            extra.update(explicit)
    fields_d: dict[str, Any] = {}
    for key, value in d.items():
        if key in known:
            fields_d[key] = value
        elif key in allowed_extra or allow_any:
            extra[key] = value
        else:
            hint = suggest(str(key), known | allowed_extra)
            type_name = d.get("type")
            note = (
                ""
                if hint
                else (
                    f"; valid keys: {', '.join(sorted(known | allowed_extra))}. Keyword "
                    f"arguments specific to a custom type go under 'extra:'"
                )
            )
            errors.append(
                f"{path}.{key}: unknown key for {cls.__name__} (type={type_name!r}){hint}{note}"
            )
    return fields_d, extra


def _lookup_type(registry: Any, type_value: Any, path: str, errors: list[str]) -> Any:
    if not isinstance(type_value, str):
        if type_value is None:
            errors.append(f"{path}.type: required key is missing")
        else:
            errors.append(f"{path}.type: expected a string, got {type_value!r}")
        return None
    if type_value not in registry:
        available = registry.available()
        errors.append(
            f"{path}.type: unknown {registry.kind} type '{type_value}'"
            f"{suggest(type_value.lower(), available)}. Available: {available}"
        )
        return None
    try:
        return registry.entry(type_value)
    except Exception as exc:  # plugin import failure
        errors.append(f"{path}.type: {exc}")
        return None


@dataclass
class EncoderConfig(_Validated):
    name: str
    type: str  # any registered encoder type ("mlp" | "cnn" | "lstm" | "text" | plugins)
    # Explicit obs key this encoder reads from.  When set, routing is by name
    # and takes priority over all heuristic rules.  Leave None for auto-routing.
    input_name: str | None = None
    # mlp/lstm: flat observation size. None = "auto" (resolved from the env).
    input_dim: int | None = None
    # cnn: [C, H, W]. None = "auto" (resolved from the env).
    input_shape: list[int] | None = None
    # shared
    latent_dim: int = 128
    layers: list[Any] = field(default_factory=list)
    # Auxiliary representation head built next to this encoder
    aux_type: AuxType | None = None
    aux_latent_dim: int = 64
    # Momentum / EMA encoder
    use_momentum: bool = False
    momentum_tau: float = 0.99
    # Recurrent wrapper
    recurrent: bool = False
    lstm_hidden: int = 256
    # Type-specific keyword arguments (see module docstring)
    extra: dict[str, Any] = field(default_factory=dict)

    def __srl_check__(self) -> list[tuple[str, str]]:
        errors = []
        if self.input_shape is not None and any(int(d) <= 0 for d in self.input_shape):
            errors.append(("input_shape", f"dimensions must be positive, got {self.input_shape}"))
        return errors

    @classmethod
    def _parse(cls, d: Any, path: str, errors: list[str]) -> EncoderConfig | None:
        if not isinstance(d, Mapping):
            errors.append(f"{path}: expected a mapping, got {type(d).__name__}")
            return None
        d = dict(d)
        if "frame_stack" in d:
            value = d.pop("frame_stack")
            if value in (None, 1):
                warnings.warn(
                    f"{path}.frame_stack is deprecated and ignored: it was never applied "
                    "by ModelBuilder. Remove it.",
                    ConfigDeprecationWarning,
                    stacklevel=4,
                )
            else:
                errors.append(
                    f"{path}.frame_stack: frame stacking is not implemented by the model "
                    f"builder (got {value!r}); stack frames in an env wrapper and set "
                    "input_shape/input_dim accordingly"
                )
        for key in ("input_dim", "input_shape"):
            if _is_auto(d.get(key)):
                d[key] = None
        from srl.registry.registry import EncoderRegistry

        entry = _lookup_type(EncoderRegistry, d.get("type"), path, errors)
        fields_d, extra = _split_extra(d, cls, entry, path, errors)
        n = len(errors)
        obj = from_mapping(cls, fields_d, path, errors)
        if obj is None or len(errors) != n:
            return None
        obj.extra = extra
        if obj.aux_type == "autoencoder" and obj.input_shape is None and entry is not None:
            if "input_shape" not in (entry.requires or ()):
                errors.append(
                    f"{path}.aux_type: 'autoencoder' reconstructs an image and needs a "
                    f"[C, H, W] input_shape (a cnn encoder); type '{obj.type}' has none, so "
                    "no decoder would be built"
                )
        return obj

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: str = "encoder") -> EncoderConfig:
        errors: list[str] = []
        obj = cls._parse(d, path, errors)
        if errors or obj is None:
            raise ConfigError(errors)
        return obj


@dataclass
class HeadConfig(_Validated):
    name: str
    # any registered head type: "gaussian" | "squashed_gaussian" | "deterministic" |
    # "value" | "twin_q" | "q" | "q_function" | plugins
    type: str
    # None = "auto" (resolved from the env action space).
    action_dim: int | None = None
    layers: list[Any] = field(default_factory=list)
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    # Gaussian (PPO/A2C/A3C) heads only. False = a single learned,
    # state-independent log_std vector (CleanRL / SB3 default for PPO);
    # True = log_std is predicted per state by a separate linear layer.
    state_dependent_std: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    __srl_ranges__ = {"action_dim": AT_LEAST_ONE}

    def __srl_check__(self) -> list[tuple[str, str]]:
        if self.log_std_min >= self.log_std_max:
            return [
                (
                    "log_std_min",
                    f"must be < log_std_max ({self.log_std_max!r}), got {self.log_std_min!r}",
                )
            ]
        return []

    @classmethod
    def _parse(cls, d: Any, path: str, errors: list[str], default_name: str | None = None):
        if not isinstance(d, Mapping):
            errors.append(f"{path}: expected a mapping, got {type(d).__name__}")
            return None
        d = dict(d)
        if default_name is not None:
            d.setdefault("name", default_name)
        if _is_auto(d.get("action_dim")):
            d["action_dim"] = None
        from srl.registry.registry import HeadRegistry

        entry = _lookup_type(HeadRegistry, d.get("type"), path, errors)
        fields_d, extra = _split_extra(d, cls, entry, path, errors)
        n = len(errors)
        obj = from_mapping(cls, fields_d, path, errors)
        if obj is None or len(errors) != n:
            return None
        obj.extra = extra
        return obj

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: str = "head") -> HeadConfig:
        errors: list[str] = []
        obj = cls._parse(d, path, errors)
        if errors or obj is None:
            raise ConfigError(errors)
        return obj


# ---------------------------------------------------------------------------
# Loss config (deprecated)
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Deprecated: kept only for :meth:`srl.losses.LossComposer.from_loss_configs`.

    The YAML ``losses:`` block it used to be parsed from was never read by
    any algorithm and is no longer part of the schema.
    """

    name: str
    weight: float = 1.0
    # optional schedule: "linear_decay" | "cosine" | "constant"
    schedule: str = "constant"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LossConfig:
        allowed = cls.__dataclass_fields__.keys()
        known = {k: v for k, v in d.items() if k in allowed}
        extra = {k: v for k, v in d.items() if k not in allowed}
        obj = cls(**known)
        obj.extra = extra
        return obj


# ---------------------------------------------------------------------------
# Visualization / ROS2
# ---------------------------------------------------------------------------


@dataclass
class PipelineNodeConfig:
    id: str
    label: str = ""
    kind: str = "process"
    details: list[str] = field(default_factory=list)


@dataclass
class PipelineEdgeConfig:
    src: str
    dst: str
    label: str = ""

    @classmethod
    def from_any(cls, value: Any) -> PipelineEdgeConfig:
        if isinstance(value, str):
            src, dst = [part.strip() for part in value.split("->", 1)]
            return cls(src=src, dst=dst)
        return cls(src=value["src"], dst=value["dst"], label=value.get("label", ""))

    def to_yaml(self) -> Any:
        return f"{self.src} -> {self.dst}" if not self.label else dataclasses.asdict(self)


@dataclass
class _TrainingPipelineSection:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[Any] = field(default_factory=list)


@dataclass
class VisualizationConfig:
    save_model_pipeline: bool = False
    model_pipeline_path: str | None = None
    save_training_pipeline: bool = False
    training_pipeline_path: str | None = None
    export_only: bool = False
    training_pipeline_nodes: list[PipelineNodeConfig] = field(
        default_factory=list, metadata={"yaml": False}
    )
    training_pipeline_edges: list[PipelineEdgeConfig] = field(
        default_factory=list, metadata={"yaml": False}
    )

    @classmethod
    def _parse(cls, data: Any, path: str, errors: list[str]) -> VisualizationConfig | None:
        if not data:
            return cls()
        if not isinstance(data, Mapping):
            errors.append(f"{path}: expected a mapping, got {type(data).__name__}")
            return None
        data = dict(data)
        pipeline = data.pop("training_pipeline", None) or {}
        n = len(errors)
        obj = from_mapping(cls, data, path, errors)
        section = from_mapping(
            _TrainingPipelineSection, pipeline, f"{path}.training_pipeline", errors
        )
        nodes: list[PipelineNodeConfig] = []
        edges: list[PipelineEdgeConfig] = []
        if section is not None:
            for i, node in enumerate(section.nodes):
                node_path = f"{path}.training_pipeline.nodes[{i}]"
                node = dict(node) if isinstance(node, Mapping) else node
                if isinstance(node, dict) and "id" in node:
                    node.setdefault("label", node["id"])
                parsed = from_mapping(PipelineNodeConfig, node, node_path, errors)
                if parsed is not None:
                    nodes.append(parsed)
            for i, edge in enumerate(section.edges):
                edge_path = f"{path}.training_pipeline.edges[{i}]"
                if isinstance(edge, str):
                    if "->" not in edge:
                        errors.append(f"{edge_path}: expected 'src -> dst', got {edge!r}")
                        continue
                    edges.append(PipelineEdgeConfig.from_any(edge))
                else:
                    parsed_e = from_mapping(PipelineEdgeConfig, edge, edge_path, errors)
                    if parsed_e is not None:
                        edges.append(parsed_e)
        if obj is None or len(errors) != n:
            return None
        obj.training_pipeline_nodes = nodes
        obj.training_pipeline_edges = edges
        return obj

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VisualizationConfig:
        errors: list[str] = []
        obj = cls._parse(data, "visualization", errors)
        if errors or obj is None:
            raise ConfigError(errors)
        return obj

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.metadata.get("yaml", True)
        }
        if self.training_pipeline_nodes or self.training_pipeline_edges:
            out["training_pipeline"] = {
                "nodes": [dataclasses.asdict(n) for n in self.training_pipeline_nodes],
                "edges": [e.to_yaml() for e in self.training_pipeline_edges],
            }
        return out


@dataclass
class ROS2ObservationConfig:
    topic: str
    msg_type: str | None = None
    queue_size: int = 10

    @classmethod
    def from_any(cls, value: Any) -> ROS2ObservationConfig:
        if isinstance(value, str):
            return cls(topic=value)
        return cls(
            topic=value["topic"],
            msg_type=value.get("msg_type"),
            queue_size=int(value.get("queue_size", 10)),
        )


@dataclass
class ROS2Config:
    observations: dict[str, ROS2ObservationConfig] = field(default_factory=dict)
    action_topic: str = "/srl/action"
    action_msg_type: str | None = "Float32MultiArray"
    action_queue_size: int = 10

    @classmethod
    def _parse(cls, data: Any, path: str, errors: list[str]) -> ROS2Config | None:
        if not data:
            return cls()
        if not isinstance(data, Mapping):
            errors.append(f"{path}: expected a mapping, got {type(data).__name__}")
            return None
        data = dict(data)
        if "obs_topics" in data and "observations" not in data:
            data["observations"] = data.pop("obs_topics")
        raw_obs = data.pop("observations", None) or {}
        observations: dict[str, ROS2ObservationConfig] = {}
        if not isinstance(raw_obs, Mapping):
            errors.append(f"{path}.observations: expected a mapping of encoder -> topic")
        else:
            for name, spec in raw_obs.items():
                obs_path = f"{path}.observations.{name}"
                if isinstance(spec, str):
                    observations[name] = ROS2ObservationConfig(topic=spec)
                else:
                    parsed = from_mapping(ROS2ObservationConfig, spec, obs_path, errors)
                    if parsed is not None:
                        observations[name] = parsed
        n = len(errors)
        obj = from_mapping(cls, data, path, errors)
        if obj is None or len(errors) != n:
            return None
        obj.observations = observations
        return obj

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ROS2Config:
        errors: list[str] = []
        obj = cls._parse(data, "ros2", errors)
        if errors or obj is None:
            raise ConfigError(errors)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": {k: dataclasses.asdict(v) for k, v in self.observations.items()},
            "action_topic": self.action_topic,
            "action_msg_type": self.action_msg_type,
            "action_queue_size": self.action_queue_size,
        }


# ---------------------------------------------------------------------------
# Top-level agent config
# ---------------------------------------------------------------------------


def _parse_flow(spec: Any, path: str, errors: list[str]) -> tuple[str, str] | None:
    if not isinstance(spec, str):
        errors.append(f"{path}: expected a 'src -> dst' string, got {spec!r}")
        return None
    text = spec.strip()
    if not text or text.startswith("#"):
        return None
    parts = [p.strip() for p in text.split("->")]
    if len(parts) != 2 or not all(parts):
        errors.append(f"{path}: expected exactly one 'src -> dst', got {spec!r}")
        return None
    return parts[0], parts[1]


@dataclass
class AgentModelConfig:
    """Full model config parsed from a YAML file.

    YAML structure example::

        encoders:
          - name: visual_enc
            type: cnn
            input_shape: [3, 64, 64]   # or: auto
            latent_dim: 256
            aux_type: autoencoder
            layers: [[32,3,0], [64,3,0], [64,3,0]]
          - name: state_enc
            type: mlp
            input_dim: auto            # resolved from the env
            latent_dim: 64
            layers: [128, 64]

        flows:
          - "visual_enc -> actor"
          - "state_enc  -> actor"
          - "visual_enc -> critic"
          - "state_enc  -> critic"

        actor:
          name: actor
          type: squashed_gaussian
          action_dim: auto             # or omit

        critic:
          name: critic
          type: twin_q
    """

    encoders: list[EncoderConfig] = field(default_factory=list)
    flows: list[str] = field(default_factory=list)
    actor: HeadConfig | None = None
    critic: HeadConfig | None = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    ros2: ROS2Config = field(default_factory=ROS2Config)

    MODEL_KEYS = ("encoders", "flows", "actor", "critic", "visualization", "ros2")

    @classmethod
    def _parse(
        cls, d: Any, errors: list[str], *, allowed_extra: frozenset[str] = EXPERIMENT_KEYS
    ) -> AgentModelConfig | None:
        if not isinstance(d, Mapping):
            errors.append(f"model config: expected a mapping, got {type(d).__name__}")
            return None
        for key in d:
            if key in cls.MODEL_KEYS or key in allowed_extra:
                continue
            if key in DEPRECATED_MODEL_KEYS:
                warnings.warn(
                    f"'{key}:' is deprecated and ignored: {DEPRECATED_MODEL_KEYS[key]}",
                    ConfigDeprecationWarning,
                    stacklevel=4,
                )
                continue
            valid = set(cls.MODEL_KEYS) | set(allowed_extra)
            hint = suggest(str(key), valid) or f"; valid top-level keys: {', '.join(sorted(valid))}"
            errors.append(f"{key}: unknown top-level key{hint}")

        raw_encoders = d.get("encoders") or []
        encoders: list[EncoderConfig] = []
        if not isinstance(raw_encoders, list):
            errors.append("encoders: expected a list of encoder mappings")
        else:
            for i, enc in enumerate(raw_encoders):
                parsed = EncoderConfig._parse(enc, f"encoders[{i}]", errors)
                if parsed is not None:
                    encoders.append(parsed)

        heads: dict[str, HeadConfig | None] = {}
        for role in ("actor", "critic"):
            raw = d.get(role)
            heads[role] = HeadConfig._parse(raw, role, errors, default_name=role) if raw else None

        flows_raw = d.get("flows") or []
        flows: list[str] = []
        if not isinstance(flows_raw, list):
            errors.append("flows: expected a list of 'src -> dst' strings")
        else:
            nodes = [e.name for e in encoders] + [h.name for h in heads.values() if h is not None]
            for i, spec in enumerate(flows_raw):
                parsed_flow = _parse_flow(spec, f"flows[{i}]", errors)
                if isinstance(spec, str):
                    flows.append(spec)
                if parsed_flow is None:
                    continue
                for end, node in zip(("source", "destination"), parsed_flow, strict=True):
                    if node not in nodes:
                        errors.append(
                            f"flows[{i}]: {end} '{node}' is not a declared encoder or head"
                            f"{suggest(node, nodes)}; declared: {nodes}"
                        )

        visualization = VisualizationConfig._parse(d.get("visualization"), "visualization", errors)
        ros2 = ROS2Config._parse(d.get("ros2"), "ros2", errors)
        if errors:
            return None
        return cls(
            encoders=encoders,
            flows=flows,
            actor=heads["actor"],
            critic=heads["critic"],
            visualization=visualization or VisualizationConfig(),
            ros2=ros2 or ROS2Config(),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentModelConfig:
        """Strictly parse the model section of ``d``.

        Experiment-level keys (``train``, ``algo``, ``env_id``, ...) are
        accepted and ignored here -- validate them with
        :func:`srl.config.load_experiment`.
        """
        errors: list[str] = []
        obj = cls._parse(d, errors)
        if errors or obj is None:
            raise ConfigError(errors)
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Plain-YAML representation that :meth:`from_dict` parses back identically."""

        def _entity(obj: Any) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for f in dataclasses.fields(obj):
                value = getattr(obj, f.name)
                if f.name == "extra":
                    if value:
                        out["extra"] = dict(value)
                    continue
                if value is None and f.default is None:
                    continue
                out[f.name] = _plain(value)
            return out

        out: dict[str, Any] = {
            "encoders": [_entity(e) for e in self.encoders],
            "flows": list(self.flows),
        }
        if self.actor is not None:
            out["actor"] = _entity(self.actor)
        if self.critic is not None:
            out["critic"] = _entity(self.critic)
        if self.visualization != VisualizationConfig():
            out["visualization"] = self.visualization.to_dict()
        if self.ros2 != ROS2Config():
            out["ros2"] = self.ros2.to_dict()
        return out


def _plain(value: Any) -> Any:
    """Convert tuples to lists recursively (YAML-safe, round-trips to the same config)."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return value


__all__ = [
    "AUX_TYPES",
    "AgentModelConfig",
    "EncoderConfig",
    "HeadConfig",
    "LayerConfig",
    "LossConfig",
    "PipelineEdgeConfig",
    "PipelineNodeConfig",
    "ROS2Config",
    "ROS2ObservationConfig",
    "VisualizationConfig",
]
