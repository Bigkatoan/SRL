"""srl.registry — model registry, flow graph, config schema, builder."""

from srl.registry.builder import ModelBuilder
from srl.registry.config_schema import AgentModelConfig, EncoderConfig, HeadConfig, LossConfig
from srl.registry.flow_graph import FlowGraph
from srl.registry.registry import (
    AlgorithmRegistry,
    AlgorithmSpec,
    EncoderRegistry,
    EnvRegistry,
    HeadRegistry,
    RegistryEntry,
    register_algorithm,
    register_encoder,
    register_env,
    register_head,
)

__all__ = [
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "EncoderRegistry",
    "EnvRegistry",
    "HeadRegistry",
    "RegistryEntry",
    "register_algorithm",
    "register_encoder",
    "register_env",
    "register_head",
    "FlowGraph",
    "AgentModelConfig",
    "EncoderConfig",
    "HeadConfig",
    "LossConfig",
    "ModelBuilder",
]
