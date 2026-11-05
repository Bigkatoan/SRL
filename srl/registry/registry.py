"""Name -> implementation registries with decorator API and entry-point plugins.

Four registries share one implementation:

=====================  ======================  =====================================
registry               entry-point group       registered object
=====================  ======================  =====================================
``EncoderRegistry``    ``srl.encoders``        ``factory(cfg: EncoderConfig) -> nn.Module``
``HeadRegistry``       ``srl.heads``           ``factory(cfg: HeadConfig, input_dim) -> nn.Module``
``AlgorithmRegistry``  ``srl.algorithms``      :class:`AlgorithmSpec`
``EnvRegistry``        ``srl.envs``            ``factory(env_id, *, device, n_envs, render) -> env``
=====================  ======================  =====================================

Built-in encoders and heads (``mlp``, ``cnn``, ``squashed_gaussian``, ...) are
registered through exactly the same API as user types (see
:mod:`srl.registry.builtins`), so there is no special-casing in the builder and
the "unknown type" error lists everything that is really available.

Third-party packages can register types without being imported explicitly,
via ``importlib.metadata`` entry points in their ``pyproject.toml``::

    [project.entry-points."srl.encoders"]
    pointnet = "my_pkg.encoders:PointNetEncoder"

Entry points are discovered lazily: only their *names* are read up front
(``available()`` / ``in``), and an entry point's module is imported the first
time that name is looked up.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from srl.config._validate import ConfigError, suggest

__all__ = [
    "RegistryEntry",
    "AlgorithmSpec",
    "EncoderRegistry",
    "HeadRegistry",
    "AlgorithmRegistry",
    "EnvRegistry",
    "register_encoder",
    "register_head",
    "register_algorithm",
    "register_env",
]


@dataclass
class RegistryEntry:
    """A registered object plus the metadata the config validator/builder needs.

    Attributes
    ----------
    obj:
        The registered class / factory / spec.
    requires:
        Config attribute names that must be set (not ``None``) for this type,
        e.g. ``("input_dim",)`` for ``mlp`` -- checked before building, after
        ``auto`` dimensions are resolved.
    extra_keys:
        Additional, type-specific YAML keys accepted at the top level of the
        encoder/head mapping and collected into ``cfg.extra``.
    allow_extra:
        Accept *any* unknown key into ``cfg.extra`` (no typo detection). Use
        ``extra_keys`` instead whenever the set is known.
    requires_action_dim:
        Heads only: the head needs ``action_dim`` (actors, Q-critics); it is
        filled from the env action space by ``resolve_auto_dims``.
    source:
        ``"builtin"``, ``"user"`` (decorator) or ``"entry-point:<dist>"``.
    """

    obj: Any
    requires: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()
    allow_extra: bool = False
    requires_action_dim: bool = False
    source: str = "user"


def _entry_points(group: str) -> list[metadata.EntryPoint]:
    """All installed entry points of ``group`` (indirection so tests can fake it)."""
    try:
        return list(metadata.entry_points(group=group))
    except Exception:  # pragma: no cover - broken environment metadata
        return []


class _Registry:
    def __init__(self, kind: str, group: str | None = None, bootstrap: str | None = None) -> None:
        self._kind = kind
        self._group = group
        self._bootstrap = bootstrap
        self._bootstrapped = bootstrap is None
        self._store: dict[str, RegistryEntry] = {}
        self._plugins: dict[str, metadata.EntryPoint] | None = None

    # ------------------------------------------------------------------ setup

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def group(self) -> str | None:
        return self._group

    def _ensure_builtins(self) -> None:
        if not self._bootstrapped:
            self._bootstrapped = True
            importlib.import_module(self._bootstrap)  # type: ignore[arg-type]

    def _discover(self) -> dict[str, metadata.EntryPoint]:
        if self._plugins is None:
            self._plugins = {}
            if self._group:
                for ep in _entry_points(self._group):
                    self._plugins.setdefault(ep.name.lower(), ep)
        return self._plugins

    def refresh_plugins(self) -> None:
        """Forget discovered entry points so the next lookup re-scans them."""
        self._plugins = None

    # --------------------------------------------------------------- register

    def register(
        self,
        name: str,
        obj: Any = None,
        *,
        override: bool = False,
        requires: Iterable[str] = (),
        extra_keys: Iterable[str] = (),
        allow_extra: bool = False,
        requires_action_dim: bool = False,
        source: str = "user",
    ) -> Callable:
        """Register ``obj`` under ``name``; usable as a decorator or directly.

        ``@register_encoder("pointnet")`` or ``register_encoder("pointnet", PointNet)``.
        Registering a name twice raises unless ``override=True``.
        """

        def decorator(target: Any) -> Any:
            self._ensure_builtins()
            key = name.lower()
            if key in self._store and not override:
                raise KeyError(
                    f"{self._kind} '{key}' already registered "
                    f"({self._store[key].source}); pass override=True to replace it."
                )
            self._store[key] = RegistryEntry(
                obj=target,
                requires=tuple(requires) or tuple(getattr(target, "srl_requires", ())),
                extra_keys=tuple(extra_keys) or tuple(getattr(target, "srl_extra_keys", ())),
                allow_extra=allow_extra or bool(getattr(target, "srl_allow_extra", False)),
                requires_action_dim=requires_action_dim
                or bool(getattr(target, "srl_requires_action_dim", False)),
                source=source,
            )
            return target

        if obj is not None:
            return decorator(obj)
        return decorator

    def unregister(self, name: str) -> None:
        self._store.pop(name.lower(), None)

    # ----------------------------------------------------------------- lookup

    def entry(self, name: str) -> RegistryEntry:
        """Full :class:`RegistryEntry` for ``name`` (loads an entry-point plugin if needed)."""
        self._ensure_builtins()
        key = str(name).lower()
        if key in self._store:
            return self._store[key]
        ep = self._discover().get(key)
        if ep is not None:
            try:
                loaded = ep.load()
            except Exception as exc:
                dist = getattr(getattr(ep, "dist", None), "name", None) or "unknown distribution"
                raise ImportError(
                    f"Failed to load {self._kind} plugin '{key}' "
                    f"(entry point {ep.value!r} in group {self._group!r}, from {dist}): {exc}"
                ) from exc
            dist = getattr(getattr(ep, "dist", None), "name", None)
            self.register(key, loaded, source=f"entry-point:{dist}" if dist else "entry-point")
            return self._store[key]
        available = self.available()
        raise ConfigError(
            f"Unknown {self._kind} '{name}'{suggest(key, available)}. " f"Available: {available}"
        )

    def get(self, name: str) -> Any:
        """The registered object for ``name``."""
        return self.entry(name).obj

    def available(self) -> list[str]:
        """Every registered name, including not-yet-loaded entry-point plugins."""
        self._ensure_builtins()
        return sorted(set(self._store) | set(self._discover()))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        self._ensure_builtins()
        key = name.lower()
        return key in self._store or key in self._discover()

    def __repr__(self) -> str:
        return f"<{self._kind} registry: {self.available()}>"


@dataclass
class AlgorithmSpec:
    """How to build an algorithm from config.

    ``agent`` is either the agent class or an ``"module:attr"`` import path
    (resolved lazily by :meth:`agent_class`, so registering the built-ins does
    not import torch-heavy algorithm modules). ``visual_config_cls`` is the
    auxiliary-encoder-loss variant (``VisualPPOConfig`` / ``VisualSACConfig``)
    or ``None``. ``off_policy`` selects replay-buffer training loops.
    """

    name: str
    config_cls: type
    agent: Any
    off_policy: bool = False
    visual_config_cls: type | None = None
    actor_types: tuple[str, ...] = field(default_factory=tuple)
    critic_types: tuple[str, ...] = field(default_factory=tuple)

    def agent_class(self) -> type:
        if isinstance(self.agent, str):
            module_name, _, attr = self.agent.partition(":")
            return getattr(importlib.import_module(module_name), attr)
        return self.agent


EncoderRegistry = _Registry("encoder", "srl.encoders", bootstrap="srl.registry.builtins")
HeadRegistry = _Registry("head", "srl.heads", bootstrap="srl.registry.builtins")
AlgorithmRegistry = _Registry("algorithm", "srl.algorithms", bootstrap="srl.registry.builtins")
EnvRegistry = _Registry("env type", "srl.envs", bootstrap="srl.registry.builtins")

# Convenience decorators
register_encoder = EncoderRegistry.register
register_head = HeadRegistry.register
register_algorithm = AlgorithmRegistry.register
register_env = EnvRegistry.register
