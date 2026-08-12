"""Model and Loss registries with decorator API."""

from __future__ import annotations

from collections.abc import Callable


class _Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: dict[str, type] = {}

    def register(self, name: str) -> Callable:
        def decorator(cls: type) -> type:
            key = name.lower()
            if key in self._store:
                raise KeyError(f"{self._kind} '{key}' already registered.")
            self._store[key] = cls
            return cls

        return decorator

    def get(
        self,
        name: str,
    ) -> type:
        key = name.lower()
        if key not in self._store:
            raise ValueError(
                f"Unknown {self._kind} '{name}'. " f"Registered: {sorted(self._store)}"
            )
        return self._store[key]

    def available(self) -> list[str]:
        return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._store


# Singleton registries
EncoderRegistry = _Registry("encoder")
HeadRegistry = _Registry("head")
LossRegistry = _Registry("loss")

# Convenience decorators
register_encoder = EncoderRegistry.register
register_head = HeadRegistry.register
register_loss = LossRegistry.register
