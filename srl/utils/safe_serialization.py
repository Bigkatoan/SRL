"""Pickle-free (``torch.load(weights_only=True)``) serialization helpers.

``torch.load(..., weights_only=False)`` unpickles arbitrary objects, i.e. it
runs code from the file. SRL's own checkpoint files are written so they load
with ``weights_only=True``: :func:`pack` turns the few non-tensor objects that
appear in agent state (numpy arrays, numpy scalars and dtypes, ``torch.device``)
into tensors / primitives with a small tag, and :func:`unpack` reverses it.
Anything else raises :class:`TypeError` at *save* time, naming the offending
path, instead of silently producing a file that needs pickle to read.

:func:`to_jsonable` / :func:`from_jsonable` do the same for JSON
(``normalizers.json``): arrays become ``{"__ndarray__": [...], "dtype": ...}``
(float64 values round-trip exactly through Python's float repr).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "LegacyCheckpointWarning",
    "from_jsonable",
    "load_trusted_or_legacy",
    "load_weights_only",
    "pack",
    "to_jsonable",
    "unpack",
]

_ND = "__srl_ndarray__"
_NP_DTYPE = "__srl_np_dtype__"
_DEVICE = "__srl_torch_device__"
# numpy dtypes torch.from_numpy cannot hold; widened on save, narrowed on load.
_WIDEN = {"uint16": np.int64, "uint32": np.int64, "uint64": np.int64}


class LegacyCheckpointWarning(UserWarning):
    """A pre-v1 (pickle-based) checkpoint was loaded with ``weights_only=False``."""


def pack(obj: Any, _path: str = "") -> Any:
    """Recursively convert *obj* into something ``weights_only=True`` can load."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes, torch.Tensor, torch.dtype)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError(f"{_path or '<root>'}: object arrays cannot be stored safely")
        name = obj.dtype.name
        arr = np.ascontiguousarray(obj)
        if name in _WIDEN:
            arr = arr.astype(_WIDEN[name])
        if not arr.flags.writeable:
            arr = arr.copy()
        return {_ND: torch.from_numpy(arr), "dtype": name}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.dtype):
        return {_NP_DTYPE: obj.name}
    if isinstance(obj, type) and issubclass(obj, np.generic):
        return {_NP_DTYPE: np.dtype(obj).name, "scalar_type": True}
    if isinstance(obj, torch.device):
        return {_DEVICE: str(obj)}
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"{_path or '<root>'}: unsupported dict key {key!r}")
            out[key] = pack(value, f"{_path}.{key}" if _path else str(key))
        return out
    if isinstance(obj, tuple):
        return tuple(pack(v, f"{_path}[{i}]") for i, v in enumerate(obj))
    if isinstance(obj, list):
        return [pack(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    if hasattr(obj, "__iter__") and type(obj).__name__ == "deque":
        return [pack(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    raise TypeError(
        f"{_path or '<root>'}: cannot store {type(obj).__module__}.{type(obj).__qualname__} "
        "without pickle; convert it to tensors / numbers / strings first"
    )


def unpack(obj: Any) -> Any:
    """Inverse of :func:`pack`."""
    if isinstance(obj, dict):
        if _ND in obj:
            arr = obj[_ND].cpu().numpy()
            return arr.astype(obj.get("dtype", arr.dtype.name))
        if _NP_DTYPE in obj:
            dtype = np.dtype(obj[_NP_DTYPE])
            return dtype.type if obj.get("scalar_type") else dtype
        if _DEVICE in obj:
            return torch.device(obj[_DEVICE])
        return {k: unpack(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(unpack(v) for v in obj)
    if isinstance(obj, list):
        return [unpack(v) for v in obj]
    return obj


def load_weights_only(path: str | Path, map_location: Any = "cpu") -> Any:
    """``torch.load(path, weights_only=True)`` + :func:`unpack`."""
    return unpack(torch.load(path, map_location=map_location, weights_only=True))


def load_trusted_or_legacy(path: str | Path, map_location: Any = "cpu") -> Any:
    """Load a ``.pt`` written by :func:`pack` + ``torch.save`` (``weights_only=True``).

    Files from older SRL versions need pickle: they are loaded with
    ``weights_only=False`` after a :class:`LegacyCheckpointWarning`.
    """
    try:
        return load_weights_only(path, map_location)
    except pickle.UnpicklingError:
        warnings.warn(
            f"{path}: legacy SRL checkpoint (pre format v1); loading it with "
            "torch.load(weights_only=False), which can execute code from the file -- "
            "only load checkpoints you trust. Re-save it (agent.save(dir)) to upgrade.",
            LegacyCheckpointWarning,
            stacklevel=3,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def to_jsonable(obj: Any) -> Any:
    """Arrays / tensors -> tagged lists, recursively (for ``json.dumps``)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        return {
            "__tensor__": t.tolist(),
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
        }
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": obj.dtype.name, "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    raise TypeError(f"cannot store {type(obj).__name__} in JSON")


def from_jsonable(obj: Any) -> Any:
    """Inverse of :func:`to_jsonable`."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            arr = np.asarray(obj["__ndarray__"], dtype=obj["dtype"])
            return arr.reshape(obj.get("shape", arr.shape))
        if "__tensor__" in obj:
            t = torch.tensor(obj["__tensor__"], dtype=getattr(torch, obj["dtype"]))
            return t.reshape(obj.get("shape", t.shape))
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj
