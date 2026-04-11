"""Shared observation-key remapping for YAML-defined encoder graphs."""

from __future__ import annotations

from typing import Any
import warnings


def apply_obs_remap(
    obs_dict: dict[str, Any],
    encoder_names: list[str],
    encoder_input_names: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Map observation dict keys onto encoder names.

    Rules applied in order:
    0. Explicit `input_name` mapping.
    1. Exact key == encoder name passthrough.
    2. Single obs -> single encoder rename.
    3. Same-count zip by order.
    4. Fallback passthrough.

    Validation:
    - Missing explicit `input_name` raises ``KeyError``.
    - Unused keys after explicit routing emit ``warnings.warn``.
    """
    if not obs_dict:
        return obs_dict

    remapped: dict[str, Any] = {}
    used_obs_keys: set[str] = set()
    encoder_input_names = encoder_input_names or {}

    named_encoders = {
        enc_name: input_name
        for enc_name, input_name in encoder_input_names.items()
        if input_name and enc_name in encoder_names
    }
    for enc_name, input_name in named_encoders.items():
        if input_name not in obs_dict:
            raise KeyError(
                f"Missing observation key '{input_name}' required by encoder '{enc_name}'."
            )
        remapped[enc_name] = obs_dict[input_name]
        used_obs_keys.add(input_name)

    unnamed_encoders = [name for name in encoder_names if name not in remapped]
    remaining_obs = {k: v for k, v in obs_dict.items() if k not in used_obs_keys}

    if not remaining_obs or not unnamed_encoders:
        fallback_mapping: dict[str, Any] = {}
    elif any(name in remaining_obs for name in unnamed_encoders):
        fallback_mapping = remaining_obs
        used_obs_keys.update(key for key in remaining_obs if key in unnamed_encoders)
    elif len(remaining_obs) == 1 and len(unnamed_encoders) == 1:
        fallback_mapping = {unnamed_encoders[0]: next(iter(remaining_obs.values()))}
        used_obs_keys.update(remaining_obs.keys())
    elif len(remaining_obs) == len(unnamed_encoders) and len(remaining_obs) > 1:
        fallback_mapping = dict(zip(unnamed_encoders, remaining_obs.values()))
        used_obs_keys.update(remaining_obs.keys())
    else:
        fallback_mapping = remaining_obs

    remapped.update(fallback_mapping)

    if named_encoders:
        unused_keys = [key for key in obs_dict.keys() if key not in used_obs_keys]
        if unused_keys:
            warnings.warn(
                "Unused observation keys after encoder input routing: "
                + ", ".join(sorted(unused_keys)),
                stacklevel=2,
            )

    return remapped