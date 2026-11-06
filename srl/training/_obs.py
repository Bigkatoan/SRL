"""Observation/transition plumbing shared by the trainers and the evaluator."""

from __future__ import annotations

from typing import Any

__all__ = ["agent_act", "obs_to_tensors", "remap_obs_to_encoders", "split_vector_transition"]


def agent_act(agent: Any, obs: dict, hidden: Any = None, deterministic: bool = False):
    """Low-level tensor action selection: ``(action, log_prob, value, hidden)``.

    Calls :meth:`BaseAgent.act`; falls back to a legacy duck-typed agent's
    tensor ``predict()`` (same 4-tuple contract) for objects that are not
    :class:`~srl.core.base_agent.BaseAgent` subclasses.
    """
    act = getattr(agent, "act", None)
    if act is None:
        act = agent.predict
    if hidden is None:
        return act(obs, deterministic=deterministic)
    return act(obs, hidden, deterministic=deterministic)


def encoder_layout(agent: Any) -> tuple[list[str], Any]:
    model = agent.model
    return list(model.encoders.keys()), getattr(model, "encoder_input_names", None)


def remap_obs_to_encoders(
    obs_dict: dict,
    encoder_names: list[str],
    encoder_input_names: dict[str, str | None] | None = None,
) -> dict:
    """Map observation dict keys → encoder input names.

    How the model receives multi-modal observations (image + vector)
    ----------------------------------------------------------------
    The obs dict returned by the environment must ultimately have keys that
    match the encoder names defined in the YAML config.  Three cases:

    Case 1 — keys already match (most explicit, always recommended):
        env returns  {"cnn_enc": <(N,3,H,W) image>,  "mlp_enc": <(N,8) state>}
        YAML encoders: cnn_enc (type: cnn), mlp_enc (type: mlp)
        → passthrough, no remapping needed.

    Case 2 — single obs, single encoder (Isaac Lab default for image-only envs):
        env returns  {"policy": <(N,3,H,W) image>}
        YAML encoder: policy_enc (type: cnn)
        → rename "policy" → "policy_enc" automatically.

    Case 3 — multiple obs groups, multiple encoders, same count:
        env returns  {"policy": <image>, "critic":  <state>}         (2 keys)
        YAML encoders: policy_enc (cnn), critic_enc (mlp)            (2 encoders)
        → zip by order: "policy" → policy_enc, "critic" → critic_enc.
        NOTE: order matters here.  Name your encoders so they match
        the env's obs group names to avoid relying on dict order.

    Matching rules applied in order:
      0. Encoder has input_name set       → route by that explicit obs key.
      1. Any obs key already equals an encoder name → passthrough.
      2. 1 obs, 1 encoder                 → rename obs key to encoder name.
      3. N obs, N encoders (N > 1)        → zip obs values to encoder names.
      4. Anything else                    → passthrough (model handles it).

    Validation:
      - Missing explicit input_name key   → KeyError.
      - Unused obs keys after explicit routing → warnings.warn.
    """
    from srl.utils.obs_remap import apply_obs_remap

    return apply_obs_remap(obs_dict, encoder_names, encoder_input_names)


def obs_to_tensors(obs_dict: dict, device, *, force_batch: bool) -> dict:
    """``force_batch=True`` means: this obs dict came from a single env step
    that *may or may not* already carry an explicit leading batch dim --
    isaaclab/mjlab always pre-batch (even at num_envs=1: vectors come back
    (1, D), images (1, C, H, W)), while a plain gymnasium/goal/racecar env
    never does (vectors (D,), images (C, H, W)).

    Both unbatched shapes have odd ndim (1 for a vector, 3 for an image) and
    both pre-batched shapes have even ndim (2, 4) -- so parity of ndim tells
    us whether to add the batch dim, regardless of whether this is a flat
    vector or an image. (A shape[0]-based heuristic here previously
    mistook an unbatched (C, H, W) image's channel count for an existing
    batch size and skipped expanding it, crashing the first real eval/
    visualize step against any CNN-encoder config.)
    """
    import numpy as np
    import torch

    tensor_obs = {}
    for key, value in obs_dict.items():
        arr = np.asarray(value)
        if force_batch and (arr.ndim == 0 or arr.ndim % 2 == 1):
            arr = np.expand_dims(arr, axis=0)
        tensor_obs[key] = torch.from_numpy(arr).float().to(device)
    return tensor_obs


def split_vector_transition(
    obs: dict, next_obs: dict, action, reward, done, trunc
) -> list[tuple[dict, dict, object, float, bool, bool]]:
    """Split a batched (vectorized-env) step into per-env transition tuples.

    `done`/`trunc` are kept SEPARATE in the returned tuples (last two fields:
    terminated, truncated) rather than OR'd into one -- off-policy bootstrap
    targets (`(1 - terminated) * next_q` in sac.py/ddpg.py/td3.py) must see
    true termination only. Callers that want "episode ended for any reason"
    (resets, episode-length logging) OR these two themselves at the point
    they need it; do not re-combine them here.
    """
    import numpy as np

    rewards = np.asarray(reward, dtype=np.float32).reshape(-1)
    dones = np.asarray(done, dtype=bool).reshape(-1)
    truncs = np.asarray(trunc, dtype=bool).reshape(-1)
    actions = np.asarray(action)
    if actions.ndim == 1:
        actions = np.expand_dims(actions, axis=0)

    transitions = []
    for index in range(len(rewards)):
        obs_i = {k: np.asarray(v)[index] for k, v in obs.items()}
        next_obs_i = {k: np.asarray(v)[index] for k, v in next_obs.items()}
        transitions.append(
            (
                obs_i,
                next_obs_i,
                np.asarray(actions[index], dtype=np.float32),
                float(rewards[index]),
                bool(dones[index]),
                bool(truncs[index]),
            )
        )
    return transitions
