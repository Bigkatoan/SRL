"""Shared helpers for SRL vectorised environments.

Final-observation contract
--------------------------
Every SRL vector env (``SyncVectorEnv``, ``AsyncVectorEnv``,
``IsaacLabWrapper``) auto-resets sub-envs whose episode just ended, so the
``next_obs`` returned from ``step()`` holds the NEXT episode's first frame
for those slots -- correct for choosing the next action, wrong for value
bootstrapping at a time-limit truncation (and for storing ``next_obs`` in a
replay buffer). The true terminal observation is exposed under one key,
``"final_obs"``:

* CPU vector envs (``SyncVectorEnv``/``AsyncVectorEnv``) return ``infos`` as
  a list of per-env dicts; ``infos[i]["final_obs"]`` is env ``i``'s terminal
  observation dict, present only on the step its episode ended.
* ``IsaacLabWrapper`` returns one batched ``info`` dict;
  ``info["final_obs"]`` is the batched (pre-reset) observation dict and
  ``info["_final_obs"]`` the boolean mask of sub-envs that ended (Gymnasium
  1.x style). Only present when the wrapper can really recover it
  (``supports_partial_reset=True``).

Callers should not parse either form themselves: use :func:`get_final_obs`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FINAL_OBS_KEY = "final_obs"
FINAL_OBS_MASK_KEY = "_final_obs"
# Legacy key still written by IsaacLabWrapper for backward compatibility.
LEGACY_FINAL_OBS_KEY = "true_final_obs"


def get_final_obs(next_obs: dict[str, Any], info: Any) -> dict[str, Any]:
    """Return ``next_obs`` with every just-ended sub-env's slot replaced by
    its TRUE terminal observation.

    The result is what a transition's ``next_obs`` / a truncation bootstrap
    should use. ``next_obs`` itself is never mutated. Works for a single
    non-vectorised env too (``info`` without any final-obs entry), in which
    case ``next_obs`` is returned unchanged -- single-env wrappers never
    auto-reset, so their ``next_obs`` already is the terminal observation.
    """
    if isinstance(info, dict):
        if FINAL_OBS_KEY in info:
            final = info[FINAL_OBS_KEY]
            mask = info.get(FINAL_OBS_MASK_KEY)
            if mask is None:
                return final
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if not mask.any():
                return next_obs
            out = {k: np.array(v, copy=True) for k, v in next_obs.items()}
            for k in out:
                out[k][mask] = np.asarray(final[k])[mask]
            return out
        if LEGACY_FINAL_OBS_KEY in info:
            return info[LEGACY_FINAL_OBS_KEY]
        return next_obs

    if isinstance(info, (list, tuple)):
        ended = [
            (i, d[FINAL_OBS_KEY])
            for i, d in enumerate(info)
            if isinstance(d, dict) and FINAL_OBS_KEY in d
        ]
        if not ended:
            return next_obs
        out = {k: np.array(v, copy=True) for k, v in next_obs.items()}
        for i, final_i in ended:
            for k in out:
                out[k][i] = np.asarray(final_i[k])
        return out

    return next_obs
