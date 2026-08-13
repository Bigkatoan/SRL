"""Regression tests for `_obs_to_tensors`'s force_batch batch-dim heuristic.

`force_batch=True` is used at eval/`--visualize` call sites, where the obs
dict came from a single env step that *may or may not* already carry a
leading batch dim: isaaclab/mjlab always pre-batch (even at num_envs=1), a
plain gymnasium/goal/racecar env never does. The old heuristic
(`arr.ndim > 1 and arr.shape[0] >= 1` => "already batched") mistook an
unbatched (C, H, W) image's channel count for an existing batch size and
skipped expanding it -- crashing the first real eval step against any
CNN-encoder config with a channel-count mismatch several layers downstream.
"""

from __future__ import annotations

from srl.cli.train import _obs_to_tensors


def test_force_batch_expands_unbatched_flat_state_vector() -> None:
    out = _obs_to_tensors({"state": [1.0, 2.0, 3.0, 4.0]}, "cpu", force_batch=True)
    assert tuple(out["state"].shape) == (1, 4)


def test_force_batch_expands_unbatched_chw_image() -> None:
    import numpy as np

    obs = {"state": np.zeros((3, 96, 96), dtype=np.float32)}
    out = _obs_to_tensors(obs, "cpu", force_batch=True)
    assert tuple(out["state"].shape) == (1, 3, 96, 96)


def test_force_batch_does_not_double_batch_isaaclab_style_vector() -> None:
    import numpy as np

    # isaaclab/mjlab always pre-batch, even at num_envs=1: (1, D) not (D,).
    obs = {"policy": np.zeros((1, 8), dtype=np.float32)}
    out = _obs_to_tensors(obs, "cpu", force_batch=True)
    assert tuple(out["policy"].shape) == (1, 8)


def test_force_batch_does_not_double_batch_isaaclab_style_image() -> None:
    import numpy as np

    obs = {"policy": np.zeros((1, 3, 96, 96), dtype=np.float32)}
    out = _obs_to_tensors(obs, "cpu", force_batch=True)
    assert tuple(out["policy"].shape) == (1, 3, 96, 96)


def test_force_batch_false_never_expands() -> None:
    import numpy as np

    obs = {"state": np.zeros((3, 96, 96), dtype=np.float32)}
    out = _obs_to_tensors(obs, "cpu", force_batch=False)
    assert tuple(out["state"].shape) == (3, 96, 96)
