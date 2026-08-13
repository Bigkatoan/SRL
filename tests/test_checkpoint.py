import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from srl.utils.checkpoint import CheckpointManager


def test_checkpoint_manager_round_trip_torch_save(tmp_path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)

    path = manager.save(model, optimizer=optimizer, step=12, metrics={"score": 1.5})

    reloaded = nn.Linear(4, 2)
    reloaded_optimizer = torch.optim.Adam(reloaded.parameters(), lr=1e-3)
    payload = manager.load(reloaded, path, optimizer=reloaded_optimizer)

    assert payload["step"] == 12
    assert payload["metrics"] == {"score": 1.5}


def test_checkpoint_manager_load_restores_rng_state_reproducibly(tmp_path) -> None:
    model = nn.Linear(4, 2)
    manager = CheckpointManager(tmp_path)
    path = manager.save(model, step=1)

    random.random()
    np.random.rand()
    torch.rand(3)  # advance all three RNGs past the saved state

    reloaded = nn.Linear(4, 2)
    manager.load(reloaded, path)
    draw1 = (random.random(), np.random.rand(), torch.rand(3).clone())

    manager.load(reloaded, path)
    draw2 = (random.random(), np.random.rand(), torch.rand(3).clone())

    assert draw1[0] == draw2[0]
    assert draw1[1] == draw2[1]
    assert torch.equal(draw1[2], draw2[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_checkpoint_manager_load_onto_cuda_does_not_crash_restoring_rng_state(tmp_path) -> None:
    """Regression test: CheckpointManager.load(..., device="cuda") passes
    device as torch.load's map_location, which relocates EVERY tensor in the
    payload onto that device -- RNG-state tensors included. torch.set_rng_state()
    requires a CPU ByteTensor regardless of what device the model weights end
    up on; --resume on a GPU run used to crash here with "RNG state must be a
    torch.ByteTensor" before _restore_rng_state pinned these tensors back to
    CPU first."""
    model = nn.Linear(4, 2).cuda()
    manager = CheckpointManager(tmp_path)
    path = manager.save(model, step=1)

    reloaded = nn.Linear(4, 2).cuda()
    manager.load(reloaded, path, device="cuda")  # must not raise


def test_checkpoint_manager_latest_returns_recent_path(tmp_path) -> None:
    model = nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path)

    manager.save(model, step=1)
    latest = manager.save(model, step=2)

    assert manager.latest() == latest
