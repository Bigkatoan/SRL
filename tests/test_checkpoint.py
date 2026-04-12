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


def test_checkpoint_manager_latest_returns_recent_path(tmp_path) -> None:
    model = nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path)

    manager.save(model, step=1)
    latest = manager.save(model, step=2)

    assert manager.latest() == latest