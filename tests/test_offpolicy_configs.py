"""Shipped off-policy configs must request the fixes they depend on.

The Fetch manipulation configs used to run SAC on plain uniform replay;
with sparse rewards that does not learn (FetchSlide: 0% success, Q diverged).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from srl.core.config import SACConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "envs"


@pytest.mark.parametrize(
    "name",
    [
        "fetch_push_sac.yaml",
        "fetch_slide_sac.yaml",
        "fetch_pick_and_place_sac.yaml",
        "fetch_reach_sac_her.yaml",
    ],
)
def test_fetch_configs_enable_her_and_normalizer(name: str) -> None:
    cfg = yaml.safe_load((CONFIGS / name).read_text())
    train = cfg["train"]
    assert cfg["env_type"] == "goal"
    assert train["use_her"] is True
    assert train["her_strategy"] == "future"
    assert train["her_ratio"] == pytest.approx(0.8)  # k = 4
    assert train["her_max_episode_len"] >= 50
    assert train["normalize_obs"] is True
    assert train.get("replay_n_step", 1) == 1  # n-step is not combined with HER
    fields = set(SACConfig.__dataclass_fields__)
    assert {k for k in train if k in ("normalize_obs", "obs_norm_clip")} <= fields
