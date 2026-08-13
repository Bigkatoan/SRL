"""Regression test for the 'q' critic head type crash.

srl/registry/builder.py's dispatch condition, srl/cli/train.py's
DDPG-compatible-heads check, and config_schema.py's type comment all treat
"q" as a valid critic head type alongside "value"/"twin_q"/"q_function" --
but `_CRITIC_HEADS` never actually registered it, so `critic: {type: q}`
passed every validation check and then crashed at build time with
`ValueError: Unknown critic head type 'q'`.
"""

from __future__ import annotations

from srl.networks.heads.critic_head import QFunctionHead, build_critic_head


def test_q_alias_builds_a_q_function_head() -> None:
    head = build_critic_head(
        head_type="q",
        input_dim=8,
        layer_configs=[{"out_features": 32, "activation": "relu"}],
        action_dim=2,
    )
    assert isinstance(head, QFunctionHead)


def test_q_and_q_function_produce_the_same_head_type() -> None:
    layer_configs = [{"out_features": 32, "activation": "relu"}]
    q_head = build_critic_head(
        head_type="q", input_dim=8, layer_configs=layer_configs, action_dim=2
    )
    q_function_head = build_critic_head(
        head_type="q_function", input_dim=8, layer_configs=layer_configs, action_dim=2
    )
    assert type(q_head) is type(q_function_head)
