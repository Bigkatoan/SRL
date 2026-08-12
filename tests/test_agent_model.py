"""AgentModel.forward()'s actor_action path.

Regression test for a bug found while auditing SRL for its first stable
release: PPO/A2C/A3C's update() always called the actor head's plain
forward(), which draws a FRESH rsample() every call. That silently broke the
PPO importance-sampling ratio (new_log_prob - old_log_prob is only
meaningful when both refer to the SAME action) and made A2C's policy
gradient an estimator of the wrong quantity (d/dtheta log pi(a'|s) * A(s,a)
for an unrelated a' != a). The actor heads already had a correctly-written
evaluate_actions(z, action) method for exactly this purpose -- it was just
never wired into AgentModel.forward(). See agent_model.py's `actor_action`
parameter and its docstring for the fix.
"""

from __future__ import annotations

import torch

from srl.registry.builder import ModelBuilder


def _build_gaussian_model():
    return ModelBuilder.from_dict(
        {
            "encoders": [{"name": "enc", "type": "mlp", "input_dim": 4, "latent_dim": 8}],
            "flows": ["enc -> actor", "enc -> critic"],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 2},
            "critic": {"name": "critic", "type": "value"},
        }
    )


def test_actor_action_reevaluates_the_given_action_not_a_fresh_sample() -> None:
    model = _build_gaussian_model()
    obs = {"enc": torch.randn(5, 4)}
    fixed_action = torch.randn(5, 2)

    result = model(obs, actor_action=fixed_action)
    actor_out = result["actor_out"]

    assert torch.equal(actor_out["action"], fixed_action)

    enc_latent = model.encoders["enc"](obs["enc"])
    expected_log_prob, expected_entropy = model.actor.evaluate_actions(enc_latent, fixed_action)
    assert torch.allclose(actor_out["log_prob"], expected_log_prob)
    assert torch.allclose(actor_out["entropy"], expected_entropy)


def test_actor_action_log_prob_is_deterministic_across_repeated_calls() -> None:
    """The whole point of the fix: re-evaluating the same (obs, action) pair
    must be deterministic, unlike a fresh rsample() each call."""
    model = _build_gaussian_model()
    obs = {"enc": torch.randn(3, 4)}
    fixed_action = torch.randn(3, 2)

    with torch.no_grad():
        log_prob_1 = model(obs, actor_action=fixed_action)["actor_out"]["log_prob"]
        log_prob_2 = model(obs, actor_action=fixed_action)["actor_out"]["log_prob"]

    assert torch.equal(log_prob_1, log_prob_2)


def test_without_actor_action_forward_still_samples_fresh_action() -> None:
    """Backward compatibility: rollout collection (predict()) must keep
    calling plain forward() and get a genuinely sampled action, not a
    deterministic re-evaluation."""
    model = _build_gaussian_model()
    obs = {"enc": torch.randn(3, 4)}

    torch.manual_seed(0)
    out1 = model(obs)["actor_out"]
    out2 = model(obs)["actor_out"]

    assert "dist" in out1
    assert not torch.equal(out1["action"], out2["action"])


def test_ppo_value_head_critic_unaffected_by_actor_action() -> None:
    """The critic's own `action` kwarg (Q-function heads) is a distinct
    parameter from `actor_action` -- passing actor_action must not touch
    ValueHead's forward(), which only accepts (z,)."""
    model = _build_gaussian_model()
    obs = {"enc": torch.randn(4, 4)}
    fixed_action = torch.randn(4, 2)

    result = model(obs, actor_action=fixed_action)
    assert result["value"].shape == (4,)
