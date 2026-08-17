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


# ---------------------------------------------------------------------------
# compute_actor / compute_critic selective forward (SAC hot-path fix)
#
# JAVIS's mjlab balance-policy task uses an asymmetric actor-critic: a
# 384-dim "actor" observation group and a separate, privileged 624-dim
# "critic" observation group, each routed to its own encoder. SAC's
# update() calls model()/target_model() five times per gradient step, but
# several of those calls only ever read one of actor_out/value from the
# result -- before this fix, AgentModel.forward() unconditionally ran BOTH
# heads (and every encoder feeding either of them) on every call, so those
# calls paid for a whole extra encoder + head forward pass (Q-function heads
# even fabricate a dummy zero action to run on) that was immediately
# discarded. See agent_model.py's `compute_actor`/`compute_critic` params.
# ---------------------------------------------------------------------------


def _build_asymmetric_sac_model():
    """Mirrors JAVIS's shape: separate actor/critic encoders, twin-Q critic."""
    return ModelBuilder.from_dict(
        {
            "encoders": [
                {"name": "actor_enc", "type": "mlp", "input_dim": 6, "latent_dim": 8},
                {"name": "critic_enc", "type": "mlp", "input_dim": 10, "latent_dim": 12},
            ],
            "flows": ["actor_enc -> actor", "critic_enc -> critic"],
            "actor": {"name": "actor", "type": "squashed_gaussian", "action_dim": 3},
            "critic": {"name": "critic", "type": "twin_q", "action_dim": 3},
        }
    )


def test_compute_critic_false_skips_critic_encoder_and_matches_full_forward() -> None:
    model = _build_asymmetric_sac_model()
    obs = {"actor_enc": torch.randn(5, 6), "critic_enc": torch.randn(5, 10)}

    torch.manual_seed(42)
    full = model(obs)
    torch.manual_seed(42)
    actor_only = model(obs, compute_critic=False)

    assert actor_only["value"] is None
    assert torch.equal(actor_only["actor_out"]["action"], full["actor_out"]["action"])
    assert torch.equal(actor_only["actor_out"]["log_prob"], full["actor_out"]["log_prob"])
    # The critic-only encoder must not appear in latents at all -- it was
    # never run, not merely run-and-ignored.
    assert "critic_enc" not in actor_only["latents"]
    assert "actor_enc" in actor_only["latents"]


def test_compute_actor_false_skips_actor_encoder_and_matches_full_forward() -> None:
    model = _build_asymmetric_sac_model()
    obs = {"actor_enc": torch.randn(5, 6), "critic_enc": torch.randn(5, 10)}
    action = torch.randn(5, 3)

    full = model(obs, action=action)
    critic_only = model(obs, action=action, compute_actor=False)

    assert critic_only["actor_out"] is None
    q1_full, q2_full = full["value"]
    q1_only, q2_only = critic_only["value"]
    assert torch.equal(q1_only, q1_full)
    assert torch.equal(q2_only, q2_full)
    assert "actor_enc" not in critic_only["latents"]
    assert "critic_enc" in critic_only["latents"]


def test_compute_actor_and_compute_critic_both_false_raises() -> None:
    model = _build_asymmetric_sac_model()
    obs = {"actor_enc": torch.randn(2, 6), "critic_enc": torch.randn(2, 10)}
    try:
        model(obs, compute_actor=False, compute_critic=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both compute_actor/critic are False")


def test_compute_flags_default_to_true_and_leave_shared_encoder_mode_unaffected() -> None:
    """A single shared encoder implicitly feeding both heads (no explicit
    flow-graph routing -- the common non-asymmetric case) must still be run
    in full even when only one head is requested: `encoder_names_for_head`
    can't isolate a shared encoder to one head, so it conservatively reports
    it as needed for both, and nothing should be skipped or break."""
    model = ModelBuilder.from_dict(
        {
            "encoders": [{"name": "enc", "type": "mlp", "input_dim": 4, "latent_dim": 8}],
            "flows": ["enc -> actor", "enc -> critic"],
            "actor": {"name": "actor", "type": "squashed_gaussian", "action_dim": 2},
            "critic": {"name": "critic", "type": "twin_q", "action_dim": 2},
        }
    )
    obs = {"enc": torch.randn(3, 4)}
    action = torch.randn(3, 2)

    result = model(obs, action=action, compute_actor=False)
    assert "enc" in result["latents"]
    assert result["value"] is not None
