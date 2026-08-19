"""Regression test for a GAE off-by-one in RolloutBuffer.compute_returns_
and_advantages, found while investigating why SRL's PPO peaks then declines
on JAVIS's mjlab balance task.

Storage convention (see rollout_buffer.py's own docstring, and
srl.cli.train._run_on_policy's add() call sequence): buffer slot t holds
obs=S_t (the state acted FROM), value=V(S_t), and done = whether *stepping*
S_t ended the episode (i.e. whether S_{t+1} is a fresh reset). So
self._dones[t] answers "is there a boundary between slot t and slot t+1".

The buggy code (before this fix) gated slot t's bootstrap on
self._dones[t + 1] instead of self._dones[t] -- correct for a reference
implementation (e.g. CleanRL) that stores dones the OPPOSITE way (dones[t]
= "is obs[t] itself fresh"), but never adjusted here for this class's own,
reversed convention.

This test hand-computes the correct GAE for a small, concrete 3-step,
1-env trajectory with a termination in the middle, and asserts the buffer
produces it -- rather than asserting against "whatever the old code
happened to output", which would just encode the bug into the test.

Trajectory (rewards/values chosen to make hand-computation easy):
  slot 0: obs=S0, value=V(S0)=1.0, reward=1.0, done=False (S1 not fresh)
  slot 1: obs=S1, value=V(S1)=2.0, reward=2.0, done=True  (S2 -- reset -- IS fresh)
  slot 2: obs=S0', value=V(S0')=5.0, reward=3.0, done=False (S3 not fresh)
  last_value (V of the state one step past the buffer) = 7.0

gamma=1.0, gae_lambda=1.0 (both neutral, so deltas add up in a way that's
easy to verify by hand without decay).

Correct (by hand, using standard GAE with proper done semantics):
  delta_2 = r2 + gamma*last_value*(1-done_2) - V(S0') = 3 + 1*7*1 - 5 = 5.0
  adv_2 = delta_2 = 5.0                                  (last_gae so far)
  delta_1 = r1 + gamma*V(S0')*(1-done_1) - V(S1) = 2 + 1*5*0 - 2 = 0.0
  adv_1 = delta_1 + gamma*lam*(1-done_1)*adv_2 = 0.0 + 1*1*0*5.0 = 0.0
                                                  (chain MUST break here --
                                                  done_1=True means nothing
                                                  from adv_2's episode, an
                                                  unrelated freshly-reset
                                                  one, should leak backward)
  delta_0 = r0 + gamma*V(S1)*(1-done_0) - V(S0) = 1 + 1*2*1 - 1 = 2.0
  adv_0 = delta_0 + gamma*lam*(1-done_0)*adv_1 = 2.0 + 1*1*1*0.0 = 2.0

returns = advantages + values -> [3.0, 2.0, 10.0]

The pre-fix code would instead have used self._dones[t+1] to gate slot t,
producing adv_0 = 1.0 (WRONGLY zeroing a bootstrap that should have
happened -- nothing terminal occurs between S0 and S1) and adv_1 = 2 + 5 - 2
= 5.0 (WRONGLY bootstrapping straight through the real termination, using
V(S0') -- a completely unrelated fresh episode's value -- as if the
original episode had continued).
"""

from __future__ import annotations

import numpy as np

from srl.core.rollout_buffer import RolloutBuffer


def _make_filled_buffer() -> RolloutBuffer:
    buf = RolloutBuffer(n_steps=3, n_envs=1, gamma=1.0, gae_lambda=1.0)
    buf.add(obs={"state": np.array([[0.0]])}, action=np.array([[0.0]]), reward=np.array([1.0]),
            done=np.array([False]), value=np.array([1.0]))
    buf.add(obs={"state": np.array([[1.0]])}, action=np.array([[0.0]]), reward=np.array([2.0]),
            done=np.array([True]), value=np.array([2.0]))
    buf.add(obs={"state": np.array([[0.0]])}, action=np.array([[0.0]]), reward=np.array([3.0]),
            done=np.array([False]), value=np.array([5.0]))
    return buf


def test_gae_breaks_chain_at_the_correct_step_not_shifted_by_one():
    buf = _make_filled_buffer()
    buf.compute_returns_and_advantages(last_value=np.array([7.0]))

    expected_advantages = np.array([2.0, 0.0, 5.0], dtype=np.float32)
    expected_returns = np.array([3.0, 2.0, 10.0], dtype=np.float32)

    np.testing.assert_allclose(buf.advantages[:, 0], expected_advantages, atol=1e-5)
    np.testing.assert_allclose(buf.returns[:, 0], expected_returns, atol=1e-5)


def test_gae_boundary_uses_the_buffers_own_last_done_without_needing_last_dones():
    """The final slot's own `done` (True) must gate the bootstrap against
    `last_value` too -- this used to require a separately-passed
    `last_dones` that no real caller ever supplied (confirmed: neither
    `srl.cli.train._run_on_policy` nor `srl.algorithms.a3c` pass it), so it
    silently always defaulted to "not done" even when the buffer's own
    last slot said otherwise.
    """
    buf = RolloutBuffer(n_steps=2, n_envs=1, gamma=1.0, gae_lambda=1.0)
    buf.add(obs={"state": np.array([[0.0]])}, action=np.array([[0.0]]), reward=np.array([1.0]),
            done=np.array([False]), value=np.array([1.0]))
    buf.add(obs={"state": np.array([[1.0]])}, action=np.array([[0.0]]), reward=np.array([2.0]),
            done=np.array([True]), value=np.array([2.0]))

    # last_value would (wrongly, pre-fix) leak into slot 1's advantage if the
    # boundary didn't respect slot 1's own done=True.
    buf.compute_returns_and_advantages(last_value=np.array([100.0]))

    # delta_1 = r1 + gamma*last_value*(1-done_1) - V(S1) = 2 + 0 - 2 = 0.0
    assert buf.advantages[1, 0] == 0.0
    assert buf.returns[1, 0] == 2.0
