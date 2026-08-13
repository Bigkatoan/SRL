"""HindsightReplayBuffer — HER (Andrychowicz et al., 2017).

Stores full episodes and re-labels goals using 'future', 'final',
'episode', or 'random' strategies.  Works as a drop-in replacement for
ReplayBuffer for goal-conditioned tasks.

Observation dict keys
---------------------
obs     : actual observation
achieved_goal (ag) : goal achieved at this timestep
desired_goal  (dg) : goal the agent is trying to reach

The reward function is provided externally via *reward_fn(achieved, desired, info)*.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from srl.core.replay_buffer import ReplayBatch


class HERReplayBuffer:
    """Episode-based buffer with Hindsight Experience Replay.

    Parameters
    ----------
    capacity:
        Maximum number of *transitions* (not episodes) to store.
    obs_dim, goal_dim, action_dim:
        Dimensions of observation, goal, and action vectors.
    reward_fn:
        ``reward_fn(achieved_goal, desired_goal, info) -> float``
    strategy:
        Goal relabelling strategy: ``'future'`` (recommended), ``'final'``,
        ``'episode'``, or ``'random'``.
    her_ratio:
        Fraction of sampled transitions that get HER relabelling.  0.8 means
        80% relabelled, 20% from original buffer.
    max_episode_len:
        Maximum steps per episode (for pre-allocation).
    obs_key:
        Dict key under which :meth:`sample` returns the observation tensors.
        SRL models take a ``dict[str, Tensor]`` keyed by encoder input name,
        so the batch must be wrapped the same way ``ReplayBuffer`` wraps
        its own -- defaults to ``"state"``, matching
        :class:`~srl.envs.goal_env_wrapper.GoalEnvWrapper`'s own ``obs_key``.

    Observation layout
    ------------------
    ``obs`` passed to :meth:`add_transition` must be the observation *without*
    the desired goal appended; :meth:`sample` appends the (possibly relabelled)
    desired goal itself.  Wiring this to ``GoalEnvWrapper``, whose flat obs is
    ``[observation | achieved_goal | desired_goal]``, therefore means storing
    ``[observation | achieved_goal]`` as ``obs`` (``obs_dim = observation +
    achieved_goal``) and ``desired_goal`` as the goal -- the concatenation in
    :meth:`sample` then reproduces exactly the layout the model was built for.
    """

    #: Goal relabelling strategies understood by :meth:`sample`.
    STRATEGIES = ("future", "final", "episode", "random")

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        reward_fn: Callable,
        strategy: str = "future",
        her_ratio: float = 0.8,
        max_episode_len: int = 1000,
        gamma: float = 0.99,
        device: str | torch.device = "cpu",
        obs_key: str = "state",
    ) -> None:
        if strategy not in self.STRATEGIES:
            # sample() dispatches on `strategy` with a plain if/elif chain and
            # no else, so an unrecognised value silently leaves every goal
            # un-relabelled -- i.e. HER quietly degrades to a plain replay
            # buffer instead of failing. Reject it at construction instead.
            raise ValueError(
                f"unknown HER strategy {strategy!r}; expected one of {list(self.STRATEGIES)}"
            )
        if not 0.0 <= her_ratio <= 1.0:
            raise ValueError(f"her_ratio must be in [0, 1], got {her_ratio}")
        self.capacity = capacity
        self.obs_key = obs_key
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.reward_fn = reward_fn
        self.strategy = strategy
        self.her_ratio = her_ratio
        self.max_episode_len = max_episode_len
        self.gamma = gamma
        self.device = torch.device(device)

        max_episodes = capacity // max_episode_len + 1
        self._max_ep = max_episodes

        # Episode storage arrays
        self._obs = np.zeros((max_episodes, max_episode_len + 1, obs_dim), dtype=np.float32)
        self._ag = np.zeros((max_episodes, max_episode_len + 1, goal_dim), dtype=np.float32)
        self._dg = np.zeros((max_episodes, max_episode_len, goal_dim), dtype=np.float32)
        self._actions = np.zeros((max_episodes, max_episode_len, action_dim), dtype=np.float32)
        # Real per-timestep termination, against the ORIGINAL desired_goal --
        # kept separate from the HER-relabelled reward-based "done" computed
        # in sample(). Needed for the (1 - her_ratio) fraction of a batch that
        # is NOT goal-relabelled: without it, sample() had no record of
        # whether those transitions truly ended the episode, and fabricated a
        # `done` from `reward == 0.0` for every sampled transition regardless,
        # discarding real termination info even for the un-relabelled slice.
        self._done = np.zeros((max_episodes, max_episode_len), dtype=np.float32)
        self._ep_len = np.zeros(max_episodes, dtype=np.int32)

        self._ep_ptr = 0
        self._n_stored = 0
        self._current_ep: list = []

    # ------------------------------------------------------------------
    # Episode writing
    # ------------------------------------------------------------------

    def add_transition(
        self,
        obs: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        next_achieved_goal: np.ndarray,
        done: bool,
        truncated: bool = False,
    ) -> None:
        """Append one transition to the in-progress episode.

        ``done`` is the *real* termination flag and is what gets stored per
        timestep (see ``_done``); ``truncated`` only closes the episode.
        Both are needed because a time-limited goal env -- every Fetch task,
        for instance -- ends by truncation with ``terminated`` always False, so
        keying the episode commit off ``done`` alone would never flush an
        episode, and transitions from many separate rollouts would be
        concatenated into one bogus "episode" that HER then relabels across.
        Storing ``truncated`` as ``done`` instead would be the opposite error:
        it fabricates terminal states at the time limit and biases the
        bootstrap target for the non-relabelled fraction of a batch.
        """
        self._current_ep.append(
            (obs, achieved_goal, desired_goal, action, next_obs, next_achieved_goal, done)
        )
        if done or truncated or len(self._current_ep) >= self.max_episode_len:
            self._commit_episode()

    def _commit_episode(self) -> None:
        ep = self._current_ep
        T = len(ep)
        idx = self._ep_ptr

        for t, (o, ag, dg, a, _next_o, _next_ag, d) in enumerate(ep):
            self._obs[idx, t] = o
            self._ag[idx, t] = ag
            self._dg[idx, t] = dg
            self._actions[idx, t] = a
            self._done[idx, t] = float(d)
        # store last obs/ag
        last_no, last_nag = ep[-1][4], ep[-1][5]
        self._obs[idx, T] = last_no
        self._ag[idx, T] = last_nag
        self._ep_len[idx] = T

        self._ep_ptr = (self._ep_ptr + 1) % self._max_ep
        self._n_stored = min(self._n_stored + 1, self._max_ep)
        self._current_ep = []

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> ReplayBatch:
        ep_idx = np.random.randint(0, self._n_stored, size=batch_size)
        t_idx = np.array([np.random.randint(0, self._ep_len[e]) for e in ep_idx], dtype=np.int32)

        n_her = int(batch_size * self.her_ratio)
        her_mask = np.zeros(batch_size, dtype=bool)
        her_mask[:n_her] = True
        np.random.shuffle(her_mask)

        obs = self._obs[ep_idx, t_idx]
        next_obs = self._obs[ep_idx, t_idx + 1]
        next_ag = self._ag[ep_idx, t_idx + 1]
        dg = self._dg[ep_idx, t_idx].copy()
        actions = self._actions[ep_idx, t_idx]

        # HER relabelling
        for i in np.where(her_mask)[0]:
            ep_len = int(self._ep_len[ep_idx[i]])
            if self.strategy == "future":
                future_t = np.random.randint(t_idx[i], ep_len + 1)
                dg[i] = self._ag[ep_idx[i], future_t]
            elif self.strategy == "final":
                dg[i] = self._ag[ep_idx[i], ep_len]
            elif self.strategy == "episode":
                rand_t = np.random.randint(0, ep_len + 1)
                dg[i] = self._ag[ep_idx[i], rand_t]
            elif self.strategy == "random":
                rand_ep = np.random.randint(0, self._n_stored)
                rand_t = np.random.randint(0, self._ep_len[rand_ep] + 1)
                dg[i] = self._ag[rand_ep, rand_t]

        # Recompute rewards with relabelled goals
        rewards = np.array(
            [self.reward_fn(next_ag[i], dg[i], {}) for i in range(batch_size)],
            dtype=np.float32,
        )
        # `done` per transition: for the her_mask fraction (goal relabelled),
        # standard HER practice is right -- termination follows the NEW
        # (relabelled) goal, i.e. "reward 0 against the relabelled goal" IS
        # the terminal condition for that synthetic trajectory. For the rest
        # (still using the true original desired_goal), termination must
        # come from what the episode actually did -- the real stored `done`
        # -- not this reward heuristic, which would silently discard genuine
        # failure/timeout terminations any time the reward isn't purely
        # "0 iff goal achieved" (i.e. anything beyond the simplest sparse
        # case), and even in that case is redundant with information already
        # recorded at collection time.
        reward_done = (rewards == 0.0).astype(np.float32)  # sparse: reward 0 = achieved
        real_done = self._done[ep_idx, t_idx]
        dones = np.where(her_mask, reward_done, real_done).astype(np.float32)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        full_obs = np.concatenate([obs, dg], axis=-1)
        full_next_obs = np.concatenate([next_obs, dg], axis=-1)

        # Wrapped as a dict, exactly like ReplayBuffer.sample(): SRL models
        # take dict observations keyed by encoder input name, and consumers
        # (e.g. SAC.update) iterate `batch.obs.items()`. Returning the bare
        # concatenated tensor here made this buffer unusable with any real
        # agent.
        return ReplayBatch(
            observations={self.obs_key: _t(full_obs)},
            actions=_t(actions),
            rewards=_t(rewards),
            next_observations={self.obs_key: _t(full_next_obs)},
            dones=_t(dones),
        )

    def __len__(self) -> int:
        """Number of stored *episodes* (this buffer is episode-indexed)."""
        return self._n_stored

    @property
    def num_transitions(self) -> int:
        """Total stored transitions across all episodes."""
        return int(self._ep_len[: self._n_stored].sum())

    def can_sample(self, batch_size: int) -> bool:
        """Whether :meth:`sample` can serve a batch of *batch_size*.

        ``__len__`` counts episodes, so the usual off-policy readiness check
        ``len(buffer) >= batch_size`` means something different here than it
        does for ``ReplayBuffer`` (where it counts transitions). Agents should
        prefer this method so the two buffers gate updates equivalently.
        """
        return self._n_stored > 0 and self.num_transitions >= batch_size

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialisable buffer state.

        ``reward_fn`` is deliberately excluded -- it is a live callable bound
        to the environment (``env.unwrapped.compute_reward``) and is
        re-attached by whoever rebuilds the buffer.
        """
        return {
            "obs": self._obs,
            "ag": self._ag,
            "dg": self._dg,
            "actions": self._actions,
            "done": self._done,
            "ep_len": self._ep_len,
            "ep_ptr": self._ep_ptr,
            "n_stored": self._n_stored,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self._obs = np.asarray(state["obs"], dtype=np.float32)
        self._ag = np.asarray(state["ag"], dtype=np.float32)
        self._dg = np.asarray(state["dg"], dtype=np.float32)
        self._actions = np.asarray(state["actions"], dtype=np.float32)
        self._done = np.asarray(state["done"], dtype=np.float32)
        self._ep_len = np.asarray(state["ep_len"], dtype=np.int32)
        self._ep_ptr = int(state["ep_ptr"])
        self._n_stored = int(state["n_stored"])
        self._current_ep = []
