# srl.core — agent contract

Every algorithm is a {class}`~srl.core.base_agent.BaseAgent`. On-policy
algorithms (PPO, A2C, A3C) derive from
{class}`~srl.core.base_agent.OnPolicyAgent`, off-policy ones (SAC, TD3, DDPG)
from {class}`~srl.core.base_agent.OffPolicyAgent`.

| Method | Level | Purpose |
|---|---|---|
| `from_config(yaml, env=None, device=...)` | public | build model + algorithm config from a YAML |
| `learn(total_timesteps, env=None, eval_env=None, callbacks=None, log_dir=None, progress=False)` | public | train |
| `predict(obs, state=None, deterministic=False)` | public | numpy observation in, `(action_in_env_units, state)` out |
| `save(dir)` / `Cls.load(dir, env=None)` | public | persistence (see {mod}`srl.core.agent_io`) |
| `state_dict()` / `load_state_dict()` | public | the in-memory checkpoint payload |
| `act(obs_tensors, hidden=None, deterministic=False)` | low level | tensor action in the model's `[-1, 1]` domain: `(action, log_prob, value, hidden)` |
| `update()` | low level | one gradient update from the agent's buffer |

Algorithms are registered by name in {data}`srl.algorithms.ALGORITHMS`
(`name -> (agent class, config class)`); that name is the YAML `algo:` key.

## BaseAgent

```{eval-rst}
.. autoclass:: srl.core.base_agent.BaseAgent
   :members: from_config, learn, predict, act, update, save, state_dict, load_state_dict,
      checkpoint_payload, load_checkpoint_payload, set_env, get_env, device
   :show-inheritance:
```

`load` works both as `Cls.load(path, env=None, device=None)` (returns a new
agent) and as `agent.load(path)` (restores in place).

## OnPolicyAgent

```{eval-rst}
.. autoclass:: srl.core.base_agent.OnPolicyAgent
   :members: learn, act, set_action_bounds, set_action_bounds_from_env, clip_action
   :show-inheritance:
```

## OffPolicyAgent

```{eval-rst}
.. autoclass:: srl.core.base_agent.OffPolicyAgent
   :members: learn, act
   :show-inheritance:
```

## Persistence

```{eval-rst}
.. automodule:: srl.core.agent_io
   :members: save_agent, load_agent, read_payload
```

## Algorithm registry

```{eval-rst}
.. autofunction:: srl.algorithms.get_algorithm
```
