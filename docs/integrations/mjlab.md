# mjlab Integration

[mjlab](https://github.com/mujocolab/mjlab) is a GPU-batched robot learning stack built on MuJoCo-Warp -- the same "one process, thousands of envs on one GPU" shape as Isaac Lab, without the multi-GB Omniverse/Isaac Sim runtime. SRL integrates it through the same `IsaacLabWrapper` used for real Isaac Lab (both envs expose a close enough API -- `reset()`/`step()`/`num_envs`/`device` -- that no separate wrapper class was needed).

Reach for this instead of [Isaac Lab](isaaclab.md) when you don't need Isaac Sim's rendering/asset pipeline and want a lighter install.

## Setup

```bash
pip install mjlab
pip install git+https://github.com/Bigkatoan/SRL.git
```

No app bootstrap, no `OMNI_KIT_ACCEPT_EULA`, no separate runtime interpreter -- mjlab is a normal Python package.

## Task registration

mjlab resolves tasks by string id through its own registry (`mjlab.tasks.registry`), populated by auto-importing every package registered under the `mjlab.tasks` entry-point group -- the same mechanism `import mjlab` itself runs at import time. Any project that registers its own tasks this way in its `pyproject.toml`:

```toml
[project.entry-points."mjlab.tasks"]
your_pkg = "your_pkg.tasks"
```

is automatically discoverable here with no extra wiring, as long as it's installed in the same environment as `mjlab`/`srl-rl`.

## Usage

`--env mjlab:<task>` (or `env_type: mjlab` in the YAML, which prefixes the id automatically -- see `_normalize_env_name` in `srl/cli/train.py`):

```yaml
# configs/envs/mjlab_example_ppo.yaml
env_id:   Your-Task-Id
env_type: mjlab
algo:     ppo

encoders:
  - name: actor_state_enc
    type: mlp
    input_name: actor   # mjlab's obs group name -- check with the task's
                         # own ObservationManager printout at env build time,
                         # or your task's observation config ("actor"/"critic"
                         # are common, but not guaranteed)
    input_dim: <your obs dim>
    latent_dim: 256
    layers:
      - {out_features: 256, activation: elu, norm: none}
      - {out_features: 256, activation: elu, norm: none}
  - name: critic_state_enc
    type: mlp
    input_name: actor
    input_dim: <your obs dim>
    latent_dim: 256
    layers:
      - {out_features: 256, activation: elu, norm: none}
      - {out_features: 256, activation: elu, norm: none}

flows:
  - "actor_state_enc -> actor"
  - "critic_state_enc -> critic"

actor: {name: actor, type: gaussian, action_dim: <your action dim>, log_std_init: -1.0}
critic: {name: critic, type: value}

losses:
  - {name: policy,  weight: 1.0}
  - {name: value,   weight: 1.0}
  - {name: entropy, weight: 0.005}

train:
  total_steps: 20_000_000
  n_envs: 4096          # mjlab-sized: GPU-batched inside ONE process, not
                         # N OS processes -- this is not the same knob as a
                         # SyncVectorEnv/AsyncVectorEnv n_envs
  n_steps: 24
  batch_size: 16384
  n_epochs: 5
  lr: 1e-3
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.005
  vf_coef: 1.0
  max_grad_norm: 1.0
```

```bash
srl-train --config configs/envs/mjlab_example_ppo.yaml --env mjlab:Your-Task-Id --device cuda
```

Off-policy (SAC/TD3/DDPG) works the same way -- see [Isaac Lab's SAC section](isaaclab.md#sac--async--gpu-buffer) for the async/GPU-buffer knobs, which apply identically here.

## Why a single encoder or explicit `input_name` both work now

Earlier versions had a real bug here: `srl/cli/train.py`'s rollout loop remaps the obs dict to encoder names *before* calling `agent.predict()`, and `AgentModel.forward()` remapped it *again* internally -- the second pass looked for the *original* raw obs key, which no longer existed once the first pass had already renamed it, raising `KeyError` for any config using explicit `input_name`. `AgentModel._remap_obs_dict` now short-circuits when the incoming dict's keys already exactly match the encoder names, so remapping only ever happens once regardless of how many places call into it. Two other isaaclab/mjlab-specific gaps were fixed alongside it:

- `_evaluate_agent` unconditionally squeezed the batch dim off actions for single-env eval, which broke isaaclab/mjlab envs (they always expect a batched `(1, action_dim)` action even at `num_envs=1`).
- Off-policy random-action warmup called `.sample()` on the action space directly, which isaaclab/mjlab's lightweight `Box`-style space dataclass doesn't implement, and (once worked around) stacked N samples of the already-*batched* `action_space` instead of `single_action_space`, producing the wrong shape. `IsaacLabWrapper` now also exposes `single_act_space` for exactly this.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `KeyError: <task id>` in `load_env_cfg` | Task package not installed / entry point not registered | Confirm `python -c "import mjlab; from mjlab.tasks.registry import list_tasks; print(list_tasks())"` lists your task id |
| Observation routing `KeyError` | `input_name` doesn't match the task's obs group | Check the `ObservationManager` table mjlab prints at env construction time |
| `IndexError`/`ValueError` on action shape during eval or warmup | Running an SRL build older than this fix | Update `srl-rl` |

## See also

- [Isaac Lab Integration](isaaclab.md)
- [GPU Replay Buffer](../training/buffers.md)
- [Runners & Training Loop](../training/runners.md)
