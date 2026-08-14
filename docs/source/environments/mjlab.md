# mjlab Environments

[mjlab](https://github.com/mujocolab/mjlab) is a GPU-batched robot learning stack
built on MuJoCo-Warp -- thousands of environments stepped in parallel inside one
process, on one GPU. SRL integrates it through the same `IsaacLabWrapper` used for
real [Isaac Lab](isaaclab.md) (both expose a close enough API -- `reset()`/`step()`/
`num_envs`/`device` -- that no separate wrapper class was needed), without the
multi-GB Omniverse/Isaac Sim runtime Isaac Lab requires.

For the install steps, task-registration mechanism, and the full YAML/CLI usage
pattern, see [mjlab Integration](../integrations/mjlab.md) -- this page is the
environments-suite companion to that one: what to expect as a *supported
environment family*, not how the integration is wired.

---

## Requirements

- NVIDIA GPU (CUDA)
- `mjlab` (`pip install mjlab`)
- Python 3.10 or 3.11

No Isaac Sim, no `OMNI_KIT_ACCEPT_EULA`, no separate runtime interpreter -- mjlab is
a normal Python package, installed into the same environment as `srl-rl`.

---

## Supported environments

Unlike the other environment suites on this page, SRL does not ship any built-in
mjlab task configs -- mjlab tasks are always **project-specific**, registered by
whatever robot/task package you (or a project you depend on) publish via the
`mjlab.tasks` entry point (see {ref}`Task registration <mjlab-task-registration>`
for the exact mechanism). SRL's job is just to train against whatever task id that
package registers, the same way it would train against any other environment
family.

The best real-world reference is [JAVIS](https://github.com/Bigkatoan/JAVIS), a
2-wheel rover project that registers and trains a real mjlab task
(`Javis-Payload-Rough`) through this exact path -- see
{ref}`the JAVIS walkthrough <mjlab-javis-example>` for a complete, runnable
example including environment setup.

| Env | Source | obs | act | n_envs | Notes |
|---|---|---|---|---|---|
| `Javis-Payload-Rough` | [JAVIS](https://github.com/Bigkatoan/JAVIS) | 384 | 2 | 512–4096 | Balance under randomized payload/CoM offset, rough terrain |

---

## Training

```bash
pip install mjlab
pip install git+https://github.com/Bigkatoan/SRL.git

srl-train --config configs/envs/your_mjlab_task_ppo.yaml \
          --env mjlab:Your-Task-Id \
          --algo ppo \
          --n-envs 4096 \
          --device cuda
```

`env_type: mjlab` in the YAML config prefixes the id automatically, so
`--env Your-Task-Id` and `--env mjlab:Your-Task-Id` are equivalent once the config
declares `env_type: mjlab` -- see {ref}`Usage <mjlab-usage>` for a full config
example.

Off-policy (SAC/TD3/DDPG) works the same way as Isaac Lab -- see
{ref}`the SAC async/GPU-buffer section <sac-async-gpu-buffer>`, which applies
identically here.

---

## Watching training live

```bash
srl-train --config configs/envs/your_mjlab_task_ppo.yaml \
          --env mjlab:Your-Task-Id --device cuda --visualize
```

mjlab is the one GPU-batched env family where `--visualize` is actually supported
(each env instance owns independent MuJoCo state, unlike Isaac Lab's single shared
render context) -- see {ref}`the CLI reference <live-viewer-visualize>` for the full
behavior.

---

## MjlabWrapper

There isn't one -- mjlab reuses `IsaacLabWrapper` directly:

```python
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

# construction goes through srl.cli.train's mjlab branch in practice
# (mjlab.tasks.registry.load_env_cfg + ManagerBasedRlEnv), shown here
# conceptually:
env = IsaacLabWrapper(mjlab_env)
obs, _ = env.reset()
# obs = {"actor": tensor(4096, 384), ...}   ← per-obs-group dict, GPU tensors
#                                              converted to numpy

action = env.single_act_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `KeyError: <task id>` in `load_env_cfg` | Task package not installed / entry point not registered | Confirm `python -c "import mjlab; from mjlab.tasks.registry import list_tasks; print(list_tasks())"` lists your task id |
| Observation routing `KeyError` | `input_name` doesn't match the task's obs group | Check the `ObservationManager` table mjlab prints at env construction time |
| `IndexError`/`ValueError` on action shape during eval or warmup | Running an SRL build older than the isaaclab/mjlab action-shape fixes | Update `srl-rl` |
| `--visualize` crashes training with a CUDA graph capture error | Periodic eval and the visualizer both stepping the GPU concurrently -- fixed by auto-disabling periodic eval when `--visualize` starts for mjlab | Update `srl-rl`; if it still happens, check the console for the "disabling periodic evaluation" message to confirm the fix is active |

---

## See also

- [mjlab Integration](../integrations/mjlab.md) -- install, task registration, full config example, the JAVIS walkthrough
- [Isaac Lab Environments](isaaclab.md)
- [GPU Replay Buffer](../gpu_replay_buffer.md)
- [Runners & Training Loop](../training/runners.md)
