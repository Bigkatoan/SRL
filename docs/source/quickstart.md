# Quick Start

This page walks you through the full workflow in ~30 lines of code.

---

## 1. Install

```bash
git clone https://github.com/Bigkatoan/SRL.git
cd SRL
pip install -e ".[mujoco]"
pip install "srl-rl[mujoco] @ git+https://github.com/Bigkatoan/SRL.git"
```

This walkthrough uses `configs/envs/halfcheetah_sac.yaml`, one of the example
configs shipped in the repo — a bare `pip install` from the git URL (no
clone) does not include `configs/`. See [Installation](installation.md) if
you only need the Python package.

---

## 2. Pick a config

SRL uses YAML as the primary control plane for model structure. The config file is not just a convenience layer; it is the main way you declare encoders, graph connectivity, heads, and the training knobs currently supported by the CLI.

Read the [YAML Core Guide](yaml_core.md) if you want the full mental model. The example below shows the basic shape:

```yaml
# configs/envs/halfcheetah_sac.yaml
algo: sac
encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 256, activation: relu, norm: layer_norm}
flows:
  - "state_enc -> actor"
  - "state_enc -> critic"
actor:
  name: actor
  type: squashed_gaussian
  action_dim: 6
critic:
  name: critic
  type: twin_q
  action_dim: 6
```

---

## 3. Train with the CLI

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml \
          --env HalfCheetah-v5 \
          --algo sac \
          --steps 1000000 \
          --log-interval 5000 \
          --episode-window 25 \
          --plot-metrics train/score_mean,sac/critic_loss
```

The CLI prints compact training summaries to the terminal and writes `summary.json`, `history.csv`, `metrics.jsonl`, and `training_curves.svg` into the selected run directory. Use `--no-plots`, `--plot-metrics`, `--log-interval`, `--episode-window`, and `--console-metrics` to customize the output.

The important design point is that the CLI is consuming the same declarative YAML that `ModelBuilder` consumes. In other words, the config file is the core representation shared by training, visualization, and reproducible experiments.

---

## 4. Train via Python API

The same YAML drives the Python API. `from_config` builds the model with
`ModelBuilder`, reads the algorithm hyperparameters from the `train:` block,
and wraps a raw Gymnasium env with SRL's wrapper plus action rescaling
(the policy acts in `[-1, 1]`, the env sees its own bounds):

```python
import gymnasium as gym

from srl import SAC

agent = SAC.from_config("configs/envs/pendulum_sac.yaml", env=gym.make("Pendulum-v1"))
agent.learn(20_000, eval_env=gym.make("Pendulum-v1"), progress=True)

env = gym.make("Pendulum-v1")
obs, _ = env.reset(seed=0)
action, _ = agent.predict(obs, deterministic=True)   # numpy in, env-unit action out
obs, reward, terminated, truncated, info = env.step(action)

agent.save("runs/pendulum")                           # a directory
agent2 = SAC.load("runs/pendulum", env=gym.make("Pendulum-v1"))
```

- `predict` takes one observation or a batch (numpy array, or the dict an SRL
  wrapper returns) and returns `(action, state)`; `state` is the recurrent
  state (`None` for feed-forward policies).
- Every algorithm (`PPO`, `A2C`, `A3C`, `SAC`, `TD3`, `DDPG`) has the same
  `from_config` / `learn` / `predict` / `save` / `load` API.
  `srl.BaseAgent.from_config(path)` picks the class from the YAML's `algo:` key.
- With `env=None`, `from_config` builds the YAML's `env_id` with
  `train.n_envs` vectorized copies, exactly like `srl-train`.
- You can also construct agents directly:
  `SAC(ModelBuilder.from_yaml(path), SACConfig(...), env=gym.make("Pendulum-v1"))`
  (the target network is created for you).

### Callbacks

```python
from srl.training import CheckpointCallback, EarlyStopping, EvalCallback

agent.learn(
    200_000,
    callbacks=[
        EvalCallback(gym.make("Pendulum-v1"), eval_freq=5_000, best_model_save_path="runs/best"),
        EarlyStopping(monitor="eval/score_mean", patience=5),
        CheckpointCallback("checkpoints/pendulum", save_interval=20_000),
    ],
    log_dir="runs/pendulum",   # TensorBoard + metrics.jsonl
)
```

Callbacks receive `on_training_start`, `on_step`, `on_rollout_end`
(on-policy), `on_update`, `on_eval` and `on_training_end`; returning `False`
from any hook stops training. See {doc}`api/training`.

---

## 5. Load a checkpoint

`Cls.load(...)` accepts a directory written by `agent.save(...)` and the
single-file checkpoints `srl-train` writes (`ckpt_<step>.pt`, `final_<step>.pt`,
`best_<step>.pt` under `<ckptdir>/<algo>_<config-stem>/`):

```python
import glob

from srl import SAC

path = sorted(glob.glob("checkpoints/sac_halfcheetah_sac/final_*.pt"))[-1]
agent = SAC.load(path)                       # model rebuilt from the stored YAML
action, _ = agent.predict(obs, deterministic=True)
```

For resuming a CLI run use `srl-train --resume <checkpoint>`; to restore just
the network weights, `CheckpointManager(dir).load(model, path)` still works.
