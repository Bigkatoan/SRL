# Example: Encoder Configurations for SAC / TD3 / Actor-Critic

This page collects YAML snippets and quick smoke tests so you can try encoders in
realistic setups.

```{note}
These configs are not shipped with the repository — copy a snippet into a file of your
own (`configs/examples/` is a reasonable home for them) before running the commands
below. The shipped configs live in [configs/envs/](https://github.com/Bigkatoan/SRL/tree/main/configs/envs).
```

## 1) SAC — vision (CNN) with a dedicated encoder optimizer

Goal: use a `cnn` encoder with a separate `encoder_optimizer` that steps every
`encoder_update_freq` critic updates.

```yaml
# configs/examples/sac_image_encoder.yaml
env_id: CarRacing-v3
env_type: flat
algo: sac

encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 96, 96]
    latent_dim: 256
    aux_type: autoencoder
    aux_latent_dim: 128
    layers:
      - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 3, stride: 1, padding: 0, activation: relu}

flows:
  - "image_enc -> actor"
  - "image_enc -> critic"

actor:
  name: actor
  type: squashed_gaussian
  action_dim: 3

critic:
  name: critic
  type: twin_q
  action_dim: 3

train:
  total_steps: 200000
  batch_size: 256
  encoder_update_freq: 2
  lr_actor: 3e-4
  lr_critic: 3e-4
  buffer_size: 100000
  start_steps: 1000
  update_after: 1000
  update_every: 50
```

Smoke test:

```bash
srl-train --config configs/examples/sac_image_encoder.yaml --device cpu --seed 0 --no-plots
```

```{note}
`encoder_update_freq` is a real `SACConfig` field, so it works from YAML. The encoder
optimizer's learning rate (`encoder_lr`) and the auxiliary objective (`aux_loss_type`)
live on `VisualSACConfig`, which the CLI never builds — without `encoder_lr` the
encoder optimizer inherits `lr_critic`. See
[Auxiliary Representation Learning](../yaml/auxiliary.md).
```

---

## 2) TD3 — MLP (state-based), fast

An example for a vector-only environment (HalfCheetah). TD3 uses a deterministic actor
and requires a twin-Q critic.

```yaml
# configs/examples/td3_state_encoder.yaml
env_id: HalfCheetah-v5
env_type: flat
algo: td3

encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 128
    layers:
      - {out_features: 256, activation: relu}
      - {out_features: 128, activation: relu}

flows:
  - "state_enc -> actor"
  - "state_enc -> critic"

actor:
  name: actor
  type: deterministic
  action_dim: 6

critic:
  name: critic
  type: twin_q
  action_dim: 6

train:
  total_steps: 300000
  batch_size: 256
  lr_actor: 1e-3
  lr_critic: 1e-3
```

Smoke test:

```bash
srl-train --config configs/examples/td3_state_encoder.yaml --device cpu --no-plots
```

```{warning}
`srl-train` rejects mismatched head types before building anything. TD3 requires
`deterministic` + `twin_q`; a single `q` critic is only valid for DDPG.
```

---

## 3) Multi-modal actor-critic (image + state)

When the actor and critic need different information, give each branch its own
encoders.

```yaml
# configs/examples/multi_modal_ac.yaml
env_id: SomeEnv-v0     # replace with a real env that returns both keys
env_type: flat
algo: sac

encoders:
  - name: image_enc
    type: cnn
    input_name: front_camera
    input_shape: [3, 84, 84]
    latent_dim: 128
  - name: state_enc
    type: mlp
    input_name: joint_states
    input_dim: 10
    latent_dim: 64

flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "image_enc -> critic"
  - "state_enc -> critic"

actor:
  name: actor
  type: squashed_gaussian
  action_dim: 4

critic:
  name: critic
  type: twin_q
  action_dim: 4

train:
  total_steps: 500000
  encoder_update_freq: 2
  lr_actor: 3e-4
  lr_critic: 3e-4
```

Both heads see `128 + 64 = 192` input features here — the builder sums the upstream
`latent_dim`s, so there is no `input_dim` to set on the heads.

---

## Tips and checklist

- Always set `input_name` when the observation dict has more than one key, so routing
  is explicit instead of relying on the count-matching fallback.
- Check the graph before a long run:

  ```python
  from srl.registry.builder import ModelBuilder

  model = ModelBuilder.from_yaml("configs/examples/multi_modal_ac.yaml")
  print(model.encoder_names_for_head("actor"))   # encoders feeding the actor
  print(model.encoder_names_for_head("critic"))
  ```

  `srl-visualize --config <file>` renders the same graph as a PNG.
- For CURL/BYOL, set `use_momentum: true` on the encoder and raise the batch size — the
  loss returns `None` without a momentum encoder.
- On Isaac Lab vision tasks, `encoder_update_freq: 2` and a large batch are a good
  starting point.
- CNN shorthand layers are `[out_channels, kernel, padding, activation, pooling]`. The
  third slot is padding, not stride — use the dict form when you need a stride.

## See also

- [Encoders guide](../encoders.md)
- [YAML: Encoders](../yaml/encoders.md)
- [Heads & Flows](../yaml/heads_flows.md)
