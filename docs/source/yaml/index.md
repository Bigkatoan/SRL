# YAML Configuration System

YAML is SRL's central configuration language. The whole model architecture — encoders,
flows, actor/critic heads, losses, and training hyperparameters — is declared in a
single YAML file, which `ModelBuilder` then materialises into a runtime model graph.

## Why SRL uses YAML

| Without YAML | With SRL + YAML |
|---|---|
| Hand-write `nn.Module` subclasses | Declare an encoder type and its layers |
| Hard-code observation routing inside `forward()` | Declare a `flows` graph |
| Copy hyperparameters between scripts | One YAML file serves training, visualization, and benchmarking |
| Experiments are hard to reproduce | The config file is the source of truth |

## Build pipeline

```
YAML file
  ↓ ModelBuilder.from_yaml(path)
  ↓ parse → AgentModelConfig, EncoderConfig, HeadConfig, LossConfig
  ↓ instantiate encoders + heads
  ↓ FlowGraph parses the `flows` edges into a DAG
  ↓
AgentModel (runtime)
  ↓ observation → encoder graph → latent concat → head dispatch
  ↓
actor(·), critic(·)
```

## A minimal YAML file

```yaml
# configs/envs/halfcheetah_sac.yaml
env_id:   HalfCheetah-v5
env_type: flat
algo:     sac

encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
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

train:
  total_steps: 1_000_000
  batch_size:  256
  lr_actor:    3e-4
  lr_critic:   3e-4
```

Run it:

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml
```

## Pages in this section

- [Encoders](encoders.md) — declaring and configuring feature extractors (MLP, CNN, LSTM, text)
- [Heads & Flows](heads_flows.md) — actor/critic heads and the routing graph
- [Auxiliary Representation Learning](auxiliary.md) — autoencoder, VAE, CURL, BYOL, DrQ, SPR, Barlow Twins
- [Training Block Reference](training_block.md) — every field in the `train:` block

## See also

- [YAML Core Guide](../yaml_core.md) — the narrative walkthrough of the declarative layer
- [Training System](../training/index.md) — trainers, runners, and optimizer patterns
- [Configuration Reference](../config_reference.md) — the complete field-level reference
