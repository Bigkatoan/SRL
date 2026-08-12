# YAML Core Guide

YAML is the core abstraction in SRL.

The library is designed so that the main shape of an agent is declared in YAML first, then materialized by the builder into a runtime model graph. In practice, this means you should think of a config file as the source of truth for how observations enter the system, how encoders transform them, how latents flow into actor and critic heads, and which built-in loss terms are active.

The Python side is still important, but its primary role is to execute what the YAML declares. The goal of this guide is to explain that declarative layer in detail and make it clear where the current system stops.

## Why YAML is central

SRL uses YAML to make model construction explicit and inspectable.

- You declare encoders instead of manually instantiating PyTorch modules in task scripts.
- You declare graph connectivity through `flows` instead of hiding data movement inside `forward()` code.
- You declare actor and critic heads independently so policy and value branches can diverge.
- You declare currently supported loss terms in one place next to the model.
- You can point CLI training, visualization, and benchmarking tools at the same config file.

This gives the library one stable representation that can be shared across training runs, experiments, visualizers, and documentation.

## Build path

The declarative build path is:

1. A YAML file is loaded by `ModelBuilder.from_yaml(...)`.
2. The raw dictionary is parsed into `AgentModelConfig`, `EncoderConfig`, `HeadConfig`, and `LossConfig` dataclasses.
3. `ModelBuilder` instantiates encoders and heads from those dataclasses.
4. `FlowGraph` parses the `flows` list into a directed acyclic graph.
5. `AgentModel` executes that graph at runtime and performs observation routing, latent concatenation, and head dispatch.

The key implementation anchors are:

- [builder.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/builder.py)
- [config_schema.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/config_schema.py)
- [flow_graph.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/flow_graph.py)
- [agent_model.py](https://github.com/Bigkatoan/SRL/blob/main/srl/networks/agent_model.py)

## Top-level structure

At a high level, a model config is organized like this:

```yaml
env_id: "HalfCheetah-v5"
env_type: "flat"

algo: sac

encoders:
  - name: actor_state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}

flows:
  - "actor_state_enc -> actor"
  - "critic_state_enc -> critic"

actor:
  name: actor
  type: squashed_gaussian
  action_dim: 6

critic:
  name: critic
  type: twin_q
  action_dim: 6

losses:
  - {name: sac_q, weight: 1.0}
  - {name: sac_policy, weight: 1.0}

train:
  total_steps: 1_000_000
  batch_size: 256
```

Conceptually:

- `encoders` define feature extractors.
- `flows` define the graph edges between modules.
- `actor` and `critic` define decision and value heads.
- `losses` define the built-in training terms that should be active.
- `train` provides algorithm-side hyperparameters consumed by the CLI.

## Encoders

Encoders are the entry points of the model graph. Each encoder converts one observation stream into a latent representation.

The current built-in encoder families are:

- `mlp` for vector observations
- `cnn` for image observations
- `lstm` for recurrent vector pipelines
- `text` for character-level text input

Important encoder fields from [config_schema.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/config_schema.py#L34):

- `name`: unique node identifier used by the flow graph
- `type`: built-in encoder type or a registry key for a custom encoder
- `input_name`: explicit observation key this encoder should consume
- `input_dim`: required for MLP/LSTM-style vector encoders
- `input_shape`: required for CNN encoders, typically `[C, H, W]`
- `latent_dim`: output latent width for non-recurrent encoders
- `layers`: per-layer structure used by the underlying builder
- `aux_type`: optional auxiliary head family, currently `autoencoder`, `contrastive`, or `byol`
- `aux_latent_dim`: projection width for contrastive/BYOL heads
- `use_momentum`: wraps the encoder in a momentum/EMA encoder
- `momentum_tau`: EMA coefficient used by the momentum wrapper
- `frame_stack`: declared in schema for stacked inputs
- `recurrent`: wraps non-LSTM encoders in an LSTM layer
- `lstm_hidden`: recurrent hidden size when `recurrent: true` or `type: lstm`

### Example: state encoder

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 12
    latent_dim: 128
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 128, activation: relu, norm: none}
```

### Example: visual encoder with auxiliary reconstruction

This pattern already exists in [car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml).

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 96, 96]
    latent_dim: 256
    aux_type: autoencoder
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

### Example: multimodal encoder set

This pattern is close to [sac_multimodal.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/sac_multimodal.yaml).

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 64, 64]
    latent_dim: 128
    aux_type: contrastive
    aux_latent_dim: 64
    use_momentum: true
    momentum_tau: 0.99

  - name: state_enc
    type: mlp
    input_dim: 18
    latent_dim: 64

  - name: lang_enc
    type: text
    latent_dim: 64
```

## Input routing

Observation routing is where the YAML-centric design becomes especially useful.

At runtime, `AgentModel` tries to map incoming observation keys to encoder names. The most important rule is now explicit routing with `input_name`.

### Recommended pattern: explicit routing

```yaml
encoders:
  - name: policy_state_enc
    type: mlp
    input_name: joint_states
    input_dim: 24
    latent_dim: 128

  - name: image_enc
    type: cnn
    input_name: front_camera
    input_shape: [3, 84, 84]
    latent_dim: 256
```

When `input_name` is present:

- the encoder reads from that exact observation key
- missing required keys raise `KeyError`
- unused observation keys produce a warning after routing

This is the most reliable setup for multimodal systems because it avoids ambiguous heuristic mapping.

### Fallback routing

If `input_name` is omitted, the runtime falls back to compatibility heuristics in [agent_model.py](https://github.com/Bigkatoan/SRL/blob/main/srl/networks/agent_model.py):

- exact key-to-encoder-name matches
- single observation to single encoder remapping
- position-based zip when counts match
- passthrough for the remaining keys

That fallback keeps older configs working, but it is weaker than explicit routing. For new configs, prefer `input_name` whenever the observation dictionary contains more than one semantic input.

## Flow graph

The `flows` list is the core of model composition.

Each item is a directed edge written as:

```yaml
flows:
  - "visual_enc -> actor"
  - "state_enc -> actor"
  - "visual_enc -> critic"
  - "state_enc -> critic"
```

`FlowGraph` interprets these strings as a DAG.

Important properties:

- multiple upstream nodes are concatenated automatically
- execution order is resolved topologically
- cycles are rejected
- actor and critic can consume different encoder sets
- intermediate encoder-to-encoder connections are also allowed if the nodes are declared

This makes model composition explicit. Instead of hiding the wiring inside Python, the graph exists as data and can be visualized or reviewed directly.

## Heads

Actor and critic heads are declared separately from encoders.

Current built-in actor head types:

- `gaussian`
- `squashed_gaussian`
- `deterministic`

Current built-in critic head types:

- `value`
- `twin_q`
- `q`

Example:

```yaml
actor:
  name: actor
  type: squashed_gaussian
  action_dim: 6
  log_std_min: -5.0
  log_std_max: 2.0
  layers:
    - {out_features: 256, activation: relu, norm: layer_norm}

critic:
  name: critic
  type: twin_q
  action_dim: 6
  layers:
    - {out_features: 256, activation: relu, norm: layer_norm}
```

The builder computes each head's input size from the declared flow graph rather than from handwritten `forward()` logic.

## Auxiliary representation learning

Encoders can request an auxiliary module with `aux_type`.

Currently supported auxiliary families in the schema and builder are:

- `autoencoder`
- `contrastive`
- `byol`

What this means today:

- the builder can attach a decoder or projection head automatically
- the config can declare that an encoder participates in auxiliary representation learning
- the runtime model carries those auxiliary modules alongside the main actor/critic graph

This is one of the strongest examples of YAML acting as more than a simple layer list. The config is already expressing representation-learning intent, not just architecture shape.

## Training config from YAML

The CLI also consumes a `train:` section from the same YAML file.

Examples already present in the repo include:

- [halfcheetah_sac.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/halfcheetah_sac.yaml)
- [isaaclab_cartpole_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_cartpole_ppo.yaml)
- [car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml)

Typical fields include:

- `total_steps`
- `n_envs`
- `batch_size`
- `lr`, `lr_actor`, `lr_critic`, `lr_alpha`
- `gamma`, `tau`, `gae_lambda`
- `n_steps`, `n_epochs`, `clip_range`
- `buffer_size`, `start_steps`, `update_after`, `update_every`

These fields are mapped by the training CLI into algorithm config dataclasses in [config.py](https://github.com/Bigkatoan/SRL/blob/main/srl/core/config.py).

## Current system boundary

YAML is the core abstraction of SRL, but it does not yet control every part of training.

What is already declarative today:

- encoder declaration
- layer declaration
- flow graph declaration
- actor/critic declaration
- explicit input routing
- built-in auxiliary module attachment
- built-in loss list declaration
- many algorithm hyperparameters under `train:`
- visualization metadata for pipeline export

What is not yet fully declarative today:

- arbitrary custom reward shaping pipelines
- generic environment wrapper stacks declared in YAML
- optimizer graph customization and parameter groups as first-class schema
- arbitrary custom loss registry selection from YAML
- full training-loop orchestration hooks declared from config alone
- complete input/output contracts and validation schemas beyond the current routing behavior

This boundary matters. The right way to present SRL is not “no Python required for everything,” but “YAML is the central control plane for model structure and a growing portion of training behavior.”

## Recommended reading path

Read the docs in this order:

1. This guide for the mental model.
2. [config_reference.md](config_reference.md) for exact field-level details.
3. [quickstart.md](quickstart.md) for end-to-end usage.
4. [algorithms.md](algorithms.md) for the current training surface and algorithm-side expectations.

## Recommended config patterns

- Use one YAML file per experiment family.
- Prefer explicit `input_name` for multimodal observation dictionaries.
- Keep actor and critic encoders separate when the task benefits from asymmetric representations.
- Use `flows` to show composition explicitly instead of encoding assumptions in task scripts.
-- Treat config files under [configs](https://github.com/Bigkatoan/SRL/tree/main/configs) as executable examples, not just references.

## Related files

- [config_reference.md](config_reference.md)
- [quickstart.md](quickstart.md)
- [algorithms.md](algorithms.md)
- [halfcheetah_sac.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/halfcheetah_sac.yaml)
- [car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml)
- [sac_multimodal.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/sac_multimodal.yaml)