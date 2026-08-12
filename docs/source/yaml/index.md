# YAML Configuration System

YAML là ngôn ngữ cấu hình trung tâm của SRL. Toàn bộ kiến trúc mô hình — encoder, flows, actor/critic heads, losses, và training hyperparameters — được khai báo trong một file YAML duy nhất, sau đó được materialize thành runtime model graph bởi `ModelBuilder`.

## Tại sao SRL dùng YAML

| Không dùng YAML | Dùng SRL + YAML |
|---|---|
| Viết `nn.Module` thủ công | Khai báo encoder type và layers |
| Hard-code observation routing trong `forward()` | Khai báo `flows` graph |
| Copy hyperparameters giữa nhiều script | Một file YAML dùng cho cả CLI, viz, benchmark |
| Khó tái lập thí nghiệm | Config file = source of truth |

## Build pipeline

```
YAML file
  ↓ ModelBuilder.from_yaml(path)
  ↓ parse → EncoderConfig, HeadConfig, LossConfig
  ↓ instantiate encoders + heads
  ↓ FlowGraph.from_edges(flows)
  ↓
AgentModel (runtime)
  ↓ observation → encoder graph → latent concat → head dispatch
  ↓
actor(·), critic(·), aux_modules(·)
```

## File YAML cơ bản

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
  lr:          3e-4
```

Chạy:

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml
```

## Các trang trong phần này

- [Encoders](encoders.md) — khai báo và cấu hình feature extractors (MLP, CNN, LSTM, text)
- [Heads & Flows](heads_flows.md) — actor/critic heads và routing graph
- [Auxiliary Representation Learning](auxiliary.md) — autoencoder, BYOL, CURL, DrQ, SPR, Barlow
- [Training Block Reference](training_block.md) — tất cả các trường trong block `train:`

## Xem thêm

- [Training System](../training/index.md) — trainer, runner, optimizer patterns
- [Config Reference](../config_reference.md) — field-level reference đầy đủ
