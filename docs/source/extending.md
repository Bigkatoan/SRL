# Extending SRL (plugins)

SRL resolves every `type:` / `algo:` / `env_type:` name in a YAML file through a
registry. The built-in types are registered through exactly the same API as yours, so
a custom encoder is a first-class citizen: it is validated, listed in error messages,
and built by `ModelBuilder` like `mlp` or `cnn`.

| Registry | Decorator | Entry-point group | What is registered |
|---|---|---|---|
| `EncoderRegistry` | `register_encoder(name)` | `srl.encoders` | `factory(cfg: EncoderConfig) -> nn.Module` |
| `HeadRegistry` | `register_head(name)` | `srl.heads` | `factory(cfg: HeadConfig, input_dim: int) -> nn.Module` |
| `AlgorithmRegistry` | `register_algorithm(name)` | `srl.algorithms` | an `AlgorithmSpec` |
| `EnvRegistry` | `register_env(name)` | `srl.envs` | `factory(env_id, *, device, n_envs, render) -> env` |

All of them live in `srl.registry`. A "factory" can be a class (its constructor is the
factory) or a function.

## Option 1 — register in Python

Import the module that registers your type before loading the config:

```python
import torch.nn as nn

from srl.registry import register_encoder


@register_encoder("pointnet", requires=("input_dim",), extra_keys=("num_points",))
class PointNetEncoder(nn.Module):
    def __init__(self, cfg):              # cfg: srl.registry.EncoderConfig
        super().__init__()
        self.num_points = int(cfg.extra.get("num_points", 1024))
        self.net = nn.Linear(cfg.input_dim, cfg.latent_dim)

    def forward(self, x):
        return self.net(x)
```

```yaml
encoders:
  - name: cloud_enc
    type: pointnet
    input_dim: auto        # resolved from the env, because requires=("input_dim",)
    latent_dim: 128
    num_points: 2048       # accepted because it is listed in extra_keys
```

Registration options:

- `requires=("input_dim",)` / `("input_shape",)` — the config attributes your type
  needs. They are checked before building, and they also tell
  `resolve_auto_dims` how to fill `auto`: `input_dim` gets the flat observation size,
  `input_shape` the `[C, H, W]` image shape.
- `extra_keys=(...)` — type-specific YAML keys accepted next to the standard
  `EncoderConfig`/`HeadConfig` fields and collected into `cfg.extra`. Any other
  unknown key is an error with a "did you mean" suggestion; free-form arguments can
  always be passed under an explicit `extra:` mapping.
- `allow_extra=True` — accept *any* extra key (disables typo detection; prefer
  `extra_keys`).
- `requires_action_dim=True` (heads) — the head needs `action_dim`, which is then
  filled from the env's action space when it is `auto`/omitted.
- `override=True` — replace an existing registration (registering an existing name,
  including a built-in one, otherwise raises).

The same options can be given as class attributes (`srl_requires`, `srl_extra_keys`,
`srl_allow_extra`, `srl_requires_action_dim`), which is handy for entry-point plugins.

## Option 2 — entry-point plugins (no import needed)

Ship the type in its own package and declare it in that package's `pyproject.toml`:

```toml
[project.entry-points."srl.encoders"]
pointnet = "my_srl_plugins.encoders:PointNetEncoder"

[project.entry-points."srl.heads"]
mixture = "my_srl_plugins.heads:MixtureHead"

[project.entry-points."srl.envs"]
mysim = "my_srl_plugins.envs:make_env"

[project.entry-points."srl.algorithms"]
my_ppo = "my_srl_plugins.algos:MY_PPO_SPEC"
```

After `pip install my-srl-plugins`, `type: pointnet` just works. Discovery is lazy:
only the entry-point *names* are read when SRL lists or validates types; the plugin
module is imported the first time its name is used. A plugin that fails to import
raises an `ImportError` naming the entry point and its distribution. Entry points
never shadow a name that is already registered (built-ins win).

### Env-type plugins

```python
def make_env(env_id: str, *, device: str = "cpu", n_envs: int = 1, render: bool = False):
    ...  # return an SRL env (obs dict, act_space, ...)
```

`env_type: mysim` then validates; `EnvRegistry.get("mysim")(env_id, device=..., n_envs=...)`
builds the env. The built-in `flat`/`goal`/`racecar`/`isaaclab`/`mjlab` types are
registered the same way.

### Algorithm plugins

```python
from dataclasses import dataclass

from srl.core.config import PPOConfig
from srl.registry import AlgorithmSpec


@dataclass
class MyPPOConfig(PPOConfig):
    my_coef: float = 0.1


MY_PPO_SPEC = AlgorithmSpec(
    name="my_ppo",
    config_cls=MyPPOConfig,
    agent="my_srl_plugins.algos:MyPPO",   # class or "module:attr", loaded lazily
    off_policy=False,
    actor_types=("gaussian",),
    critic_types=("value",),
)
```

`algo: my_ppo` then validates its `train:` block strictly against `MyPPOConfig`
(including the inherited enum/range checks), and `srl.config.build_algo_config("my_ppo", train)`
builds it.

## Testing a plugin without installing it

`srl.registry.registry._entry_points` is the single function that queries
`importlib.metadata`; tests can monkeypatch it to return fake `EntryPoint` objects and
call `Registry.refresh_plugins()` (see `tests/test_config_resolution.py`).
