# srl.config and srl.registry

## Loading and dumping experiments

```{eval-rst}
.. autofunction:: srl.config.experiment.load_experiment
.. autofunction:: srl.config.experiment.dump_resolved_config
.. autofunction:: srl.config.experiment.experiment_to_dict
.. autofunction:: srl.config.experiment.infer_algo_name
.. autoclass:: srl.config.experiment.ExperimentConfig
.. autoclass:: srl.config.experiment.EnvConfig
```

## The `train:` block

```{eval-rst}
.. autofunction:: srl.config.algos.build_algo_config
.. autofunction:: srl.config.algos.resolve_algo_config_cls
.. autofunction:: srl.config.algos.build_trainer_config
.. autofunction:: srl.config.algos.build_runner_config
.. autofunction:: srl.config.algos.normalize_offpolicy_schedule
.. autofunction:: srl.config.algos.split_train_section
.. autoclass:: srl.config.algos.TrainerConfig
.. autoclass:: srl.config.algos.OffPolicySchedule
```

## `auto` dimensions

```{eval-rst}
.. autofunction:: srl.config.auto_dims.resolve_auto_dims
.. autofunction:: srl.config.auto_dims.resolve_auto_dims_from_env
```

## Errors and warnings

```{eval-rst}
.. autoclass:: srl.config._validate.ConfigError
.. autoclass:: srl.config._validate.ConfigDeprecationWarning
```

## Registries

```{eval-rst}
.. automodule:: srl.registry.registry
   :members: RegistryEntry, AlgorithmSpec
```
