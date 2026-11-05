"""srl.config -- strict loading, validation and dumping of experiment YAML files.

Typical use (what ``srl-train`` / the trainer do)::

    from srl.config import load_experiment, resolve_auto_dims_from_env, dump_resolved_config

    exp = load_experiment("configs/envs/halfcheetah_sac.yaml")   # raises ConfigError on typos
    env = make_env(exp.env)
    model_cfg = resolve_auto_dims_from_env(exp.model, env)      # fills `auto` dims, cross-checks
    model = ModelBuilder.from_config(model_cfg)
    dump_resolved_config(exp, run_dir / "config.yaml")

Exports are resolved lazily so ``srl.core.config`` can import the validator
(``srl.config._validate``) without an import cycle.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # errors / warnings
    "ConfigError": "srl.config._validate",
    "ConfigDeprecationWarning": "srl.config._validate",
    # whole experiment
    "SCHEMA_VERSION": "srl.config.experiment",
    "EnvConfig": "srl.config.experiment",
    "ExperimentConfig": "srl.config.experiment",
    "infer_algo_name": "srl.config.experiment",
    "load_experiment": "srl.config.experiment",
    "experiment_to_dict": "srl.config.experiment",
    "dump_resolved_config": "srl.config.experiment",
    # train: block
    "TrainerConfig": "srl.config.algos",
    "OffPolicySchedule": "srl.config.algos",
    "algorithm_spec": "srl.config.algos",
    "build_algo_config": "srl.config.algos",
    "build_runner_config": "srl.config.algos",
    "build_trainer_config": "srl.config.algos",
    "normalize_offpolicy_schedule": "srl.config.algos",
    "resolve_algo_config_cls": "srl.config.algos",
    "split_train_section": "srl.config.algos",
    # auto dims
    "resolve_auto_dims": "srl.config.auto_dims",
    "resolve_auto_dims_from_env": "srl.config.auto_dims",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'srl.config' has no attribute '{name}'")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
