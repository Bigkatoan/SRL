# srl.training

Trainers, evaluation, callbacks and environment construction — the code
behind {meth}`BaseAgent.learn() <srl.core.base_agent.BaseAgent.learn>` and
`srl-train`.

## Trainers

```{eval-rst}
.. autoclass:: srl.training.Trainer
   :members: run
   :show-inheritance:

.. autoclass:: srl.training.OnPolicyTrainer
   :show-inheritance:

.. autoclass:: srl.training.OffPolicyTrainer
   :show-inheritance:
```

## Evaluation

```{eval-rst}
.. autoclass:: srl.training.Evaluator
   :members: evaluate, maybe_evaluate, finalize, reset_schedule, close
   :show-inheritance:

.. autofunction:: srl.training.evaluate_agent
```

## Callbacks

```{eval-rst}
.. automodule:: srl.training.callbacks
   :no-members:

.. autoclass:: srl.training.Callback
   :members: on_training_start, on_step, on_rollout_end, on_update, on_eval, on_training_end

.. autoclass:: srl.training.CallbackList
   :show-inheritance:

.. autoclass:: srl.training.CheckpointCallback
   :show-inheritance:

.. autoclass:: srl.training.EvalCallback
   :show-inheritance:

.. autoclass:: srl.training.EarlyStopping
   :show-inheritance:

.. autoclass:: srl.training.LogCallback
   :show-inheritance:
```

## Environments

```{eval-rst}
.. autofunction:: srl.training.make_env

.. autofunction:: srl.training.wrap_env

.. autofunction:: srl.training.build_training_env
```

## Configuration and logging

```{eval-rst}
.. autofunction:: srl.training.make_algo_config

.. autoexception:: srl.training.ConfigurationError

.. autoclass:: srl.training.NullLogger
   :show-inheritance:
```
