# srl.utils

## Logger

```{eval-rst}
.. autoclass:: srl.utils.logger.Logger
   :members:
   :show-inheritance:
```

## CheckpointManager

```{eval-rst}
.. autoclass:: srl.utils.checkpoint.CheckpointManager
   :members:
   :show-inheritance:
```

## Safe serialization

```{eval-rst}
.. automodule:: srl.utils.safe_serialization
   :members: pack, unpack, load_weights_only, load_trusted_or_legacy, LegacyCheckpointWarning
```

## Callbacks

The callbacks moved to {mod}`srl.training.callbacks` (see the
[srl.training reference](training.md#callbacks)); `srl.utils.callbacks`
re-exports them for backward compatibility.
