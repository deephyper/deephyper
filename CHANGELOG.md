# Changelog - DeepHyper v0.1.1-rc0

## New features

* Create a new `Operation` from `tensorflow.keras.layers`:

```
>>> import tensorflow as tf
>>> from deephyper.search.nas.model.space.node import VariableNode
>>> from deephyper.search.nas.model.space.op.op1d import Operation
>>> vnode = VariableNode()
>>> vnode.add_op(Operation(layer=tf.keras.layers.Dense(10)))
```

* `TrainerTrainValid` now has a default callback: `tf.keras.callbacks.CSVLogger(...)`