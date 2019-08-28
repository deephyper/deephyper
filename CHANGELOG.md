# Changelog - DeepHyper v0.1.1-rc0

This release is mostly introducing features for Neural Architecture Search with DeepHyper.

## NEW: DeepHyper command line interface

For hyperparameter search use `deephyper hps ...` here is an example for the
hyperparameter polynome2 benchmark:

```bash
deephyper hps ambs --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run
```

For neural architecture search use `deephyper nas ...` here is an example for the
neural architecture search linearReg benchmark:

```bash
deephyper nas regevo --problem deephyper.benchmark.nas.linearReg.Problem
```

Use commands such as `deephyper --help`, `deephyper nas --help` or `deephyper nas regevo --help` to find out more about the command line interface.

## NEW: Create an Operation from a keras Layer

* Create a new `Operation` directly from `tensorflow.keras.layers`:

```
>>> import tensorflow as tf
>>> from deephyper.search.nas.model.space.node import VariableNode
>>> from deephyper.search.nas.model.space.op.op1d import Operation
>>> vnode = VariableNode()
>>> vnode.add_op(Operation(layer=tf.keras.layers.Dense(10)))
```

* `TrainerTrainValid` now has a default callback: `tf.keras.callbacks.CSVLogger(...)`

* Deephyper cli:
    * deephyper nas ppo,regevo,random,ambs
    * deephyper hps ambs,ga
* Ray evaluator
* Problem seed:
    * Problem(seed=seed)
    * Cli —seed
* NAS ambs: n_jobs
* MimeNode
