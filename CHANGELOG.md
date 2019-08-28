# Changelog - DeepHyper v0.1.1-rc0

This release is mostly introducing features for Neural Architecture Search with DeepHyper.

## DeepHyper command line interface

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

## Create an Operation from a keras Layer

* Create a new `Operation` directly from `tensorflow.keras.layers`:

```python
>>> import tensorflow as tf
>>> from deephyper.search.nas.model.space.node import VariableNode
>>> from deephyper.search.nas.model.space.op.op1d import Operation
>>> vnode = VariableNode()
>>> vnode.add_op(Operation(layer=tf.keras.layers.Dense(10)))
```

## Trainer default CSVLogger callback

* `TrainerTrainValid` now has a default callback: `tf.keras.callbacks.CSVLogger(...)`

## Ray evaluator

The ray evaluator is now available through `... --evaluator ray...` for both hyperparameter
and neural architecture search.

## Seeds for reproducibility

To use a seed for any run do `Problem(seed=seed)` while creating your problem object.

## AMBS learner distributed

Use the `.. --n-jobs ...` to define how to distribute the learner computation in AMBS.

## MimeNode to replicate actions

* MimeNode
