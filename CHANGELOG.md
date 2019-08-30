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

The goal of `MimeNode` is to replicate the action applied on the targeted variable node.

```python
import tensorflow as tf

from deephyper.search.nas.model.space.node import VariableNode, MimeNode
from deephyper.search.nas.model.space.op.op1d import Dense

vnode = VariableNode()
dense_10_op = Dense(10)
vnode.add_op(dense_10_op)
vnode.add_op(Dense(20))

mnode = MimeNode(vnode)
dense_30_op = Dense(30)
mnode.add_op(dense_30_op)
mnode.add_op(Dense(40))

# The first operation "Dense(10)" has been choosen
# for the mimed node: vnode
vnode.set_op(0)

assert vnode.op == dense_10_op

# mnode is miming the choise made for vnode as you can see
# the first operation was choosen as well
assert mnode.op == dense_30_op
```

## MirrorNode to reuse same operation

The goal of `MirroNode` is to replicate the action applied on the targeted `VariableNode`,
`ConstantNode` or `MimeNode`.

```python
import tensorflow as tf

from deephyper.search.nas.model.space.node import VariableNode, MirrorNode
from deephyper.search.nas.model.space.op.op1d import Dense

vnode = VariableNode()
dense_10_op = Dense(10)
vnode.add_op(dense_10_op)
vnode.add_op(Dense(20))

mnode = MirrorNode(vnode)

# The operation "Dense(10)" is being set for vnode.
vnode.set_op(0)

# The same operation (i.e. same instance) is now returned by both vnode and mnode.
assert vnode.op == dense_10_op
assert mnode.op == dense_10_op
```

## Tensorboard and Beholder callbacks available for post-training

Tensorboard and Beholder callbacks can now be used during the post-training. Beholder is
a Tensorboard which enable you to visualize the evolution of the trainable parameters of
model during the training.

```python
Problem.post_training(
    ...
    callbacks=dict(
        TensorBoard={
            'log_dir':'tb_logs',
            'histogram_freq':1,
            'batch_size':64,
            'write_graph':True,
            'write_grads':True,
            'write_images':True,
            'update_freq':'epoch',
            'beholder': True
        })
)
```

