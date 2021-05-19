Problem
*******

When doing neural architecture search with DeepHyper there are two different settings that you can use:
* static settings related to the definition of your task.
* dynamic settings which corresponds to the search algorithm and the execution policy.

The static settings are defined using the :class:`deephyper.problem.NaProblem`. Let us see step by step how to define this ``NaProblem``. First, when creating a ``NaProblem`` object it is a good practice to define a random seed for reproducibility purposes.

.. code-block:: python

    from deephyper.problem import NaProblem

    Problem = NaProblem(seed=2019)

Loading the data
================

Then, a ``load_data`` callable has to be defined:

.. code-block:: python

    Problem.load_data(load_data, load_data_kwargs)

This ``load_data`` callable can follow two different interfaces: Numpy arrays or generators.

Numpy arrays
------------

In the case of Numpy arrays, the callable passed to ``Problem.load_data(...)`` has to return the following tuple: ``(X_train, y_train), (X_valid, y_valid)``. In the most simple case where the model takes a single input, each of these elements is a Numpy array. Generally, ``X_train`` and ``y_train`` have to be of the same length (i.e., same ``array.shape[0]``) which is also the case for ``X_valid`` and ``y_valid``. Similarly, the shape of the elements of ``X_train`` and ``X_valid`` which is also the case for ``y_train`` and ``y_valid``. An example ``load_data`` function can be

.. code-block:: python

    import numpy as np

    def load_data(N=100):
        X = np.zeros((N, 1))
        y = np.zeros((N,1))
        return (X, y), (X, y)


It is also possible for the model to take several inputs. In fact, experimentaly it can be notices that separating some inputs with different inputs can significantly help the learning of the model. Also, sometimes different inputs may be of the "types" for example two molecular fingerprints. In this case, it can be very interesting to share the weights of the model to process these two inputs. In the case of multi-inputs models the ``load_data`` function will also return ``(X_train, y_train), (X_valid, y_valid)`` bu where ``X_train`` and ``X_valid`` are two lists of Numpy arrays. For example, the following is correct:

.. code-block:: python

    import numpy as np

    def load_data(N=100):
        X = np.zeros((N, 1))
        y = np.zeros((N,1))
        return ([X, X], y), ([X, X], y)


Generators
----------

Returning generators with a single input:

.. code-block:: python

    def load_data(N=100):

        tX, ty = np.zeros((N,1)), np.zeros((N,1))
        vX, vy = np.zeros((N,1)), np.zeros((N,1))

        def train_gen():
            for x, y in zip(tX, ty):
                yield ({"input_0": x}, y)

        def valid_gen():
            for x, y in zip(vX, vy):
                yield ({"input_0": x}, y)

        res = {
            "train_gen": train_gen,
            "train_size": N,
            "valid_gen": valid_gen,
            "valid_size": N,
            "types": ({"input_0": tf.float64}, tf.float64),
            "shapes": ({"input_0": (1, )}, (1, ))
            }
        return res

Returning generators with multiple inputs:

.. code-block:: python

    def load_data(N=100):

        tX0, tX1, ty = np.zeros((N,1)), np.zeros((N,1)), np.zeros((N,1)),
        vX0, vX1, vy = np.zeros((N,1)), np.zeros((N,1)), np.zeros((N,1)),

        def train_gen():
            for x0, x1, y in zip(tX0, tX1, ty):
                yield ({
                    "input_0": x0,
                    "input_1": x1
                    }, y)

        def valid_gen():
            for x0, x1, y in zip(vX0, vX1, vy):
                yield ({
                    "input_0": x0,
                    "input_1": x1
                }, y)

        res = {
            "train_gen": train_gen,
            "train_size": N,
            "valid_gen": valid_gen,
            "valid_size": N,
            "types": ({"input_0": tf.float64, "input_1": tf.float64}, tf.float64),
            "shapes": ({"input_0": (5, ), "input_1": (5, )}, (1, ))
            }
        print(f'load_data:\n', pformat(res))
        return res


Defining the search space
=========================

Then, a function defining the neural architecture search space has to be defined.

.. code-block:: python

    Problem.search_space(create_search_space, num_layers=10)


The ``create_search_space`` function has to follow a specific interface. First it has to return a :class:`deephyper.nas.space.KSearchSpace` or a :class:`deephyper.nas.space.AutoKSearchSpace`. An example ``create_search_space`` function can be:

.. code-block:: python

    import collections

    import tensorflow as tf

    from deephyper.nas.space import AutoKSearchSpace
    from deephyper.nas.space.node import ConstantNode, VariableNode
    from deephyper.nas.space.op.basic import Tensor
    from deephyper.nas.space.op.connect import Connect
    from deephyper.nas.space.op.merge import AddByProjecting
    from deephyper.nas.space.op.op1d import Dense, Identity


    def add_dense_to_(node):
        node.add_op(Identity()) # we do not want to create a layer in this case

        activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
        for units in range(16, 97, 16):
            for activation in activations:
                node.add_op(Dense(units=units, activation=activation))


    def create_search_space(input_shape=(1,),
                            output_shape=(1,),
                            num_layers=10,
                            *args, **kwargs):

        # print("input_shape:", input_shape, ", output_shape:", output_shape,", num_layers:", num_layers)
        arch = AutoKSearchSpace(input_shape, output_shape, regression=True)
        source = prev_input = arch.input_nodes[0]

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(num_layers):
            vnode = VariableNode()
            add_dense_to_(vnode)

            arch.connect(prev_input, vnode)

            # * Cell output
            cell_output = vnode

            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Tensor([]))
                skipco.add_op(Connect(arch, anchor))
                arch.connect(skipco, cmerge)

            # ! for next iter
            prev_input = cmerge
            anchor_points.append(prev_input)


        return arch


Defining the hyperparameters
============================

Fixed hyperparameters
---------------------

In neural architecture search a fixed configuration of hyperparameters is set. For example:

.. code-block:: python

    Problem.hyperparameters(
        batch_size=256,
        learning_rate=0.01,
        optimizer="adam",
        num_epochs=20,
        verbose=0,
        callbacks=dict(...),
    )

Searched hyperparameters
------------------------

It is also possible to search over hyperparameters and neural architectures in same time. For example:

.. code-block:: python

    Problem.hyperparameters(
        batch_size=Problem.add_hyperparameter((16, 2048, "log-uniform"), "batch_size"),
        learning_rate=Problem.add_hyperparameter(
            (1e-4, 0.01, "log-uniform"),
            "learning_rate",
        ),
        optimizer=Problem.add_hyperparameter(
            ["sgd", "rmsprop", "adagrad", "adam", "adadelta", "adamax", "nadam"], "optimizer"
        ),
        patience_ReduceLROnPlateau=Problem.add_hyperparameter(
            (3, 30), "patience_ReduceLROnPlateau"
        ),
        patience_EarlyStopping=Problem.add_hyperparameter((3, 30), "patience_EarlyStopping"),
        num_epochs=100,
        verbose=0,
        callbacks=dict(
            ReduceLROnPlateau=dict(monitor="val_r2", mode="max", verbose=0, patience=5),
            EarlyStopping=dict(
                monitor="val_r2", min_delta=0, mode="max", verbose=0, patience=10
            ),
        ),
    )


Defining the loss function
==========================

Then a loss function has to be defined:

.. code-block:: python

    Problem.loss("categorical_crossentropy")


A custom loss can also be defined:

.. code-block:: python

    def NLL(y, rv_y):
        return -rv_y.log_prob(y)

    Problem.loss(NLL)

The loss can also be automatically searched:

.. code-block:: python

    Problem.loss(
        Problem.add_hyperparameter(
            ["mae", "mse", "huber_loss", "log_cosh", "mape", "msle"], "loss"
        )
    )



Defining metrics
================

A list of metrics can be defined to be monitored or used as an objective. It can be a keyword or a callable. For example, if it is a keyword:

.. code-block:: python

    Problem.metrics(["acc"])

In case you need multiple metrics:

.. code-block:: python

    Problem.metrics["mae", "mse"]

In case you want to use a custom metric:

.. code-block:: python

    def sparse_perplexity(y_true, y_pred):
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        perplexity = tf.pow(2.0, cross_entropy)
        return perplexity

    Problem.metrics([sparse_perplexity])

Defining the objective
======================

DeepHyper will maximise the defined objective. If you want to use the validation accuracy at the last epoch:

.. code-block:: python

    Problem.objective("val_acc")

It can accept some prefix and suffix such as:

.. code-block:: python

    Problem.objective("-val_acc__max")

It can be a callable:

.. code-block:: python

    def myobjective(history: dict) -> float:
        return history["val_acc"][-1]

    Problem.objective(myobjective)


Defining post-training settings
===============================

Some other settings for the post-training can be defined.

.. code-block:: python

    Problem.post_training(
        num_epochs=1000,
        metrics=['r2'],
        callbacks=dict(
            ModelCheckpoint={
                'monitor': 'val_r2',
                'mode': 'max',
                'save_best_only': True,
                'verbose': 1
            },
            EarlyStopping={
                'monitor': 'val_r2',
                'mode': 'max',
                'verbose': 1,
                'patience': 10
            },
            TensorBoard={
                'log_dir':'tb_logs',
                'histogram_freq':1,
                'batch_size':64,
                'write_graph':True,
                'write_grads':True,
                'write_images':True,
                'update_freq':'epoch'
            }
        )
    )