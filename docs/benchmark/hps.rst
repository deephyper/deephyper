Hyper Parameters Search (HPS)
*****************************

.. automodule:: deephyper.benchmark.hps

============== ================ =====================================
      Hyper Parameters Search Benchmarks ``deephyper.benchmark.hps``
---------------------------------------------------------------------
     Name            Type                    Description
============== ================ =====================================
 b1
 b2
 b3
 capsule
 cifar10cnn     Classification   https://www.cs.toronto.edu/~kriz/cifar.html
 dummy1
 dummy2
 gcn
 mnistcnn       Classification   http://yann.lecun.com/exdb/mnist/
 mnistmlp       Classification   http://yann.lecun.com/exdb/mnist/
 rosen2
 rosen10
 rosen30
============== ================ =====================================

.. _create-new-hps-problem:

Create a new HPS problem
========================

For HPS a benchmark is defined by a problem definition and a function that runs the model.

::

      problem_folder/
            __init__.py
            problem.py
            model_run.py

The problem contains the parameters you want to search over. They are defined
by their name, their space and a default value for the starting point. Deephyper recognizes three types of parameters :
- continuous
- discrete ordinal (for instance integers)
- discrete non-ordinal (for instance a list of tokens)
For example if we want to create an hyper parameter search problem for Mnist dataset :


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/problem.py


and that's it, we just defined a problem with one dimension 'num_n_l1' where we are going to search the best number of neurons for the first dense layer.

Now we need to define how to run hour mnist model while taking in account this 'num_n_l1' parameter chosen by the search. Let's take an basic example from Keras documentation with a small modification to use the 'num_n_l1' parameter :


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/mnist_mlp.py


.. WARNING::
    When designing a new optimization experiment, keep in mind `model_run.py`
    must be runnable from an arbitrary working directory. This means that Python
    modules simply located in the same directory as the `model_run.py` will not be
    part of the default Python import path, and importing them will cause an `ImportError`!

To ensure that modules located alongside the `model_run.py` script are always importable, a
quick workaround is to explicitly add the problem folder to `sys.path` at the top of the script

::

    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # import user modules below here
