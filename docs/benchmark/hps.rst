Hyper Parameters Search (HPS)
*****************************

.. automodule:: deephyper.benchmark.hps

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
by their name, their space and a default value for the starting point. Deephyper
recognizes three types of parameters:
- continuous
- discrete ordinal (for instance integers)
- discrete non-ordinal (for instance a list of tokens)
For example if we want to create an hyper parameter search problem for Mnist dataset:


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/problem.py


and that's it, we just defined a problem with 8 dimensions: epochs, nhidden, nunits,
activation, batch_size, dropout, optimizer and learning_rate. Now we need to define a
function which will run our mnist model while taking in account the parameters chosen by
the search.


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/mnist_mlp.py


.. WARNING::
    When designing a new optimization experiment, keep in mind ``model_run.py``
    must be runnable from an arbitrary working directory. This means that Python
    modules simply located in the same directory as the ``model_run.py`` will not be
    part of the default Python import path, and importing them will cause an ``ImportError``!

To ensure that modules located alongside the ``model_run.py`` script are
always importable, a quick workaround is to explicitly add the problem
folder to ``sys.path`` at the top of the script:

::

    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # import user modules below here

.. _available-hps-benchmarks:

Available HPS benchmarks
========================

============== ================ =====================================
      Hyper Parameters Search Benchmarks ``deephyper.benchmark.hps``
---------------------------------------------------------------------
     Name            Type                    Description
============== ================ =====================================
 mnistmlp       Classification   http://yann.lecun.com/exdb/mnist/
 polynome2      Dummy
============== ================ =====================================
