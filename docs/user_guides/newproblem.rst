Defining search problems
*************************

The ``--problem`` and ``--run`` arguments to DeepHyper search methods must be loadable
Python objects, or paths to files containing the objects.  To ensure that imports work
correctly when DeepHyper loads your code, it is best to place your modules inside a
**problem directory** installed in a virtual environment.

This is easily accomplished with:

.. code-block:: console
    :caption: bash

    $ deephyper start-project demo
    $ cd demo/demo/
    $ deephyper new-problem hps my_problem

This will create a ``my_problem`` package that is globally importable within your
virtual environment. Now you can write modules with your ``Problem`` instance and ``run`` function inside
in the ``my_problem`` subdirectory. You are free to define and import
helper modules with the ``from my_problem ...`` syntax.

You can then pass ``--problem my_problem.problem.Problem`` to DeepHyper to
indicate that the ``Problem`` instance is located in the ``problem.py``
module of the ``my_problem`` package.