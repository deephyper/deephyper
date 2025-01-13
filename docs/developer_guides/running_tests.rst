Running Tests
*************

For automatic tests in DeepHyper we use the `Pytest <https://docs.pytest.org/en/latest/index.html>`_ package.

Tests corresponding to  ``deephyper`` modules are located in the ``tests`` folder.

The default test command that corresponds to the default installation of the package ``pip install deephyper`` is (from the root of the repository):

.. code-block:: console

    pytest

This command will run all the tests in the ``tests/`` folder without any ``pytest.mark.*``.

We use markers to classify tests that have specific requirements. Possible marks are:

- ``slow``: marks to define a slow test. For example, the training of a model or the execution of Bayesian optimization.
- ``torch``: marks to define a test that requires PyTorch installed.
- ``tf_keras2``: marks to define a test that requires Tensorflow/Keras 2 installed.
- ``ray``: marks tests which needs Ray installed.
- ``mpi``: marks tests which needs mpi4py and an MPI implementation (e.g., openmpi, mpich) installed.
- ``redis``: marks tests which needs Redis-Stack installed (i.e., includes RedisJSON).
- ``jax``: marks tests which needs JAX installed.
- ``sdv``: marks tests which needs SDV package installed.
- ``memory_profiling``: marks tests which use ``psutil`` installed to profile memory as its behaviour can vary depending on the system were it installed.

The command that we use to run tests with specific markers is:

.. code-block:: console

    pytest --run-marks-subset "slow,torch,tf_keras2,mpi,redis" tests/

This command will run all the tests in the ``tests/`` folder that have a subset of mentionned markers such as:

.. code-block:: console

    @pytest.mark.slow
    def test_some_test_just_slow():
        ...

    @pytest.mark.slow
    @pytest.mark.torch
    def test_some_test_using_torch():
        ...

    @pytest.mark.mpi
    def test_some_test_using_mpi():
        ...


Testing Examples and Notebooks
==============================

To test examples from ``deephyper/examples/`` or tutorial notebooks the ``develop`` branch of deephyper can be installed with pip by using the following command::

    !pip install -e "git+https://github.com/deephyper/deephyper.git@develop#egg=deephyper"


Writing Tests
=============

Tests are located in the ``tests`` folder. Each module from ``deephyper`` should have a corresponding test module with the same name but with the ``test_`` prefix.

For example, the ``deephyper/stopper/_median_stopper.py`` module should have a corresponding ``tests/deephyper/stopper/ test__median_stopper.py`` module.

The test module should start by importing ``pytest``.

.. code-block:: python

    import pytest

Then, each test function should have a name starting with ``test_``. For example, the ``test__median_stopper.py`` module should have a ``test_median_stopper`` function.

.. code-block:: python

    def test_median_stopper():
        ...

This function can use markers to classify its type. For example, the ``test_median_stopper`` function could be decorated with the ``@pytest.mark.redis`` marker if the test uses the ``RedisStorage``.

.. code-block:: python

    @pytest.mark.redis
    def test_median_stopper():
        ...

Each test function creating data (files or directly) should use a temporary directory and make sure the corresponding files are deleted at the end of the test. The ``tmp_path`` fixture is used for this purpose.

.. code-block:: python

    @pytest.mark.redis
    def test_median_stopper(tmp_path):
        ...


.. note::

    If you want to know more about temporary directory or file check the Pytest documentation: `How to use temporary directories and files in tests <https://docs.pytest.org/en/latest/how-to/tmp_path.html>`_.


Profiling Tests
===============

Tests can become slow. To identify sections of code that are slow during tests the Pytest-Profile plugin can be easily installed and used:

.. code-block:: bash

    $ pip install pytest-profiling
    $ pytest tests/hpo/test__cbo.py --profile
