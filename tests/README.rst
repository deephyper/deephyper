Tests
*****

For automatic tests in DeepHyper we chose to use the `Pytest <https://docs.pytest.org/en/latest/index.html>`_ package.


Developer Installation
======================


Follow the :ref:`local-dev-installation`.

Running Tests
=============

Tests corresponding to  ``deephyper`` modules are located in the ``tests`` folder. These tests are marked with possible marks such as:

- fast: tests that are fast to run.
- slow: tests that are slow to run.
- hps: tests that are related to hyperparameter search.
- nas: tests that are related to neural architecture search.
- mpi: tests that require MPI.
- ray: tests that require Ray.
- redis: tests that require Redis.

To run corresponding tests these markers can be used such as:

.. code-block:: console

    pytest --run fast,hps tests/

Testing Notebooks
=================

To test notebooks the ``develop`` branch of deephyper can be installed with pip by using the following command::

    pip install -e git+https://github.com/deephyper/deephyper.git@develop#egg=deephyper


Writting Tests
==============

Tests are located in the ``tests`` folder. Each module from ``deephyper`` should have a corresponding test module with the same name but with the ``test_`` prefix.

For example, the ``deephyper/stopper/_median_stopper.py`` module should have a corresponding ``tests/deephyper/stopper/ test__median_stopper.py`` module.

The test module should start by importing ``pytest``.

.. code-block:: python

    import pytest

Then, each test function should have a name starting with ``test_``. For example, the ``test__median_stopper.py`` module should have a ``test_median_stopper`` function.

.. code-block:: python

    def test_median_stopper():
        ...

This function should use decorators to classify its type. For example, the ``test_median_stopper`` function should be decorated with the ``@pytest.mark.fast`` decorator and ``@pytest.mark.hps`` decorator.

.. code-block:: python

    @pytest.mark.fast
    @pytest.mark.hps
    def test_median_stopper():
        ...

Each test function creating data (files or directly) should use a temporary directory and make sure the corresponding files are deleted at the end of the test. The ``tmp_path`` fixture is used for this purpose.

.. code-block:: python

    @pytest.mark.fast
    @pytest.mark.hps
    def test_median_stopper(tmp_path):
        ...


.. note::

    If you want to know more about temporary directory or file check the Pytest documentation: `How to use temporary directories and files in tests <https://docs.pytest.org/en/latest/how-to/tmp_path.html>`_.
