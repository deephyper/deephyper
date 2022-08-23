Tests
*****

For automatic tests in DeepHyper we chose to use the `Pytest <https://docs.pytest.org/en/latest/index.html>`_ package.


Developer Installation
======================


Follow the :ref:`local-dev-installation`.

Run Tests
=========

Tests are marked with possible marks such as:

- fast
- slow
- hps
- nas
- mpi
- ray

To run corresponding tests these markers can be used such as:

.. code-block:: console

    pytest --run fast,hps tests/