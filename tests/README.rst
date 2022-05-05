Tests
*****

For automatic tests in DeepHyper we chose to use the `Pytest <https://docs.pytest.org/en/latest/index.html>`_ package.


Developer Installation
======================


Follow the :ref:`local-dev-installation`.

Run Tests
=========

This is the basic and simplest command line to run test.
All test marked as ``@pytest.mark.slow`` will be skipped::

    cd deephyper/
    pytest --cov=deephyper --run-hps --run-fast tests