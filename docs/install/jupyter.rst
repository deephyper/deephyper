.. _install-jupyter:

Install DeepHyper with Jupyter
******************************

Sometimes the environment in which DeepHyper was installed is not detected by Jupyter. To create a custom Jupyter kernel run the following from your activated Conda environment:

.. code-block:: console

    $ python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Now when you open a Jupyter notebook the ``Python (deephyper)`` kernel will be available.

.. note:: More details about DeepHyper's installation can be found in the :ref:`install-pip` section.
