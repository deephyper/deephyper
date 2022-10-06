Spack
*****

`Spack <https://spack.readthedocs.io/en/latest/>`_ is package management tool designed for large supercomputing centers.

After installing Spack on your system the following command can be executed to install DeepHyper:

.. code-block:: console

    $ spack install py-deephyper
    $ spack load py-deephyper

.. warning::

    The Spack installation will only provide the default DeepHyper installation (i.e., hyperparameter optimization). All features will not be included by default.