Spack
*****

`Spack <https://spack.readthedocs.io/en/latest/>`_ is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments.


Start by installing Spack on your system. The following command will install Spack in the current directory:

.. code-block:: console
    
    $ git clone -c feature.manyFiles=true https://github.com/spack/spack.git
    $ . ./spack/share/spack/setup-env.sh

Download the deephyper Spack package repository:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper-spack-packages.git
    $ spack repo add deephyper-spack-packages

Create a new environment for DeepHyper:

.. code-block:: console

    $ spack env create deephyper
    $ spack env activate deephyper

.. code-block:: console

    $ spack install py-deephyper
    $ spack load py-deephyper

.. warning::

    The Spack installation will only provide the default DeepHyper installation (i.e., hyperparameter optimization). All features will not be included by default.