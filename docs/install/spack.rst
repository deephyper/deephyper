Spack
*****

`Spack <https://spack.readthedocs.io/en/latest/>`_ is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments. We use Spack to build from source some dependencies of DeepHyper.

Start by installing Spack on your system. The following command will install Spack in the current directory:

.. code-block:: console
    
    $ git clone -c feature.manyFiles=true https://github.com/spack/spack.git
    $ . ./spack/share/spack/setup-env.sh


Installing DeepHyper with Spack
===============================

Additonal documentation here: https://github.com/deephyper/deephyper-spack-packages.

Clone and add the DeepHyper Spack repository to Spack

.. code-block:: console
    
    $ git clone https://github.com/deephyper/deephyper-spack-packages.git
    $ spack repo add deephyper-spack-packages

Create Spack environment for DeepHyper

.. code-block:: console
    
    $ spack env create deephyper
    $ spack env activate deephyper

Add and Install DeepHyper to the spack environment created above

.. code-block:: console
    
    $ spack add py-deephyper
    $ spack install

Install Machine Learning Features 

.. code-block:: console

    $ spack add deephyper +sdv # SDV support
    $ spack add deephyper +jax-cpu # Jax support
    $ spack add deephyper +tf-keras2 # Tensorflow and Keras support
    $ spack add deephyper +torch # PyTorch support

Install Storage and Parallel Backends

.. code-block:: console
    
    $ spack add deephyper +mpi # MPI support for MPICommEvaluator
    $ spack add deephyper +ray # Ray support for RayEvaluator
    $ spack add deephyper +redis # Redis/RedisJSON/py-redis support for RedisStorage and Distributed Search

Install Dev tools

.. code-block:: console
    
    $ spack add deephyper +dev

Run :code:`spack install` again after adding any additonal features
