.. _install-pip:

Install with pip
****************

DeepHyper installation requires ``Python>=3.10``.

.. warning:: All packages required for your application need to be installed in the same environment.

DeepHyper is available on `PyPI <https://pypi.org/project/deephyper/>`_ and can be installed with ``pip`` on Linux, macOS by downloading pre-built binaries (wheels) from PyPI:

.. code-block:: console

    $ # Core set of features
    $ pip install "deephyper[core]" # <=> "deephyper[jax-cpu,torch]"
    
    $ # Isolated features
    $ pip install "deephyper"             # install hyperparameter optimization (HPO)
    $ pip install "deephyper[jax-cpu]"    # install JAX with CPU support for Learning Curve Extrapolation Stopper
    $ pip install "deephyper[jax-cuda]"   # install JAX with GPU (Cuda) support for Learning Curve Extrapolation Stopper
    $ pip install "deephyper[torch]"      # install with Pytorch support
    
    $ # For developers (Tests, Documentation, etc...)
    $ pip install "deephyper[dev]"        # install developer stack (tests, documentation, etc.)


In Bayesian optimization, the Mondrian Forest surrogate model can be used. This model provides better uncertainty estimates used in the acquisition function. To install the Mondrian Forest surrogate model, you need to install the modified ``scikit-garden`` package from our repository. This package is not available on PyPI but can be installed through ``pip`` from the GitHub repository:

.. code-block:: console

    $ pip install "scikit-garden @ git+https://github.com/deephyper/scikit-garden@master"
    

Distributed Computation
=======================

DeepHyper supports distributed computation with different backends. ``MPI`` demonstrated better scaling capabilities but ``Ray`` is more flexible and easier to use on smaller scales. ``Redis`` is required for decentralized search (i.e., different search instances communicating with each other through a shared database). The following commands install the "client" or "python binding" of the corresponding libraries but for ``MPI`` and ``Redis`` you will also need to install the corresponding libraries on your system prior to the ``pip ...`` command.

.. code-block:: console

    $ pip install "deephyper[mpi]"            # install python bindings for MPI
    $ pip install "deephyper[ray]"            # install Ray
    $ pip install "deephyper[redis]"          # install Redis client
    $ pip install "deephyper[redis-hiredis]"  # install Redis client with Hiredis for better performance


Redis
-----

For ``Redis``, the `redis-stack <https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/>`_ should be installed.

MPI
---

For ``MPI`` we advice to follow the `MPI official installation guide <https://www.open-mpi.org/faq/?category=building>`_ to install the client/server features. But, in many computing facilities an ``MPI`` installation will already be provided or it can also be installed through a package manager (e.g., ``apt`` or ``brew``).