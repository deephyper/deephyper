.. _install-pip:

Install DeepHyper with pip
**************************

DeepHyper installation requires ``Python>=3.7``.

.. warning:: All packages required for your application need to be installed in the same environment.

DeepHyper is available on `PyPI <https://pypi.org/project/deephyper/>`_ and can be installed with ``pip`` on Linux, macOS by downloading pre-built binaries (wheels) from PyPI:

.. code-block:: console

    $ # Default set of features (HPS, NAS, AutoDEUQ, Transfer-Learning and LCE Stopper) 
    $ pip install "deephyper[default]" # <=> "deephyper[hps,nas,autodeuq,hps-tl,jax-cpu]"
    
    $ # Isolated features
    $ pip install "deephyper[hps]" # Install Hyperparameter Search.
    $ pip install "deephyper[nas]" # Install Neural Architecture Search.
    $ pip install "deephyper[autodeuq]" # Install Automated Deep Ensemble with Uncertainty Quantification.
    $ pip install "deephyper[hps-tl]" # Install Transfer-Learning for HPS.
    $ pip install "deephyper[jax-cpu]" # Install JAX with CPU support for Learning Curve Extrapolation Stopper.
    $ pip install "deephyper[jax-cuda]" # Install JAX with GPU (cuda) support for Learning Curve Extrapolation Stopper.
    $ pip install "deephyper[automl]" # Install Automated Machine Learning features.
    
    $ # Others
    $ pip install "deephyper[analytics]" # Install Analytics tools (for developers).
    $ pip install "deephyper[dev]" # Install Developer Stack (tests, documentation, etc...)

Distributed Computation
=======================

DeepHyper supports distributed computation with different backends. ``MPI`` demonstrated better scaling capabilities but ``Ray`` is more flexible and easier to use on smaller scales. ``Redis`` is required for distributed search (i.e., different search instances communicating with each other through a shared database). The following command install the "client" or "python binding" of the corresponding libraries but for ``MPI`` and ``Redis`` you will also need to install the corresponding libraries on your system prior to the ``pip ...`` command.

.. code-block:: console

    $ pip install "deephyper[mpi]" # Install MPI features for MPICommEvaluator.
    $ pip install "deephyper[ray]" # Install Ray features for RayEvaluator.
    $ pip install "deephyper[redis]" # Install Redis Client for RedisStorage with Distributed Search.

For ``Redis`` we advice to follow the `Redis official installation guide <https://redis.io/topics/quickstart>`_ to install the client/server features. Then, the ``RedisJson`` also needs to be installed by following the `Redis JSON official installation guide <https://redis.io/docs/stack/json/>`_.

For ``MPI`` we advice to follow the `MPI official installation guide <https://www.open-mpi.org/faq/?category=building>`_ to install the client/server features. But, in many centers an ``MPI`` installation will already be provided or it can also be installed through a package manager (e.g., ``apt`` or ``brew``).