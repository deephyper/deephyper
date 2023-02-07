Local
*****

DeepHyper installation requires ``Python>=3.7``. By default, only hyperparameter search features will be installed. For other features, see below.

.. note:: All other packages required for your model need to be installed in the same environment.

.. _local-conda-environment:

Conda environment
=================

This installation procedure shows you how to create your own Conda virtual environment and install DeepHyper in it. 

Linux
-----

Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, and create the ``dh`` environment:

.. code-block:: console

    $ conda create -n dh python=3.8 -y
    $ conda activate dh
    $ conda install gxx_linux-64 gcc_linux-64

Finally install DeepHyper in the previously created ``dh`` environment:

.. code-block:: console

    $ pip install pip --upgrade
    $ pip install deephyper["analytics"] # install the core packages. 

MacOS
-----

Install Xcode command line tools:

.. code-block:: console

    xcode-select --install

Check you current platform:

.. code-block:: console

    python3 -c "import platform; print(platform.platform());"

x86_64
######

If your architecture is ``x86_64`` install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, and create the ``dh`` environment:

.. code-block:: console

    $ conda create -n dh python=3.8 -y
    $ conda activate dh

Then install DeepHyper in the previously created ``dh`` environment:

.. code-block:: console

    $ pip install pip --upgrade
    $ pip install deephyper["analytics"]


arm64
#####

If your architecture is  ``arm64`` download `MiniForge <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>`_ then install it:

.. code-block:: console

    chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
    sh ~/Downloads/Miniforge3-MacOSX-arm64.sh

After installing MiniForge clone the DeepHyper repo and install the package:

.. code-block:: console

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    conda env create -f install/environment.macOS.arm64.yml
    


Jupyter Notebooks
=================

To create a custom Jupyter kernel run the following from your activated Conda environment:

.. code-block:: console

    $ python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Now when you open a Jupyter notebook the ``Python (deephyper)`` kernel will be available.


.. _local-docker-installation:

Docker Image (CPU)
==================

A `Docker <https://www.docker.com>`_ image with DeepHyper is provided. Assuming `Docker <https://www.docker.com>`_ is installed on the system you are using you can access the image with the following commands:


.. code-block:: console

    $ docker pull ghcr.io/deephyper/deephyper:0.3.3
    $ docker run -i -t ghcr.io/deephyper/deephyper:0.3.3 /bin/bash

.. _local-dev-installation:

Developer Installation
======================

Follow the :ref:`local-conda-environment` installation and replace ``pip install deephyper[analytics]`` by:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper.git
    $ cd deephyper/ && git checkout develop
    $ pip install -e ".[dev,analytics]"

Other features.
=================

# MPI features
Deephyper can use MPI to paralellze evaluations during the excution of search algorithms. To use MPI features, an implementation of the MPI interface, such as `openmpi` is required.

.. code-block:: console

    $ pip install mpi4py

# Ray features
Deephyper can use `Ray <https://docs.ray.io/en/latest/ray-overview/installation.html>`_ to parallelize evaluations during the excution of search algorithms. Ray is also required for `deephyper.ensemble` library. To use Ray features, the `ray` package must be installed 

.. code-block:: console

    $ pip install ray


To enable other features:

.. code-block:: console
    
    $ pip install "deephyper[default]" # Install HPS, NAS, AutoDEUQ, Transfer-Learning for HPS, LCE Stopper,
    $ pip install "deephyper[hps]" # Install Hyperparameter Search.
    $ pip install "deephyper[nas]" # Install Neural Architecture Search.
    $ pip install "deephyper[autodeuq]" # Install Automated Deep Ensemble with Uncertainty Quantification.
    $ pip install "deephyper[jax-cpu]" # Install JAX with CPU support for Learning Curve Extrapolation Stopper.
    $ pip install "deephyper[jax-cuda]" # Install JAX with GPU (cuda) support for Learning Curve Extrapolation Stopper.
    $ pip install "deephyper[automl]" # Install Automated Machine Learning features.
    $ pip install "deephyper[mpi]" # Install MPI features for MPICommEvaluator.
    $ pip install "deephyper[ray]" # Install Ray features for RayEvaluator.
    $ pip install "deephyper[redis]" # Install Redis Client for RedisStorage with Distributed Search.
    $ pip install "deephyper[analytics]" # Install Analytics tools (for developers).
    $ pip install "deephyper[dev]" # Install Developer Stack (tests, documentation, etc...)