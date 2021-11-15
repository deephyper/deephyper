Local
*****

DeepHyper installation requires ``Python>=3.7 and <3.10``.

.. _local-conda-environment:

Conda environment
=================

This installation procedure shows you how to create your own Conda virtual environment and install DeepHyper in it. After installing `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, create the ``dh`` environment:

.. code-block:: console

    $ conda create -n dh python=3.8
    $ conda activate dh

For **Linux-based** systems, it is then required to have the following additionnal dependencies (skip this step if you have **MacOS**):

.. code-block:: console

    $ apt-get install build-essential
    $ # or
    $ conda install gxx_linux-64 gcc_linux-64

Finally install DeepHyper in the previously created ``dh`` environment:

.. code-block:: console

    $ pip install pip --upgrade
    $ # DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
    $ pip install deephyper["analytics"]


Jupyter Notebooks
=================

To create a custom Jupyter kernel run the following from your activated Conda environment:

.. code-block:: console

    $ python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Now when you open a Jupyter notebook the ``Python (deephyper)`` kernel will be available.


Docker Image (CPU)
==================

A `Docker <https://www.docker.com>`_ image is provided with DeepHyper. Assuming Docker is installed on the system you are using you can access the image with the following commands:

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