Local
*****

DeepHyper installation requires ``Python>=3.7 and <3.9``.

.. _local-conda-environment:

Conda environment
=================

This installation procedure shows you how to create your own Conda virtual environment and install DeepHyper in it. After installing Anaconda or Miniconda, creates the ``dh`` environment:

.. code-block:: console

    conda create -n dh python=3.8
    conda activate dh

For Linux-based systems, it is then required to have the following additionnal dependencies:

.. code-block:: console

    apt-get install build-essential
    # or
    conda install gxx_linux-64 gcc_linux-64

Then install DeepHyper in the previously created ``dh`` environment:

.. code-block:: console

    pip install deephyper["analytics"]


Jupyter Notebooks
=================

To create a custom Jupyter kernel run the following from your activated Conda environment:

.. code-block:: console

    python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Now when you will open a Jupyter notebook the ``Python (deephyper)`` kernel will be available.

.. _local-dev-installation:

Developer Installation
======================

Follow the :ref:`local-conda-environment` installation and replace ``pip install deephyper[analytics]`` by:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper.git
    $ cd deephyper/ && git checkout develop
    $ pip install -e ".[dev,analytics]"


