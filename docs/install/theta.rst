Theta (Argonne LCF)
*******************

`Theta <https://www.alcf.anl.gov/theta>`_ is a 11.69 petaflops system based on the second-generation Intel Xeon Phi processor at Argonne Leadership Computing Facility (ALCF). It serves as a stepping stone to the ALCF's next leadership-class supercomputer, Aurora.
Theta is a massively parallel, many-core system based on Intel processors and interconnect technology, a new memory space, and a Lustre-based parallel file system, all integrated by Cray’s HPC software stack.

.. _theta-module-installation:

Already installed module
========================

This installation procedure shows you how to access the installed DeepHyper module on Theta. After logging in Theta, to access Deephyper run the following commands:

.. code-block:: console

    $ module load conda/2021-09-22
    $ conda activate base

Then to verify the installation do:

.. code-block:: console

    $ python
    >>> import deephyper
    >>> deephyper.__version__
    '0.3.0'

.. _theta-conda-environment:

Conda environment
=================

This installation procedure shows you how to create your own Conda virtual environment and install DeepHyper in it.

.. admonition:: Storage/File Systems
    :class: dropdown, important

    It is important to run the following commands from the appropriate storage space because some features of DeepHyper can generate a consequent quantity of data such as model checkpointing. The storage spaces available at the ALCF are:

    - ``/lus/grand/projects/``
    - ``/lus/eagle/projects/``
    - ``/lus/theta-fs0/projects/``

    For more details refer to `ALCF Documentation <https://www.alcf.anl.gov/support-center/theta/theta-file-systems>`_.

After logging in Theta, go to your project folder (replace ``PROJECTNAME`` by your own project name):

.. code-block:: console

    $ cd /lus/theta-fs0/projects/PROJECTNAME

Then create the ``dhknl`` environment:

.. code-block:: console

    $ module load miniconda-3
    $ conda create -p dhknl python=3.8 -y
    $ conda activate dhknl/

It is then required to have the following additionnal dependencies:

.. code-block:: console

    $ conda install gxx_linux-64 gcc_linux-64 -y

Finally install DeepHyper in the previously created ``dhknl`` environment:

.. code-block:: console

    $ pip install pip --upgrade
    $ # DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
    $ pip install deephyper[analytics]
    $ conda install tensorflow -c intel -y


.. note::
    Horovod can be installed to use data-parallelism during the evaluations of DeepHyper. To do so use ``pip install deephyper[analytics,hvd]`` while or after installing.


Jupyter Notebooks
=================

To use Jupyter notebooks on Theta go to `Theta Jupyter <https://jupyter.alcf.anl.gov/theta>`_ and use your regular authentication method. The `Jupyter Hub tutorial <https://www.alcf.anl.gov/user-guides/jupyter-hub>`_ from Argonne Leadership Computing Facility might help you in case of troubles.

To create a custom Jupyter kernel run the following from your activated Conda environment:

.. code-block:: console

    $ python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"


Now when openning a notebook from Jupyter Hub at ALCF make sure to use the ``Python (deephyper)`` kernel before executing otherwise you will not have all required dependencies.


Developer installation
======================

Follow the :ref:`theta-conda-environment` installation and replace ``pip install deephyper[analytics]`` by:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper.git
    $ cd deephyper/ && git checkout develop
    $ pip install -e ".[dev,analytics]"