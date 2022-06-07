ThetaGPU (Argonne LCF)
**********************

`ThetaGPU <https://www.alcf.anl.gov/theta>`_  is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes at Argonne Leadership Computing Facility (ALCF). See the `documentation <https://argonne-lcf.github.io/ThetaGPU-Docs/>`_ of ThetaGPU from the Datascience group at Argonne National Laboratory for more information. The system documentation from the ALCF can be accessed `here <https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu>`_.

.. _thetagpu-module-installation:

Already installed module
========================

This installation procedure shows you how to access the installed DeepHyper module on ThetaGPU. After logging in Theta, connect to a ThetaGPU service node:

.. code-block:: console

    $ ssh thetagpusn1

Then, to access Deephyper run the following commands:

.. code-block:: console

    $ module load conda/2021-09-22
    $ conda activate base

Finally, to verify the installation do:

.. code-block:: console

    $ python
    >>> import deephyper
    >>> deephyper.__version__
    '0.3.0'

.. _thetagpu-conda-environment:

Conda environment
=================

This installation procedure shows you how to create your own Conda virtual environment and install DeepHyper in it.

.. admonition:: Storage/File Systems
    :class: dropdown, important

    It is important to run the following commands from the appropriate storage space because some features of DeepHyper can generate a consequante quantity of data such as model checkpointing. The storage spaces available at the ALCF are:

    - ``/lus/grand/projects/``
    - ``/lus/eagle/projects/``
    - ``/lus/theta-fs0/projects/``

    For more details refer to `ALCF Documentation <https://www.alcf.anl.gov/support-center/theta/theta-file-systems>`_.

After logging in Theta, locate yourself on one of the ThetaGPU service node (``thetagpusnX``) and move to your project folder (replace ``PROJECTNAME`` by your own project name):

.. code-block:: console

    $ ssh thetagpusn1
    $ cd /lus/theta-fs0/projects/PROJECTNAME

Then create the ``dhgpu`` environment:

.. code-block:: console

    $ module load conda/2021-11-30
    $ module load openmpi/openmpi-4.0.5
    $ conda create -p dhgpu --clone base
    $ conda activate dhgpu/

Install DeepHyper in the previously created ``dhgpu`` environment:

.. code-block:: console

    $ pip install pip --upgrade
    $ # DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
    $ pip install deephyper["analytics"]

Finally install mpi4py in the previously created ``dhgpu`` environment:

.. code-block:: console

    $ git clone https://github.com/mpi4py/mpi4py.git
    $ cd mpi4py/
    $ MPICC=mpicc python setup.py install
    $ cd ..

Developer installation
======================

Follow the :ref:`thetagpu-conda-environment` installation and replace ``pip install deephyper[analytics]`` by:

.. code-block:: console

    $ git clone -b develop https://github.com/deephyper/deephyper.git
    $ pip install -e "deephyper[dev,analytics]"


Internet Access
===============

If the node you are on does not have outbound network connectivity, set the following to access the proxy host:

.. code-block:: console

    $ export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
    $ export https_proxy=http://proxy.tmi.alcf.anl.gov:3128