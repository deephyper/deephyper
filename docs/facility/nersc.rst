National Energy Research Scientific Computing (NERSC)
*****************************************************

Perlmutter
==========

`Perlmutter <https://docs.nersc.gov/systems/perlmutter/architecture/>`_, a HPE Cray supercomputer at NERSC, is a heterogeneous system with both GPU-accelerated and CPU-only nodes. Phase 1 of the installation is made up of 12 GPU-accelerated cabinets housing over 1,500 nodes. Phase 2 adds 12 CPU cabinets with more than 3,000 nodes. Each GPU node of Perlmutter has 4x NVIDIA A100 GPUs. 

.. perlmutter-conda-environment:

Conda environment
-----------------

For connecting to Perlmutter, check `documentation <https://docs.nersc.gov/systems/perlmutter/#connecting-to-perlmutter>`_. One can also configure SSH according to the `instructions <https://docs.nersc.gov/connect/mfa/#ssh-configuration-file-options>`_. To connect to Perlmutter via terminal, use:

.. code-block:: console

    $ ssh <username>@perlmutter-p1.nersc.gov


Load the pre-installed modules available on Perlmutter:

.. code-block:: console

    $ module load PrgEnv-nvidia cudatoolkit python
    $ module load cudnn/8.2.0

Then, create a conda environment:

.. code-block:: console

    $ conda create -n dh python=3.9 -y
    $ conda activate dh
    $ conda install gxx_linux-64 gcc_linux-64


Now a crucial step is to install CUDA aware mpi4py, following the instructions given in the `mpi4py documentation <https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#building-cuda-aware-mpi4py>`_:

.. code-block:: console

    $ MPICC="cc -target-accel=nvidia80 -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

Perlmutter (as of November 2022) lacks support for a few mpi4py functionalities such as the MPI_Mprobe. More details are provided on the NERSC issues `page <https://docs.nersc.gov/current/#ongoing-issues>`_. The workaround is to disable using matched probes to receive objects as follows: 

.. code-block:: console

    $ export MPI4PY_RC_RECV_MPROBE='False'


Then we install some of the other dependencies:
    
.. code-block:: console

    $ pip install tensorflow==2.9.2
    $ pip install kiwisolver
    $ pip install cycler
    $ pip install matplotlib
    $ pip install progressbar2
    $ pip install networkx[default]

Finally we install deephyper:

.. code-block:: console

    $ pip install deephyper


Finally you can verify the version of the installed deephyper package:

.. code-block:: console

    $ python
    >>> import deephyper
    >>> deephyper.__version__

Do not forget to reload the installed dependencies each time you want to use DeepHyper:

.. code-block:: bash

    module load PrgEnv-nvidia cudatoolkit python
    module load cudnn/8.2.0
    source /global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11/etc/profile.d/conda.sh
    export MPI4PY_RC_RECV_MPROBE='False'
    conda activate dh
    
    
