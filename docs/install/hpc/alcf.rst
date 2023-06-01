Argonne Leadership Computing Facility (ALCF)
********************************************

Polaris
=======

`Polaris <https://www.alcf.anl.gov/polaris>`_ is a 44 petaflops system based on the HPE Appolo Gen10+ platform. It is composed of heterogenous computing capabilities with 1 AMD EPYC "Milan" processor and 4 NVIDIA A100 GPUs per node.

.. _polaris-module-installation:

Already installed module
------------------------

This installation procedure shows you how to access the installed DeepHyper module on Polaris. After logging in Polaris, to access Deephyper run the following commands:

.. code-block:: console

    $ module load conda/2022-09-08
    $ conda activate base

Then to verify the installation do:

.. code-block:: console

    $ python
    >>> import deephyper
    >>> deephyper.__version__
    '0.4.2'

.. warning:: The ``deephyper`` installation provided in the conda module is not always up to date. If you need a more recent version of DeepHyper, please refer to the :ref:`conda-environment` installation procedure.

.. _polaris-from-source:

This script creates a conda environment activation script ``activate-dhenv.sh`` in the build directory, which can be sourced
to activate the created environment, and a ``redis.conf`` file, which should be referenced when starting a Redis storage server.

Installation from source
------------------------

This installation procedure shows you how to build DeepHyper from source on Polaris. This installation, will provide DeepHyper's default set of features with MPI backend for the `Evaluator` and the Redis backend for the `Storage`. After logging in Polaris, the following script can be executed from a `build` directory:

.. literalinclude:: ../../../install/alcf/polaris.sh
    :language: bash
    :caption: **file**: ``install/alcf/polaris.sh``
    :linenos:

This script creates a conda environment activation script ``activate-dhenv.sh`` in the build directory, which can be sourced
to activate the created environment, and a ``redis.conf`` file, which should be referenced when starting a Redis storage server.

Theta
=====

`Theta <https://www.alcf.anl.gov/theta>`_ is a 11.69 petaflops system based on the second-generation Intel Xeon Phi processor at Argonne Leadership Computing Facility (ALCF). It serves as a stepping stone to the ALCF's next leadership-class supercomputer, Aurora.
Theta is a massively parallel, many-core system based on Intel processors and interconnect technology, a new memory space, and a Lustre-based parallel file system, all integrated by Cray’s HPC software stack.

.. _theta-module-installation:

Already installed module
------------------------

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
-----------------

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
-----------------

To use Jupyter notebooks on Theta go to `Theta Jupyter <https://jupyter.alcf.anl.gov/theta>`_ and use your regular authentication method. The `Jupyter Hub tutorial <https://www.alcf.anl.gov/user-guides/jupyter-hub>`_ from Argonne Leadership Computing Facility might help you in case of troubles.

To create a custom Jupyter kernel run the following from your activated Conda environment:

.. code-block:: console

    $ python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"


Now when openning a notebook from Jupyter Hub at ALCF make sure to use the ``Python (deephyper)`` kernel before executing otherwise you will not have all required dependencies.


ThetaGPU
========

`ThetaGPU <https://www.alcf.anl.gov/theta>`_  is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes at Argonne Leadership Computing Facility (ALCF). See the `documentation <https://argonne-lcf.github.io/ThetaGPU-Docs/>`_ of ThetaGPU from the Datascience group at Argonne National Laboratory for more information. The system documentation from the ALCF can be accessed `here <https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu>`_.

.. _thetagpu-module-installation:

Already installed module
------------------------

This installation procedure shows you how to access the installed DeepHyper module on ThetaGPU. It may be useful to wrap these commands in this ``activate-dhenv.sh`` script :

.. code-block:: bash
    :caption: **file**: ``activate-dhenv.sh``

    #!/bin/bash

    . /etc/profile

    module load conda/2022-07-01
    conda activate base

To then effectively call this activation script in your scripts, you can use ``source ...``, here is an exemple to test the good activation of the conda environment (replace the ``$PROJECT_NAME`` with your project, e-g: ``#COBALT -A datascience``) :

.. code-block:: bash
    :caption: **file**: ``job-test-activation.sh``

    #!/bin/bash
    #COBALT -q single-gpu
    #COBALT -n 1
    #COBALT -t 20
    #COBALT -A $PROJECT_NAME
    #COBALT --attrs filesystems=home,theta-fs0,grand,eagle

    source activate-dhenv.sh
    python -c "import deephyper; print(f'DeepHyper version: {deephyper.__version__}')"

You should obtain a ``DeepHyper version: x.x.x`` in the output cobaltlog file from this job after submitting it with :

.. code-block:: console

    $ qsub-gpu job-test-activation.sh

.. _thetagpu-conda-environment:

Conda environment
-----------------

This installation procedure shows you how to create your own Conda virtual environment and install DeepHyper in it.

.. admonition:: Storage/File Systems
    :class: dropdown, important

    It is important to run the following commands from the appropriate storage space because some features of DeepHyper can generate a consequante quantity of data such as model checkpointing. The storage spaces available at the ALCF are:

    - ``/lus/grand/projects/``
    - ``/lus/eagle/projects/``
    - ``/lus/theta-fs0/projects/``

    For more details refer to `ALCF Documentation <https://www.alcf.anl.gov/support-center/theta/theta-file-systems>`_.

As this procedure needs to be performed on ThetaGPU, we will directly execute it in this ``job-install-dhenv.sh`` submission script (replace the ``$PROJECT_NAME`` with the name of your project allocation, e-g: ``#COBALT -A datascience``):

.. code-block:: bash
    :caption: **file**: ``job-install-dhenv.sh``

    #!/bin/bash
    #COBALT -q single-gpu
    #COBALT -n 1
    #COBALT -t 60
    #COBALT -A $PROJECT_NAME
    #COBALT --attrs filesystems=home,theta-fs0,grand

    . /etc/profile

    # create the dhgpu environment:
    module load conda/2022-07-01

    conda create -p dhenv --clone base -y
    conda activate dhenv/

    # install DeepHyper in the previously created dhgpu environment:
    pip install pip --upgrade
    pip install deephyper["analytics"]

Then submit this job by executing the following command :

.. code-block:: console
    
    $ qsub-gpu job-test-activation.sh

Once this job is finished you can test the good installation by creating this ``activate-dhenv.sh`` script and submitting the ``job-test-activation.sh`` job from :ref:`thetagpu-module-installation`:

.. code-block:: bash
    :caption: **file**: ``activate-dhenv.sh``

    #!/bin/bash

    . /etc/profile

    module load conda/2022-07-01
    conda activate dhenv/

mpi4py installation
~~~~~~~~~~~~~~~~~~~

You might need to additionaly install ``mpi4py`` to your environment in order to use functionnalities such as the ``"mpicomm"`` evaluator, you simply need to add this after ``pip install deephyper["analytics"]`` :

.. code-block:: console

    $ git clone https://github.com/mpi4py/mpi4py.git
    $ cd mpi4py/
    $ MPICC=mpicc python setup.py install
    $ cd ..


Internet Access
~~~~~~~~~~~~~~~

If the node you are on does not have outbound network connectivity, set the following to access the proxy host:

.. code-block:: console

    $ export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
    $ export https_proxy=http://proxy.tmi.alcf.anl.gov:3128


Cooley
======

.. warning::

    This page is outdated and refers to last known installation procedures for Cooley.

`Cooley <https://www.alcf.anl.gov/user-guides/cooley>`_ is a GPU cluster at Argonne Leadership Computing Facility (ALCF). It has a total of 126 compute nodes; each node has 12 CPU cores and one NVIDIA Tesla K80 dual-GPU card.

Before installating DeepHyper, go to your project folder::

    cd /lus/theta-fs0/projects/PROJECTNAME
    mkdir cooley && cd cooley/

DeepHyper can be installed on Cooley by following these commands::

    git clone https://github.com/deephyper/deephyper.git --depth 1
    ./deephyper/install/cooley.sh

Then, restart your session.

.. warning::
    You will note that a new file ``~/.bashrc_cooley`` was created and sourced in the ``~/.bashrc``. This is to avoid conflicting installations between the different systems available at the ALCF.

.. note::
    To test you installation run::

        ./deephyper/tests/system/test_cooley.sh


A manual installation can also be performed with the following set of commands::

    # Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O miniconda.sh
    bash $PWD/miniconda.sh -b -p $PWD/miniconda
    rm -f miniconda.sh

    # Install Postgresql
    wget http://get.enterprisedb.com/postgresql/postgresql-9.6.13-4-linux-x64-binaries.tar.gz -O postgresql.tar.gz
    tar -xf postgresql.tar.gz
    rm -f postgresql.tar.gz

    # adding Cuda
    echo "+cuda-10.2" >> ~/.soft.cooley
    resoft

    source $PWD/miniconda/bin/activate

    # Create conda env for DeepHyper
    conda create -p dh-cooley python=3.8 -y
    conda activate dh-cooley/
    conda install gxx_linux-64 gcc_linux-64 -y
    # DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
    pip install deephyper[analytics,balsam]
    conda install tensorflow-gpu

.. warning::
    The same ``.bashrc`` is used both on Theta and Cooley. Hence adding a ``module load`` instruction to the ``.bashrc`` will not work on Cooley. In order to solve this issue you can add a specific statement to your ``.bashrc`` file and create separate *bashrc* files for Theta and Cooley and use them as follows.
    ::

        # Theta Specific
        if [[ $HOSTNAME = *"theta"* ]];
        then
            source ~/.bashrc_theta
        # Cooley Specific
        else
            source ~/.bashrc_cooley
        fi
