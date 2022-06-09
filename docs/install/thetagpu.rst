ThetaGPU (Argonne LCF)
**********************

`ThetaGPU <https://www.alcf.anl.gov/theta>`_  is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes at Argonne Leadership Computing Facility (ALCF). See the `documentation <https://argonne-lcf.github.io/ThetaGPU-Docs/>`_ of ThetaGPU from the Datascience group at Argonne National Laboratory for more information. The system documentation from the ALCF can be accessed `here <https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu>`_.

.. _thetagpu-module-installation:

Already installed module
========================

This installation procedure shows you how to access the installed DeepHyper module on ThetaGPU. It may be useful to wrap these commands in this ``activate-dhenv.sh`` script :

.. code-block:: bash
    :caption: **file**: ``activate-dhenv.sh``

    #!/bin/bash

    . /etc/profile

    module load conda/2021-09-22
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
=================

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
    module load conda/2021-11-30

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

    module load conda/2021-09-22
    conda activate dhenv/

mpi4py installation
-------------------

You might need to additionaly install ``mpi4py`` to your environment in order to use functionnalities such as the ``"mpicomm"`` evaluator, you simply need to add this after ``pip install deephyper["analytics"]`` :

.. code-block:: console

    $ git clone https://github.com/mpi4py/mpi4py.git
    $ cd mpi4py/
    $ MPICC=mpicc python setup.py install
    $ cd ..

Developer installation
======================

Follow the :ref:`thetagpu-conda-environment` installation and replace ``pip install deephyper[analytics]`` by:

.. code-block:: bash

    git clone -b develop https://github.com/deephyper/deephyper.git
    pip install -e "deephyper[dev,analytics]"

Internet Access
===============

If the node you are on does not have outbound network connectivity, set the following to access the proxy host:

.. code-block:: console

    $ export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
    $ export https_proxy=http://proxy.tmi.alcf.anl.gov:3128