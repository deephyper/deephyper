Cooley (ALCF)
*************

.. warning::

    This page is outdated and refers to last known installation procedures for Cooley.

`Cooley <https://www.alcf.anl.gov/user-guides/cooley>`_ is a GPU cluster at Argonne Leadership Computing Facility (ALCF). It has a total of 126 compute nodes; each node has 12 CPU cores and one NVIDIA Tesla K80 dual-GPU card.


User installation
=================

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
