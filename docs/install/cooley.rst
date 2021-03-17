Cooley (ALCF)
*************

`Cooley <https://www.alcf.anl.gov/user-guides/cooley>`_ is a GPU cluster at Argonne Leadership Computing Facility (ALCF). It has a total of 126 compute nodes; each node has 12 CPU cores and one NVIDIA Tesla K80 dual-GPU card.


1. **Install Postgresql**

* Download the software::

    wget http://get.enterprisedb.com/postgresql/postgresql-9.6.13-4-linux-x64-binaries.tar.gz


* Open the downloaded archive::

    tar -xvf postgresql-9.6.13-4-linux-x64-binaries.tar.gz


* Add the bin path to your ``bashrc``::

    echo "export PATH=$PWD/pgsql/bin:$PATH" >> ~/.bashrc; source ~/.bashrc


* Check the installation is working properly::

    pg_ctl --version


2. **Get anaconda from SoftEnv**::

    soft add +anaconda3-4.0.0

3. **Get cuda from SoftEnv**::

    soft add +cuda-10.0

.. note::

    Now you may add the following to your ``~/.soft``::

        # ~/.soft
        +cuda-10.0
        +anaconda3-4.0.0
        @default

4. **Install deephyper**

* Create a new conda environment::

    conda create -n dh-env python=3.6

* Activate your freshly created conda environment::

    source activate dh-env

* Install deephyper from pypi::

    pip install deephyper


.. WARNING::
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
