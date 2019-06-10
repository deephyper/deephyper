Cooley installation
*******************

Using SoftEnv
=============

1. **Get cuda from SoftEnv**::

    soft add +cuda-10.0

2. **Get anaconda from SoftEnv**::

    soft add +anaconda3-4.0.0

.. note::

    You may add the following to your ``~/.soft``::

        # ~/.soft
        +cuda-10.0
        +anaconda3-4.0.0
        @default

3. **Install deephyper**

* Specify youo want the deephyper-gpu installation::

    echo "export DH_GPU=true" >> ~/.bashrc; source ~/.bashrc


* Create a new conda environment::

    conda create -n dh-env python=3.6

* Activate your freshly created conda environment::

    source activate dh-env

* Install deephyper from pypi::

    pip install deephyper

By Hand
=======
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

* Specify youo want the deephyper-gpu installation::

    echo "export DH_GPU=true" >> ~/.bashrc; source ~/.bashrc


* Create a new conda environment::

    conda create -n dh-env python=3.6

* Activate your freshly created conda environment::

    source activate dh-env

* Install deephyper from pypi::

    pip install deephyper


.. WARNING::

    ::

        # Theta Specific
        if [[ $HOSTNAME = *"theta"* ]];
        then
            source ~/.bashrc_theta
        # Cooley Specific
        else
            source ~/.bashrc_cooley
        fi
