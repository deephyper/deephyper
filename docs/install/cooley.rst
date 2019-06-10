Cooley installation
*******************

1. **Install Postgresql**

* Download the software::

    wget http://get.enterprisedb.com/postgresql/postgresql-9.6.13-4-linux-x64-binaries.tar.gz


* Open the downloaded archive::

    tar -xvf postgresql-9.6.13-4-linux-x64-binaries.tar.gz


* Add the bin path to your ``bashrc``::

    echo "export PATH=$PWD/pgsql/bin:$PATH" >> ~/.bashrc; source ~/.bashrc


* Check the installation is working properly::

    pg_ctl --version


2. **Now install conda**

* Download the software::

    wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

* Add execute rights to the installation script::

    chmod +x Anaconda3-2019.03-Linux-x86_64.sh

* Run the installation script::

    sh Anaconda3-2019.03-Linux-x86_64.sh

.. note::

    It is better to use your project folder because of bigger disk space.

    ::

        Anaconda3 will now be installed into this location:
        /home/regele/anaconda3

        - Press ENTER to confirm the location
        - Press CTRL-C to abort the installation
        - Or specify a different location below

        [/home/regele/anaconda3] >>> /projects/datascience/regele

3. **Get cuda library**

* Add the cuda lib to your environment::

    export LD_LIBRARY_PATH=/soft/visualization/cuda-10.0/lib64:$LD_LIBRARY_PATH

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
