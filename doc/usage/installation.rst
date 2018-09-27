Installation
************

Theta
=====

Cooley
======

::

    soft add +anaconda

::

    conda create --name deephyper-cooley intelpython3_core  python=3.6
    source activate deephyper-cooley
    conda install h5py scikit-learn pandas mpi4py

    conda config --add channels conda-forge
    conda install absl-py
    conda install keras scikit-optimize
    conda install xgboost deap
    conda install -c anaconda tensorflow tensorflow-gpu keras keras-gpu
    conda install jinja2

    conda install psycopg2
    cd hpc-edge-service
    pip install -e .

    pip install filelock
    pip install git+https://github.com/tkipf/keras-gcn.git


::

    wget https://get.enterprisedb.com/postgresql/postgresql-10.4-1-linux-x64-binaries.tar.gz
