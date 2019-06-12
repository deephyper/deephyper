Theta installation
******************

.. _theta-user-installation:

User installation
=================

When you are a user deephyper can be directly installed as a module on Theta.

::

    module load deephyper

.. note::
    You might put
    ``module load deephyper`` in your ``~/.bashrc`` if you want to use
    *deephyper* in all new session.

Developer installation
======================

1. Load the miniconda module::

    module load miniconda-3.6/conda-4.5.12

.. note::
    The miniconda module is using the `Intel channel <https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda>`_ which has optimized wheels using
    MKL/DNN (available on KNL nodes with Xeon Phi CPU) for some packages.

2. Load the balsam module::

    module load balsam/0.3


3. Create a virtual environment for your deephyper installation as a developer::

    python -m venv --system-site-packages deephyper-dev-env

4. Activate this freshly created virtual environment::

    source deephyper-dev-env/bin/activate

.. note::
    To activate your virtualenv easier in the future you can define an alias
    in your ``~/.bashrc`` such as ``alias act="source ~/deephyper-dev-env/bin/activate"``. Now you will clone deephyper sources and install it with ``pip``

5. Clone the deephyper repo::

    git clone https://github.com/deephyper/deephyper.git

6. Go to the root directory of the repo::

    cd deephyper/


7. Switch to the develop branch::

    git checkout develop

8. Install the package::

    pip install -e .

