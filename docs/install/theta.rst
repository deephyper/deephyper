Theta
******

`Theta <https://www.alcf.anl.gov/theta>`_ is a 11.69 petaflops system based on the second-generation Intel Xeon Phi processor at Argonne Leadership Computing Facility (ALCF).
It serves as a stepping stone to the ALCF's next leadership-class supercomputer, Aurora.
Theta is a massively parallel, many-core system based on Intel processors and interconnect technology, a new memory search_space,
and a Lustre-based parallel file system, all integrated by Cray’s HPC software stack.

.. _theta-user-installation:

User installation
=================

DeepHyper is already installed in Theta and can be directly loaded as a module as follows.

::

    module load deephyper

.. note::
    You might put
    ``module load deephyper`` in your ``~/.bashrc`` if you want to use
    *DeepHyper* in all new session.

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
    For a temporary compatibility fix - please upgrade setuptools at this step using ``pip install --upgrade setuptools``

.. note::
    To activate your virtualenv easier in the future you can define an alias
    in your ``~/.bashrc`` such as ``alias act="source ~/deephyper-dev-env/bin/activate"``. Now you will clone deephyper sources and install it with ``pip``

5. Clone the deephyper repo::

    git clone https://github.com/deephyper/deephyper_repo.git

6. Go to the root directory of the repo::

    cd deephyper_repo/


7. Switch to the develop branch::

    git checkout develop

8. Install the package::

    pip install -e .

