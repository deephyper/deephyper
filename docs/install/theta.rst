Theta installation
******************

User installation
=================

When you are a user deephyper can be directly installed as a module on Theta.

::

    module load deephyper

Developer installation
======================

Load the miniconda module which is using Intel optimized wheels for some of the dependencies we need:
::

    module load miniconda-3.6/conda-4.5.12

Load the balsam module:
::

    module load balsam/0.3


Create a virtual environment for your deephyper installation as a developer:
::

    mkdir deephyper-dev-env

::

    python -m venv --system-site-packages deephyper-dev-env

Activate this freshly created virtual environment:
::

    source deephyper-dev-env/bin/activate

To activate your virtualenv easier in the future you can define an alias in your ``~/.bashrc`` such as ``alias act="source ~/deephyper-dev-env/bin/activate"``. Now you will clone deephyper sources and install it with ``pip``:

::

    git clone https://github.com/deephyper/deephyper.git

::

    cd deephyper/


Switch to the develop branch:
::

    git checkout develop

::

    pip install -e .

