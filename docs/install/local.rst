Local 
******

DeepHyper installation requires Python 3.6.

We recommend creating isolated Python environments on your local machine using ``virtaulenv`` or ``miniconda``.

User installation
=================

From Pypi::

    pip install deephyper

.. note::

    If you want to use ``tensorflow-gpu`` assuming you already have CUDA installed. Just do ``DH_GPU=true pip install -e .``

From github::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    pip install -e .

Developer installation
======================

If you want to install deephyper with test and documentation packages.

From pypi::

    pip install 'deephyper[tests,docs]'

From github::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    pip install -e '.[tests,docs]'


