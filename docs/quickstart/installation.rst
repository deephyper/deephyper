Installation
************

Local
=====

User installation
-----------------

You can run the following commands if you want to install deephyper on your local machine.

From Pypi:
::

    pip install deephyper

From github:
::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    pip install -e .

.. todo:: install deephyper on GPU

Contributor installation
------------------------

If you want to install deephyper with test and documentation packages.

From pypi:
::

    pip install 'deephyper[tests,docs]'

From github:
::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    pip install -e '.[tests,docs]'



Argonne Leadership Computing Facility
=====================================

Theta - User
------------

When you are a user deephyper can be directly installed as a module on Theta.

::

    module load deephyper

Theta - Developer
-----------------

Load the miniconda module which is using Intel optimized wheels for some of the dependencies we need:
::

    module load miniconda-3.6/conda-4.5.12

Load the balsam module:
::

    module load balsam

Create a virtual environment for your deephyper installation as a developer:
::

    mkdir deephyper-dev-env

::

    python -m venv deephyper-dev-env

Activate this freshly created virtual environment:
::

    source deephyper-dev-env/bin/activate

To activate your virtualenv easier in the future you can define an alias in your ``~/.bashrc`` such as ``alias act="~/deep/bin/activate ~/deep"``. Now you will clone deephyper sources and install it with ``pip``:

::

    git clone https://github.com/deephyper/deephyper.git

::

    cd deephyper/


Switch to the develop branch:
::

    git checkout develop

::

    pip install -e .


Contribute to documentation
===========================

Build
-----

To build the documentation you just need to be in the ``deephyper/docs`` folder and run ``make html`` assuming you have MakeFile installed on your computer. Then you can see the build documentation inside the ``doc/s_build`` folder just by opening the ``index.html`` file with your web browser.

Useful informations
-------------------

The documentation is made with Sphinx and the following extensions are used :

============= =============
 Extensions
---------------------------
 Name          Description
============= =============
 autodoc       automatically insert docstrings from modules
 napoleon      inline code documentation
 doctest       automatically test code snippets in doctest blocks
 intersphinx   link between Sphinx documentation of different projects
 todo          write "todo" entries that can be shown or hidden on build
 coverage      checks for documentation coverage
 mathjax       include math, rendered in the browser by MathJax
 ifconfig      conditional inclusion of content based on config values
 viewcode      include links to the source code of documented Python objects
 githubpages   create .nojekyll file to publish the document on GitHub pages
============= =============


Sphinx uses reStructuredText files, click on this `link <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ if you want to have an overview of the corresponding syntax and mechanism.

.. WARNING::
    Our documentation try to take part of the inline documentation in the code to auto-generate documentation from it. For that reason we highly recommend you to follow specific rules when writing inline documentation : https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html.

