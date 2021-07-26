Local
******

DeepHyper installation requires ``Python>=3.7``.

We recommend creating isolated Python environments on your local machine using ``virtualenv`` or ``miniconda``, for example::

    conda create -n deephyper python=3.8
    conda activate deephyper

Some features of DeepHyper are using the ``MPI`` library, which you have to install yourself depending on your system.

For Linux systems, it is required to have the following additionnal dependencies::

    apt-get install build-essential
    # or
    conda install gxx_linux-64 gcc_linux-64

User Installation
=================

.. warning:: This section assumes that you already have an active virtual or conda environment as well as the prerequisites detailed in the introduction of this page.

The user installation corresponds to someone who wants to use the features of the software without contributing to its code.

From PyPI::

    pip install deephyper

From Github::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    pip install -e .

.. _horovod-local-install:

Horovod
-------

If you want to install Horovod with DeepHyper (an MPI distribution is required)::

    pip install 'deephyper[balsam,hvd]'

.. _analytics-local-install:

Analytics
---------

From PyPI::

    pip install 'deephyper[analytics]'


Then to make DeepHyper accessible in a notebook create a new *IPython* kernel with (before running the command make sure that your virtual environment is activated if you are using one)::

    python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Now when you will open a Jupyter notebook the "Python (deephyper)" kernel will be available.

Developer Installation
======================

.. warning:: This section assumes that you already have an active virtual or conda environment as well as the prerequisites detailed in the introduction of this page.

The developer installation corresponds to someone who wants to use and/or contribute to the software.

From GitHub::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    git checkout develop
    pip install -e '.[analytics]'


Documentation & Tests installation
----------------------------------

If you want to install deephyper with test and documentation packages.

From PyPI::

    pip install 'deephyper[tests,docs]'

From Github::

    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    pip install -e '.[tests,docs]'


Build
-----

To build the documentation you just need to be in the ``deephyper/docs`` folder and run ``make html`` assuming you have ``make`` command line utility installed on your computer. Then you can see the build documentation inside the ``doc/s_build`` folder just by opening the ``index.html`` file with your web browser.

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
