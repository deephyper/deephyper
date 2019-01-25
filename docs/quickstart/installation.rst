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



Theta
=====

Deephyper can be directly installed as a module on Theta.

::

    module load deephyper


Cooley
======

    .. todo:: installation on Cooley

Contribute to documentation
===========================

Build
-----

To build the documentation you just need to be in the ``deephyper/doc`` folder and run ``make html`` assuming you have MakeFile installed on your computer. Then you can see the build documentation inside the ``doc/_build`` folder just by opening the ``index.html`` file with your web browser.

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

<aside class="notice">
Our documentation try to take part of the inline documentation in the code to auto-generate documentation from it. For that reason we highly recommend you to follow specific rules when writing inline documentation : https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html.
</aside>
