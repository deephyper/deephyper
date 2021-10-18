Documentation
*************

To build the documentation we use the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ package. The theme used is `Sphinx Book Theme <https://sphinx-book-theme.readthedocs.io/en/latest/>`_.

Developer Installation
======================

Follow the :ref:`local-dev-installation`. and have `Pandoc <https://pandoc.org/installing.html>`_ installed.

Build the Documentation
=======================

Once your virtual environment with DeepHyper is activated run the following commands:

.. code-block::

    $ cd deephyper/docs/
    $ make html

Then open ``_build/html/index.html`` in the navigator.

Additionnal information
-----------------------

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
 ...            ...
============= =============


Sphinx uses reStructuredText files, click on this `link <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ if you want to have an overview of the corresponding syntax and mechanism.

.. WARNING::
    Our documentation try to take part of the inline documentation in the code to auto-generate documentation from it. For that reason we highly recommend you to follow specific rules when writing inline documentation : https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html.