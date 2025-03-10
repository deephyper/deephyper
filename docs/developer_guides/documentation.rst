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

Add an Example
==============

The `Examples <https://deephyper.readthedocs.io/en/stable/examples/index.html>`_ are generated using the `Sphinx-Gallery <https://sphinx-gallery.github.io/stable/index.html>`_ extension.

Example Source Files
--------------------

All example source files are located in the ``examples/`` directory (a.k.a., the main gallery) at the root of the repository. Each example source file should follow the naming convention: ``plot_*.py``. Each subfolder of ``examples`` that includes such example source files is called a sub-gallery.

Structuring Example Scripts
---------------------------

To learn about the syntax used to define examples checkout the following guide: `Structuring Python scripts for Sphinx-Gallery <https://sphinx-gallery.github.io/stable/syntax.html>`_

Format Dropdown for Code Blocks
-------------------------------

When writing examples, some portions of codes can be heavy for the reader while not being interesting. For example, code that formats a plot or code that loads data. To hide but still include these code portions we use the `Dropdown <https://sphinx-design.readthedocs.io/en/latest/dropdowns.html>`_ from Sphinx-Design. An example on how to use it:

.. code-block:: python

    # %%
    # The title of a text block
    # -------------------------
    # 
    # Some text here...

    # .. dropdown:: A title for my dropdown
    bar = foo(x)

The blank line between ``# Some text here...`` and ``# .. dropdown:: A title for my dropdown`` is necessary to make the difference between the text block and the code block.


Setting the Preview Figure
-------------------------

To set the figure from the example that should be displayed on the gallery page as preview use:

.. code-block:: python

    # sphinx_gallery_thumbnail_number = FIGURE_INDEX
    _ = plt.subplots(...)


Build Examples
--------------

Examples are automatically built when generating the documentation using the ``make html`` command from the ``docs/`` directory. During the build process, an ``.md5`` hash is created for each example. This ensures that examples are only recompiled if their content has changed, reducing unnecessary computation.

Output Files
------------

The generated files from each example are stored in ``docs/examples/``. These include:

- ``*.json``
- ``*.ipynb``
- ``*.py``
- ``*.md5``
- ``*.rst``
- ``*.zip``

Committing Built Examples
-------------------------

Once an example is finalized, all generated files should be committed to the repository. This prevents unnecessary recompilation on Read the Docs, as example execution can vary in resource consumption.