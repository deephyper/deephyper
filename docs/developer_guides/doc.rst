Documentation
*************

To build the documentation we use the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ package. The theme used is `Sphinx Book Theme <https://sphinx-book-theme.readthedocs.io/en/latest/>`_.

Developer Installation
======================

From an activated Python virtual environment run the following commands:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper.git
    $ cd deephyper/ && git checkout develop
    $ pip install -e ".[dev,analytics]"

Build the Documentation
=======================

Once your virtual environment with DeepHyper is activated run the following commands:

.. code-block::

    $ cd deephyper/docs/
    $ make html

Then open ``_build/html/index.html`` in the navigator.