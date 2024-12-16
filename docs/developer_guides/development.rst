Development
===========

Development Flow
----------------

To manage new features, bug-fix and release we use `Git-Flow <https://danielkummer.github.io/git-flow-cheatsheet/>`_.
External contributions can be submitted through Github Pull Requests.

For maintainers:
- The ``master`` branch is used for production (i.e., features ready for the public to use).
- The ``develop`` branch is used to merge intermediate features during before releasing to the ``master`` branch.
- All ``feature/$FEATURE_NAME`` branches are used to develop each intermediate feature (many can be used in parallel).

For other developers, it is better to use Github Pull Requests so that the submitted code can be reviewed by maintainers of the repository. The first, step is to fork the github repository (i.e., do not forget to "synchronise" your fork regularly with the main repository), then develop new features the ``develop`` branch and once completed or well advanced submit it to the main repository ``develop`` branch.


Developer Installation
----------------------

The installation of the development dependencies (e.g., code formatting, linter, documentation) uses the ``[dev]`` marker. Before cloning the repository from Github, ``git-lfs`` needs to be installed (see more on `Installing Git Large File System <https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>`_).

Git Large File System (LFS) is used to store all files which are necessary and not quickly retrievable such as figures derived from tutorials or data for testing. 

.. code-block:: console

    $ git clone -b develop git@github.com:deephyper/deephyper.git
    $ cd deephyper/

    $ # Generic installation
    $ pip install -e ".[default,dev]"

    $ # MacOS (arm64) installation
    $ conda env create -f install/environment.dev.macOS.arm64.yml

    $ pre-commit install
    

.. warning::

    When installing the ``develop`` branch of DeepHyper the published documentation may be outdated. In case of trouble, do not hesitate to contact the maintainers of the repository on Slack or Github.

.. note:: More details about DeepHyper's optional modules can be found in the :ref:`install-pip` section.

Code linting and formatting
---------------------------

Python code should be linted and formatted with `Ruff <https://github.com/astral-sh/ruff>`_. Linter and formatter checks are performed on the **examples** and **deephyper** directories. The linter checks are run with the following command:

.. code-block:: console

   $ ruff check examples deephyper

While the formatter checks are run with the following command:

.. code-block:: console

  $ ruff format --check examples deephyper

See the `Ruff documentation <https://docs.astral.sh/ruff/>`_ for more information.
