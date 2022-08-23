Development
************

To manage new features, bug-fix and release we use `Git-Flow <https://danielkummer.github.io/git-flow-cheatsheet/>`_.
External contributions can be submitted through Github Pull Requests.


The code needs to be formatted with Black (``pip install black``). Then to check the diff you can do the following from the root of the repository:

.. code-block:: console

    black --diff --check $(git ls-files '*.py')

And to apply the formatting you can do from the same location:

.. code-block:: console

    black $(git ls-files '*.py')