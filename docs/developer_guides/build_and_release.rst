Build and Release
*****************

This document describes how to build and release new wheels for DeepHyper.

First check the status of Python versions as it is important to know which are currently supported and which are not. 
See https://devguide.python.org/versions/ for more information.

Then,

1. Go to the develop branch: 

.. code-block:: bash
    
    $ git checkout develop

2. Sync ``develop`` with ``master``:

.. code-block:: bash
    
    $ git merge master

3. Start the git flow release branch:

.. code-block:: bash

    $ git flow release start <version>

3. Check the `deephyper/__version__.py`, edit `__version__` and `__version_suffix__` and commit.

4. End the release process:

.. code-block:: bash

    $ EDITOR=vim git flow release finish <version>

5. Push develop and master branch then check if triggered github actions finished properly.

.. code-block:: bash

    $ git push origin master develop

6. Push new tags: 

.. code-block:: bash

    $ git push origin --tags

7. Remove old builds that may be in the project root directory with ``rm -rf dist/``.

8. Install the latest version of build and twine with ``uv pip install --upgrade build twine``.

9. Try to build the package wheel with ``python -m build``.

10. Upload the built package to PyPI using ``twine upload dist/*``.
