Build and Release
*****************

This document describes how to build and release new wheels for DeepHyper.

First check the status of Python versions as it is important to know which becomes are currently supported and which are not. See https://devguide.python.org/versions/ for more information.

Then,

1. Go to the develop branch: `git checkout develop`.
2. Run unittests located at `deephyper/tests/` with:

.. code-block:: bash

    $ pytest --run fast,slow,hps,nas,ray,mpi tests

3. Run doctests located at `deephyper/docs/` with:

.. code-block:: bash

    $ make doctest

4. Check the `deephyper/__version__.py`, edit `VERSION` and `__version_suffix__`.
5. Start the git flow release branch:

.. code-block:: bash

    $ git flow release start <version>

6. End the release:

.. code-block:: bash

    $ EDITOR=vim git flow release finish <version>

7. Push commits and tags: 

.. code-block:: bash

    $ git push origin --tags

8. Make sure to be on the correct branch/tag

**For final release only**:
9. build and publish the wheel to `https://test.pypi.org/`: `python setup.py testupload`
10. try to install the wheel upload to Test PyPI: `python setup.py testinstall`
11. build and publish the wheel to `https://pypi.org/`: `python setup.py upload`
12. try to install the wheel upload to PyPI: `pip install deephyper==$VERSION`
