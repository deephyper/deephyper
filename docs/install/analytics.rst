Analytics
*********

.. _analytics-local-install:

From Pypi::

    pip install 'deephyper[analytics]'

or from github::

    pip install -e '.[analytics]'


Then to make DeepHyper accessible in a notebook create a new *IPython* kernel with (before running the command make sure that your virtual environment is activated if you are using one)::

    python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Now when you will open a Jupyter notebook the "Python (deephyper)" kernel will be available.

Theta
=====

Then go to `Theta Jupyter <https://jupyter.alcf.anl.gov/theta>`_ and use
your regular authentication method. The `Jupyter Hub tutorial <https://www.alcf.anl.gov/user-guides/jupyter-hub>`_
from Argonne Leadership Computing Facility might help you in case of troubles.

.. WARNING::

    Now when openning a generated notebook make sure to use the *"Python (deephyper)"* kernel before executing otherwise you will not have all required dependencies.
