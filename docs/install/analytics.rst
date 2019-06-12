Analytics installation
**********************

Local
=====


From Pypi::

    pip install 'deephyper[analytics]'

or from github::

    pip install -e '.[analytics]'

Theta
=====

Analytics are already part of the theta *deephyper* module hence
you just need to do::

    module load deephyper

Make sure to have a *IPython* kernel using the good python interpretor::

    python -m ipykernel install --user --name deephyper --display-name "Python (deephyper)"

Then go to `Theta Jupyter <https://jupyter.alcf.anl.gov/theta>`_ and use
your regular authentication method. The `Jupyter Hub tutorial <https://www.alcf.anl.gov/user-guides/jupyter-hub>`_
from Argonne Leadership Computing Facility might help you in case of troubles.

.. WARNING::

    Now when openning a generated notebook make sure to use the *"Python (deephyper)"* kernel before executing otherwise you will not have all required dependencies.
