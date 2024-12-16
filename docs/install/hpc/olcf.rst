Oak Ridge Leadership Computing Facility (OLCF)
**********************************************

Crusher
=======

`Crusher <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html>`_ is a DOE supercomputer at OLCF.

To install DeepHyper on Crusher after connecting to the system run the following commands:

.. code-block:: console

    $ module load cray-python
    $ python -m venv dh-env python=3.10
    $ source dh-env/bin/activate
    $ pip install deephyper
    
When submiting/allocating a job to the Slurm scheduler, make sure you activate the previously created Python virtual environment in your script:

.. code-block:: console

    $ module load ... # Load other modules you require to use
    $ module load cray-python
    $ source dh-env/bin/activate # Assuming to be in the same directory as when we execute the previous set of commands

.. warning::

    The Crusher installation will only provide the default DeepHyper installation (i.e., hyperparameter optimization). All features will not be included by default (transfer learning, ...). Check out the local installation page for further details.
