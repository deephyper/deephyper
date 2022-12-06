Crusher
*****

`Crusher <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html>`_ a DOE supercomputer at OLCF

Install DeepHyper:

.. code-block:: console

    $ module load cray-python
    $ python3.9 -m venv $PWD/environ_dh python=3.9
    $ source environ_dh/bin/activate
    $ git clone -b develop https://github.com/deephyper/deephyper.git
    $ pip install --no-cache-dir -e "deephyper/"
    $ pip install parse
    $ pip install sdv
    
When submiting/allocating a job to SLURM, make sure you activate the virtual environment in your script:

.. code-block:: console
    $ module load ... #your modules
    $ module load cray-python
    $ source $PWD/environ_dh/bin/activate

.. warning::

    The Crusher installation will only provide the default DeepHyper installation (i.e., hyperparameter optimization). All features will not be included by default.
