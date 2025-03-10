.. _install-conda:

Install with Conda
******************

.. warning:: All packages required for your application need to be installed in the same environment.

DeepHyper is a pure Python package and therefore does not require a conda building recipy to be installed because DeepHyper's code does not need to be compiled. However, many of DeepHyper's dependencies can require compilation (depending on your system) to optimize computation. For this reason, using conda can be a good option to install DeepHyper and its dependencies.

Linux
-----

Install Miniconda form the `official installation guide <https://docs.conda.io/en/latest/miniconda.html>`_, and create a new environment for DeepHyper called ``dh``:

.. code-block:: console

    $ conda create -n dh python=3.12 -y
    $ conda activate dh

Finally install DeepHyper with ``pip`` within this environement:

.. code-block:: console

    $ pip install pip --upgrade
    $ pip install deephyper

.. note:: More details about DeepHyper's optional modules can be found in the :ref:`install-pip` section.

MacOS
-----

MacOS users can either have ``x86_64`` or ``arm64`` architecture. To check your architecture, open a terminal and run the following command:

.. code-block:: console

    python3 -c "import platform; print(platform.platform());"


In any case on MacOS, you need to install Xcode command line tools first:

.. code-block:: console

    xcode-select --install



Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, and create the ``dh`` environment:

.. code-block:: console

    $ conda create -n dh python=3.12 -y
    $ conda activate dh

Then install DeepHyper in the previously created ``dh`` environment:

.. code-block:: console

    $ pip install pip --upgrade
    $ pip install deephyper

.. note:: More details about DeepHyper's optional modules can be found in the :ref:`install-pip` section.
