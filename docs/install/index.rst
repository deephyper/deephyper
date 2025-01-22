.. _installation:

Installation
============

On this page, you will find documentation about installation options of DeepHyper.
The package in itself is a pure Python package and therefore does not require any "compilation".
However, some features of the package can require external libraries to be installed that require compilation when pre-built binaries are not available (or not optimized) for your target system (e.g., Tensorflow or Numpy).

DeepHyper installation requires ``Python>=3.10``.

.. toctree::
   :maxdepth: 1
   :caption: Generic Installations
   :name: mastertoc

   conda <conda>
   jupyter <jupyter>
   pip (recommended) <pip>
   spack <spack>
   uv <uv>

.. toctree::
   :maxdepth: 1
   :caption: High Performance Computing Centers

   hpc/alcf
   hpc/nersc
   hpc/olcf
