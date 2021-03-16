ThetaGPU (ALCF)
***************

`ThetaGPU <theta healing>`_  is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes at Argonne Leadership Computing Facility (ALCF).

The documention of ThetaGPU from the Datascience group at Argonne National Laboratory can be accessed `here <https://argonne-lcf.github.io/ThetaGPU-Docs/>`_.


.. _thetagpu-user-installation:

User installation
=================

Locate yourself on one of the login nodes of ThetaGPU

::

    ssh thetagpusn1

.. note::
    It is advised to do this procedure from your project directory::

        cd /lus/theta-fs0/projects/$PROJECTNAME

Then DeepHyper can be installed on ThetaGPU by following these commands.

::

    source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3/setup.sh
    conda config --set pip_interop_enabled False
    conda create -p dhgpu --clone base
    conda activate dhgpu/
    pip install pip --upgrade
    pip install deephyper
