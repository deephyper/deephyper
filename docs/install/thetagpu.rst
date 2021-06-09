ThetaGPU (ALCF)
***************

`ThetaGPU <theta healing>`_  is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes at Argonne Leadership Computing Facility (ALCF).

The documention of ThetaGPU from the Datascience group at Argonne National Laboratory can be accessed `here <https://argonne-lcf.github.io/ThetaGPU-Docs/>`_. The system documentation from the ALCF can be accessed `here <https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu>`_.


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

    module load conda/tensorflow/2020-11-11
    conda create -p dhgpu --clone base
    conda activate dhgpu/
    pip install pip --upgrade
    pip install deephyper


Developer installation
======================

The developer installation allows to edit the ``deephyper/`` code base without re-installation:

::

    module load conda/tensorflow/2020-11-11
    conda create -p dhgpu --clone base
    conda activate dhgpu/
    pip install pip --upgrade
    git clone https://github.com/deephyper/deephyper.git
    cd deephyper/
    git checkout develop
    pip install -e .


Proxy
=====

If the node you are on does not have outbound network connectivity, set the following to access the proxy host

::

    export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
    export https_proxy=http://proxy.tmi.alcf.anl.gov:3128