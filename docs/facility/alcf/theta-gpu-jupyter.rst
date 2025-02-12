.. _tutorial-alcf-04:

Execution on the ThetaGPU supercomputer (within a Jupyter notebook)
*******************************************************************

In this tutorial we are going to learn how to run an interactive Jupyter notebook on the **ThetaGPU** supercomputer at the ALCF using Ray. `ThetaGPU <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is a 3.9 petaflops system based on NVIDIA DGX A100.

After logging in Theta:

1. From ``thetaloginX``, start an interactive job (**note** which ``thetagpuXX`` node you get placed onto will vary) by replacing your ``$PROJECT_NAME`` and ``$QUEUE_NAME`` (e.g. of available queues are ``full-node`` and ``single-gpu``):

.. code-block:: console

    (thetalogin5) $ qsub -I -A $PROJECT_NAME -n 1 -q $QUEUE_NAME -t 60
    Job routed to queue "full-node".
    Wait for job 10003623 to start...
    Opening interactive session to thetagpu21

2. Wait for the interactive session to start. Then, from the ThetaGPU compute node (`thetagpuXX`), execute the following commands to initialize your DeepHyper environment (adapt to your needs):

.. code-block:: console

    $ . /etc/profile
    $ module load conda/2022-07-01
    $ conda activate $CONDA_ENV_PATH

3. Then, start the Jupyter notebook server:

.. code-block:: console

    $ jupyter notebook &

.. note::

    In the case of a multi-GPUs node, it is possible that the Jupyter notebook process will lock one of the available GPUs. Therefore, launch the notebook with the following command instead:

    .. code-block:: console

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 jupyter notebook &

4. Take note of the hostname of the current compute node (e.g. ``thetagpuXX``):

.. code-block:: console

    echo $HOSTNAME

5. Leave the interactive session running and open a new terminal window on your local machine.

6. In the new terminal window, execute the SSH command to link the local port to the ThetaGPU compute node after replacing with you ``$USERNAME`` and corresponding ``thetagpuXX``:

.. code-block:: console

    $ ssh -tt -L 8888:localhost:8888 $USERNAME@theta.alcf.anl.gov "ssh -L 8888:localhost:8888 thetagpuXX"

7. Open the Jupyter URL (`http:localhost:8888/?token=....`) in a local browser. This URL was printed out at step 4.
