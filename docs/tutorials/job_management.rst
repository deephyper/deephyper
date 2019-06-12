How to manage your Balsam jobs?
*******************************

Let's say we just ran the problem from the previous tutorial
:ref:`create-new-nas-problem`.

If we look at the current state of our balsam database with::

    balsam ls --wf TEST

We should expect something like this::

    SHOW BALSAM DB


As you can see some jobs of our *TEST* workflow are in state ``RESTART_READY``. According to the `Balsam documentation <https://balsam.readthedocs.io/en/latest/index.html>`_ the job will be run again if a new launcher with the *TEST* workflow is executed.

.. image:: https://balsam.readthedocs.io/en/latest/_images/state-flow.png


This is why you should delete all jobs of the same workflow if you want to execute the same experiment again. To do so you can use::

    balsam rm jobs --name | --id

You can also use balsam django models directly such as::

    EXAMPLE

then execute::

    python script_with_django_model.py arg1 arg2
