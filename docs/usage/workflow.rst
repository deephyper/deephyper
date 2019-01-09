Workflow
********

On local computer
=================

TODO (see Quick Start for now)




On super computer (Theta/Cooley)
================================

General Workflow
----------------

Load the deephyper module

::

    # You can add this line in your ~/.bashrc so that
    # deephyper will be automatically loaded at your login
    module load deephyper

Check your available balsam databases with

::

    balsam which --list

If you didn't create any database yet, let's create one

::

    balsam create DB_NAME


Now you can start or connect to the balsam database with

::

    source balsamactivate DB_NAME

Before running a search you need to create an balsam application corresponding to the executable. Inside deephyper we provide different applications. For example, you can access AMBS search through the environment variable ``DH_AMBS``.

::

    balsam app --name APPLICATION_NAME --exec EXECUTABLE_PATH


To see your available balsam applications do

::

    balsam ls apps

Now you need to add a balsam job that is going to run your application with a specific configuration, for example with a set of arguments.

::

    balsam job --name JOB_NAME --workflow WF_NAME --application APPLICATION_NAME --args '--problem foo.problem.Problem --run foo.run.run'

to see the available arguments of a specific search just run

::

    python $EXECUTABLE_PATH_OF_SEARCH
    # for example : python $DH_AMBS

You can finally run your search on Theta with

::

    balsam submit-launch -q QUEUE_NAME -n NUMBER_OF_NODES -t TIME_IN_MINUTES -A PROJECT_NAME --job-mode serial --wf-filter WORKFLOW_NAME

This last command submits the theta job to the queue for you. The workflow is always an optional parameter. If you don't give a workflow filter the balsam launcher will start all available jobs inside the database.
