Workflow
********

On local computer
=================

On super computer (Theta/Cooley)
================================

General Workflow
----------------

We are using ``ENV_NAME`` as an alias for your conda environment. (tricks: a shortcut for ``source`` is `.`)

Activate your current conda environment

::

    source activate ENV_NAME

Check your balsam database with

::

    balsam which --list

If you didn't create any database yet, let's create one

::

    balsam create DB_NAME


Now you can start the balsam database with

::

    source balsamactivate DB_NAME
