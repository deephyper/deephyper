Running on Theta
****************

User
====

Hyperparameter Search
---------------------

First we are going to run a search on ``deephyper.benchmark.hps.polynome2``
benchmark. In order to achieve this goal we first have to load the deephyper
module on Theta. This module is bringing deephyper in you current environment
but also the cray python distribution and the balsam software::

    module load deephyper

Then you can create a new postgress database in the current directory, this
database is used by the balsam software::

    balsam init testdb

Once the database has been created you can start it, or link to it if
it is already running::

    source balsamactivate testdb

The database is now running, let's create our first balsam application
in order to run a Asynchronous Model-Based Search (AMBS)::

    balsam app --name AMBS --exec "$(which python) -m deephyper.search.hps.ambs"

You can run the following command to print all the applications available
in your current balsam environment::

    balsam ls apps

Then create a new job::

    balsam job --name test --application AMBS --workflow TEST --args '--evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run'

Print jobs referenced in Balsam database::

    balsam ls jobs

.. note::

    In our case we are setting ``PROJECT_NAME`` to *datascience*::

        export PROJECT_NAME=datascience

Finally you can submit a cobalt job to Theta which will start by running
your master job named test::

    balsam submit-launch -n 128 -q default -t 180 -A $PROJECT_NAME --job-mode serial --wf-filter TEST


Now if you want to look at the logs, go to ``testdb/data/TEST``. You'll see
one directory prefixed with ``test``. Inside this directory you will find the
logs of you search. All the other directories prefixed with ``task`` correspond
to the logs of your ``--run`` function, here the run function is corresponding
to the training of a neural network.

Neural Architecture Search
--------------------------

::

    balsam app --name PPO --exec "$(which python) -m deephyper.search.nas.ppo"


::

    balsam job --name test --workflow TEST --app PPO --num-nodes 11 --args '--evaluator balsam --run deephyper.search.nas.model.run.alpha.run --problem naspb.pblp.problem_skip_co_0.Problem --ent-coef 0.01 --noptepochs 10 --network ppo_lnlstm_128 --gamma 1.0 --lam 0.95 --max-evals 1000000'

::

    balsam submit-launch -n 128 -q default -t 180 -A $PROJECT_NAME --job-mode mpi --wf-filter TEST
