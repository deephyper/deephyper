Running on Theta
****************

Hyperparameter Search
==========================

First we are going to run a search on ``deephyper.benchmark.hps.polynome2``
benchmark. In order to achieve this goal we first have to load the deephyper
module on Theta. This module contains a Cray Python 3.6-based virtual environment
with Balsam and DeepHyper installed.::

    module load deephyper

Next, create a Balsam database in one of your ``/projects`` subdirectories::

    balsam init testdb

Once the database has been created you can start (or re-connect to) it::

    source balsamactivate testdb

The database is now up. We can submit an Asynchronous Model-Based Search (AMBS)
run through Balsam as follows::

    deephyper balsam-submit hps test -p deephyper.benchmark.hps.polynome2.Problem -r deephyper.benchmark.hps.polynome2.run \ 
    -t 60 -q debug-cache-quad -n 4 -A datascience -j mpi

This creates an AMBS hyperparameter search job for the given `Problem` and `run` arguments.  The parameters on the second line
indicate: 60 minute wall-time, submission to `debug-cache-quad` queue, running on 4 nodes, charging to `datascience` allocation,
and using the `mpi` job mode of Balsam. Refer to the Command Line Interface documentation for more information on this command.

You can use `balsam ls` to see the job that was added to the database::

    balsam ls jobs

Now if you want to look at the logs, go to ``testdb/data/TEST``. You'll see
one directory prefixed with ``test``. Inside this directory you will find the
logs of your search. All the other directories prefixed with ``task`` correspond
to the logs of your ``--run`` function, here the run function is corresponding
to the training of a neural network.

Neural Architecture Search
==========================

::

    balsam app --name PPO --exec "$(which python) -m deephyper.search.nas.ppo"


::

    balsam job --name test --workflow TEST --app PPO --num-nodes 11 --args '--evaluator balsam --run deephyper.search.nas.model.run.alpha.run --problem naspb.pblp.problem_skip_co_0.Problem --ent-coef 0.01 --noptepochs 10 --network ppo_lnlstm_128 --gamma 1.0 --lam 0.95 --max-evals 1000000'

::

    balsam submit-launch -n 128 -q default -t 180 -A $PROJECT_NAME --job-mode mpi --wf-filter TEST
