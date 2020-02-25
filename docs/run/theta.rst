Running on Theta
****************

User
====

Hyperparameter Search
---------------------

First we are going to run a search on the ``deephyper.benchmark.hps.polynome2``
benchmark. In order to achieve this goal we must load the DeepHyper
module on Theta. This module is not only bringing DeepHyper into you current environment,
but also the Cray Python distribution and the Balsam software::

    module load deephyper

Then, you can create a new Postgres database in the current directory; this
database is used by the Balsam software::

    balsam init testdb

Once the database has been created you can start it or link to it if
it is already running::

    source balsamactivate testdb

The database is now running. Let's define our first Balsam application
in order to run a Asynchronous Model-Based Search (AMBS)::

    balsam app --name AMBS --exec "$(which python) -m deephyper.search.hps.ambs"

You can run the following command to print all the applications available
in your current Balsam environment::

    balsam ls apps

Create a new job::

    balsam job --name test --application AMBS --workflow TEST --args '--evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run'

Print jobs referenced in Balsam database::

    balsam ls jobs

.. note::

    In our case we are setting ``PROJECT_NAME`` to *datascience*::

        export PROJECT_NAME=datascience

Finally, you can submit a Cobalt job to Theta which will start by running
your master job named test::

    balsam submit-launch -n 128 -q default -t 180 -A $PROJECT_NAME --job-mode serial --wf-filter TEST

Each Balsam job runs in its own working directory, which is generated from the ``workflow``
and ``name`` attributes as follows: ``testdb/data/{workflow}/{name}_{id}``. In the context of 
DeepHyper, we want to follow the output of the search task.
Navigate to the ``TEST`` workflow directory ``testdb/data/TEST``.  Here, you'll see
one directory prefixed with ``test`` corresponding to the AMBS application that we named ``test``.
Inside this directory you will find the DeepHyper search logs.

Each evaluation of the ``--run`` function in the course of a search counts as a separate Balsam job
prefixed with ``task``.  Therefore, the directories matching ``testdb/data/TEST/task*`` will contain
the individual logs of each run function evaluation. Here, the run function 
corresponds to the training of a neural network and returns a suitable fitness metric such as
the trained model's validation accuracy.

The above specification of Balsam ``--job-mode=serial`` can be used for tasks that **do
not** use MPI (and hence are restricted to a single node). With the current settings,
Balsam executes a ``Master`` orchestrator process on the entirety of the first node, and
each of the remaining 127 nodes will execute isolated DeepHyper "workers". One worker
(compute node) will execute the AMBS searcher/learner process. The remaining 126 compute
nodes will execute independent evaluators defined by
``deephyper.benchmark.hps.polynome2.run``. Only a single call to the job scheduler's
``mpirun``-equivalent command is made at the beginning of ``submit-launch`` command; jobs
are run as forked processes on all of the ``Worker`` ranks.

DeepHyper automatically infers the number of available workers from the Balsam environment
variables ``BALSAM_LAUNCHER_NODES`` and ``BALSAM_JOB_MODE``. Additionally, the user can
define the ``DEEPHYPER_WORKERS_PER_NODE`` environment variable to pack several tasks per
node in this Balsam job mode. In that case, the ``--node-packing-count`` option of
``balsam job`` should be changed from its default value of 1 to be consistent with the
environment variable and ensure full job occupancy of the ``Worker`` ranks. For example,
to run 4 DeepHyper workers per node, the above command is modified::

  balsam job --name test --application AMBS --workflow TEST --node-packing-count=4 --env DEEPHYPER_WORKERS_PER_NODE=4 --args '--evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run'

The resulting behavior is as follows:
  - ``Node0`` executes the ``Master`` orchestrator process
  - ``Node1`` executes the AMBS searcher process and 3x DeepHyper evaluators
  - ``Node2, ..., Node126`` each execute 4x DeepHyper evaluators

.. note::
   If Balsam is launched in this mode with only one node, ``balsam submit-launch -n 1
   ...``, the ``Master`` process will share the node with 4x DeepHyper workers. It will
   not contribute to the worker node occupancy calculations in Balsam.
    
The default Balsam job mode is ``--job-mode=mpi``. There are several key differences
when compared with the serial job mode:
1. The tasks may or may not use MPI (and multiple nodes).
2. No more than one task can be executed on a node at a time (current restriction).
   - Hence, ``DEEPHYPER_WORKERS_PER_NODE`` should be set to 1.
3. The launcher runs on the head node (or Machine Oriented Mini-server (MOM) node on
   Theta) and continuously submits jobs using the ``mpirun``-equivalent command for the
   given job scheduler. There is no notion of a ``Master`` process that consumes a compute
   node. 

See `Balsam documentation <https://balsam.readthedocs.io/en/latest/userguide/submit/>`_
for more information. 

Neural Architecture Search
--------------------------

::

    balsam app --name PPO --exec "$(which python) -m deephyper.search.nas.ppo"


::

    balsam job --name test --workflow TEST --app PPO --num-nodes 11 --args '--evaluator balsam --run deephyper.search.nas.model.run.alpha.run --problem naspb.pblp.problem_skip_co_0.Problem --ent-coef 0.01 --noptepochs 10 --network ppo_lnlstm_128 --gamma 1.0 --lam 0.95 --max-evals 1000000'

::

    balsam submit-launch -n 128 -q default -t 180 -A $PROJECT_NAME --job-mode mpi --wf-filter TEST
