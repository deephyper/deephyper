Frequently Asked Questions (F.A.Q.)
===================================

1. :ref:`which-search-should-I-use`
2. :ref:`what-is-num-workers`
3. :ref:`how-to-scale-bo`
4. :ref:`why-more-results-than-max-evals`
5. :ref:`how-to-consider-uncertainty-nn-training`
6. :ref:`how-to-do-checkpointing-with-bo`
7. :ref:`how-to-debug`

.. _which-search-should-I-use:

Which search algorithm should I use?
------------------------------------

* If you come with an existing machine learning pipeline, then it is better to use ``deephyper.hpo`` algorithms such as ``CBO`` search (Centralized Bayesian Optimization) which can fine-tune this pipeline.
* If you come without an existing machine learning pipeline then it is better to use ``deephyper.search.nas`` algorithms which provides an existing supervised-learning pipeline.


.. _what-is-num-workers:

What is ``num_workers``?
------------------------

The ``num_workers`` is related to the ``Evaluator`` class from the ``deephyper.evaluator`` module. It represents the number of "concurrent" evaluations ("concurrent" because not always "parallel", for example, with a ``thread``-evaluator and an I/O bound ``run``-function). The ``num_workers`` does not restrict or isolate the number of resources that each ``run``-function evaluation will use (e.g., the number of GPUs).


.. _how-to-scale-bo:

How to scale the number of parallel evaluations in a Bayesian optimization search?
----------------------------------------------------------------------------------

Here we are interested in increasing the number of parallel evaluations performed by a Bayesian optimization search.

* If only a **few evaluations** of the ``run``-function can be afforded (``~ max_evals < 100``) then a good setting is ``CBO(problem, evaluator, surrogate_model="GP")`` with a relatively low number of parallel workers (``num_workers <= 10``). The ``surrogate_model="GP"`` parameter sets the surrogate model to Gaussian Process which has a cubic temporal complexity w.r.t. the number of collected evaluations.
* If a **large number of  evaluations** can be afforded (``~ max_evals > 100``) for the ``run``-function then a good setting is ``CBO(problem, evaluator, multi_point_strategy="qUCB" )`` which will replace the default iterative constant-liar strategy by a one-shot strategy. In this case, we tested with a number of parallel workers up to ``num_workers == 4196`` with a ``run``-function having a run-time of 60 secondes in average and bounded between 30 to 90 secondes (which is relatively fast compared to a neural network training).
* If **the number of collected evaluations** becomes large (i.e., fitting the surrogate model becomes more expensive which also depends on the number of parameters in the search space) then it is better to use a distributed Bayesian optimization (DBO) scheme to avoid congestion in the master's queue of received jobs. In DBO, each worker has a local Bayesian optimizer attribute which avoids congestion problems. Therefore the search use should be ``MPIDistributedBO(problem, run_function)``.


.. _why-more-results-than-max-evals:

Why is the number of evaluations returned larger than ``max_evals``?
--------------------------------------------------------------------

Algorithms provided in DeepHyper are parallel search algorithms. The search can be performed asynchronously (direclty submitting jobs to idle workers) or synchronously (waiting for all workers to finish their jobs before re-submitting new jobs). If ``evaluator.num_workers == 1`` then the number of results will be equal to ``max_evals`` (classic sequential optimization loop). However, if ```evaluator.num_workers > 1`` then batches of size ``> 1`` can be received anytime which makes the number of results grow with a dynamic increment between 1 and ``evaluator.num_workers``. Therefore, if the current number of collected evaluations is 7, 3 new evaluations are received in one batch and ```max_evals=8`` the search will return 10 results.

.. _how-to-consider-uncertainty-nn-training:

How do you consider uncertainty in a neural network training?
-------------------------------------------------------------

The surrogate model in Bayesian optimization is estimating both aleatoric (data uncertainty, therefore noise of neural network training from different random initialization) and epistemic uncertainty (areas of the search space unknown to the the model). This is done differently for different surrogate models. For example, the Random-Forest estimator uses the law of total variance to estimate these quantities. The Random-Forest surrogate model follows the "random-best" split rule instead of "best" split to estimate better epistemic uncertainty. By default, the same configuration of parameters is not evaluated multiple times with ``CBO(..., filter_duplicated=True)`` because empirically it brings better results. Therefore if the user wants to take into consideration the noise from neural networks training it should be ``CBO(..., filter_duplicated=False)``.

.. _how-to-do-checkpointing-with-bo:

How to perform checkpointing with Bayesian optimization?
--------------------------------------------------------

The ``CBO(..., log_dir=".")`` algorithm will save new results in ``{log_dir}/results.csv`` (by default to the current directory) each time they are received. Then in case of failure and the search needs to be re-launched it can be done with the following:

.. code-block:: python

    search = CBO(..., log_dir="new-log-dir") # to avoid writting on top of previous set of results
    search.fit_surrogate("results.csv") # load checkpoint
    results = search.search(max_evals=100) # continue the search


.. _how-to-debug:

How to debug with DeepHyper?
----------------------------

As a starting point is can be useful to activate the ``logger``:

.. code-block:: python

    import logging

    logging.basicConfig(
        # filename=path_log_file, # optional if we want to store the logs to disk
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

