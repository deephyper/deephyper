Frequently Asked Questions (F.A.Q.)
===================================

#. :ref:`what-is-num-workers`
#. :ref:`how-to-scale-bo`
#. :ref:`why-more-results-than-max-evals`
#. :ref:`how-to-consider-uncertainty-nn-training`
#. :ref:`how-to-do-checkpointing-with-bo`
#. :ref:`how-to-debug`

.. _what-is-num-workers:

What is ``num_workers``?
------------------------

The ``num_workers`` is related to the ``Evaluator`` class from the ``deephyper.evaluator`` module. It represents the number of "concurrent" evaluations ("concurrent" because not always "parallel", for example, with a ``thread``-evaluator and an I/O bound ``run``-function). The ``num_workers`` does not restrict or isolate the number of resources that each ``run``-function evaluation will use (e.g., the number of GPUs).


.. _how-to-scale-bo:

How to scale the number of parallel evaluations in a Bayesian optimization search?
----------------------------------------------------------------------------------

Here we are interested in increasing the number of parallel evaluations performed by a Bayesian optimization search.

* If only a **few evaluations** of the ``run``-function can be afforded (``~ max_evals < 100``) then good settings can be:

  #. ``CBO(problem, acq_optimizer="mixedga")`` with a relatively low number of parallel workers (``num_workers <= 10``).
  #. ``CBO(problem, surrogate_model="GP")`` to set the surrogate model to Gaussian Process which has a cubic temporal complexity w.r.t. the number of collected evaluations.

* If a **large number of  evaluations** can be afforded (``~ max_evals > 300``) in a relatively short period of time then a good setting is ``CBO(problem, multi_point_strategy="qUCB", acq_optimizer_kwargs={"acq_optimizer_freq": 10})``. Look at the following examples for more detail:
   
  #. :ref:`sphx_glr_examples_examples_parallelism_plot_from_serial_to_parallel_hpo.py` 
  #. :ref:`sphx_glr_examples_examples_parallelism_plot_scaling_bo.py`

.. _why-more-results-than-max-evals:

Why is the number of evaluations returned larger than ``max_evals``?
--------------------------------------------------------------------

Algorithms provided in DeepHyper are parallel search algorithms. The search can be performed asynchronously (direclty submitting jobs 
to idle workers) or synchronously (waiting for all workers to finish their jobs before re-submitting new jobs). If ``evaluator.num_workers == 1`` 
then the number of results will be equal to ``max_evals`` (classic sequential optimization loop). However, if ``evaluator.num_workers > 1`` 
then batches of size ``> 1`` can be received anytime which makes the number of results grow with a dynamic increment between 1 and 
``evaluator.num_workers``. Therefore, if the current number of collected evaluations is 7, 3 new evaluations are received in one batch and 
``max_evals=8`` the search will return 10 results.

.. _how-to-consider-uncertainty-nn-training:

How do you consider noisy objectives for example the variability of a neural network training?
----------------------------------------------------------------------------------------------


In this case, it is important to understand the difference between the "observed objective" (i.e., the value returned by the black-box function)
and the "estimated objective" (i.e., the expected value estimated by a surrogate model). Look at the following example for more details:
:ref:`sphx_glr_examples_examples_bbo_plot_black_box_optimization_noisy.py`.

The surrogate model in Bayesian optimization is estimating the "expected observed objective". Of course the hyperaparameters of the surrogate 
should be adapted depending on the level of noise. Also some surrogate model can estimate both aleatoric (data uncertainty, therefore noise of 
neural network training from different random initialization) and epistemic uncertainty (areas of the search space unknown to the the model). 
This is done differently for different surrogate models. For example, the Random-Forest estimator uses the law of total variance to estimate 
these quantities. The Random-Forest surrogate model follows the "random-best" split rule instead of "best" split to estimate better epistemic 
uncertainty. By default, the same configuration of parameters is not evaluated multiple times with ``CBO(..., filter_duplicated=True)`` because 
empirically it brings better results. Therefore if the user wants to take into consideration the noise from neural networks training it should 
be ``CBO(..., filter_duplicated=False)``.

.. _how-to-do-checkpointing-with-bo:

How to perform checkpointing with Bayesian optimization?
--------------------------------------------------------

The ``CBO(..., log_dir=".")`` algorithm will save new results in ``{log_dir}/results.csv`` (by default to the current directory) each time they 
are received. Then in case of failure and the search needs to be re-launched it can be done with the following:

.. code-block:: python

    search = CBO(..., log_dir=".", checkpoint_restart=True) 
    results = search.search(run, max_evals=100) # continue the search


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

