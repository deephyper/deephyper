Frequently Asked Questions (F.A.Q.)
===================================

1. :ref:`which-search-should-I-use`
2. :ref:`what-is-num-workers`
3. :ref:`how-to-scale-bo`

.. _which-search-should-I-use:

Which search algorithm should I use?
------------------------------------

TODO


.. _what-is-num-workers:

What is ``num_workers``?
------------------------

The ``num_workers`` is related to the ``Evaluator`` class. It represents the number of concurrent (not always "parallel" for example with a thread-base evaluator and an I/O bound ``run``-function) evaluations. The ``num_workers`` does not restrict the number of resources that each ``run``-function evaluation will use (e.g., the number of GPUs).


.. _how-to-scale-bo:

How to scale a Bayesian optimization search?
--------------------------------------------

* CBO: centralized bayesian optimization (default parameters: RandomForest, UCB, CL)
* CBO with faster multi-point acquisition function (qUCB)
* DBO: distributed BO to avoid congestion issues and have better convergence
