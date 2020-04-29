"""
NasBench-101 is a database of trained neural networks corresponding to a specific cell-based search space for convolution neural networks (`Link to the paper <https://arxiv.org/abs/1902.09635>`_). To use this benchmark you need to install the ``nasbench`` package available at `github.com/google-research/nasbench <https://github.com/google-research/nasbench>`_.

Then, to download the full data set use::

    sh download_full.sh

Or, if you want to Download the dataset only with models evaluated for 108 epochs::

    sh download_only108.sh

An example usage with the regularized evolution search if you used ``download_only108.sh``::

    python -m deephyper.search.nas.regevo --problem deephyper.benchmark.nas.nasbench101.problem.Problem --evaluator threadPool --run deephyper.benchmark.nas.nasbench101.run_only108.run --max-evals 1000

Or, if you used ``download_full.sh``::

    python -m deephyper.search.nas.regevo --problem deephyper.benchmark.nas.nasbench101.problem.Problem --evaluator threadPool --run deephyper.benchmark.nas.nasbench101.run_full.run --max-evals 1000
"""