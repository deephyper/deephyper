"""
NasBench 101, a benchmark using: https://github.com/google-research/nasbench
To download the full data set use::

    sh download_full.sh

To Download the dataset only with models evaluated for 108 epochs::

    sh download_only109.sh

Example usage with the regularized evolution search::

    python -m deephyper.search.nas.regevo --problem deephyper.benchmark.nas.nasbench101.problem.Problem --evaluator threadPool --run deephyper.benchmark.nas.nasbench101.run.run --max-evals 10000
"""