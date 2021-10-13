import pytest
import unittest
import time

from deephyper.search.nas import Random
from deephyper.search.nas import RegularizedEvolution
from deephyper.search.nas import RegularizedEvolutionMixed
from deephyper.search.nas import AMBSMixed
from deephyper.search.nas import AgEBO
from deephyper.evaluator.evaluate import Evaluator
from deephyper.benchmark.nas.linearReg import Problem as linear_reg_problem
from deephyper.benchmark.nas.linearRegHybrid import Problem as linear_reg_hybrid_problem
from deephyper.nas.run.quick import run as quick_run
from deephyper.nas.run.slow import run as slow_run

class TestNeuralArchitectureSearchAlgorithms(unittest.TestCase):

    def evaluate_search(self, search_cls, problem):
        # Test "max_evals" stopping criteria
        evaluator = Evaluator.create(
            quick_run, method="subprocess", method_kwargs={"num_workers": 1}
        )

        search = search_cls(problem, evaluator)

        res = search.search(max_evals=10)
        self.assertEqual(len(res), 10)

        # Test "max_evals" and "timeout" stopping criterias
        evaluator = Evaluator.create(
            slow_run, method="subprocess", method_kwargs={"num_workers": 1}
        )

        search = search_cls(problem, evaluator)

        with pytest.raises(TypeError): # timeout should be an int
            res = search.search(max_evals=10, timeout=1.0)
        t1 = time.time()
        res = search.search(max_evals=10, timeout=1)
        d = time.time() - t1
        self.assertAlmostEqual(d, 1, delta=0.1)

    def test_random(self):

        self.evaluate_search(Random, linear_reg_problem)
        self.evaluate_search(Random, linear_reg_hybrid_problem)

    def test_regevo(self):
        self.evaluate_search(RegularizedEvolution, linear_reg_problem)

    def test_regevomixed(self):
        self.evaluate_search(RegularizedEvolutionMixed, linear_reg_problem)
        self.evaluate_search(RegularizedEvolutionMixed, linear_reg_hybrid_problem)

    def test_ambsmixed(self):
        self.evaluate_search(AMBSMixed, linear_reg_problem)
        self.evaluate_search(AMBSMixed, linear_reg_hybrid_problem)

    def test_agebo(self):
        with pytest.raises(ValueError):
            self.evaluate_search(AgEBO, linear_reg_problem)

        self.evaluate_search(AgEBO, linear_reg_hybrid_problem)
