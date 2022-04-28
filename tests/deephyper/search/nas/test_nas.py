import pytest
import unittest


@pytest.mark.nas
class TestNeuralArchitectureSearchAlgorithms(unittest.TestCase):
    def evaluate_search(self, search_cls, problem):
        from deephyper.evaluator import Evaluator
        from deephyper.nas.run import run_debug_arch

        # Test "max_evals" stopping criteria
        evaluator = Evaluator.create(run_debug_arch, method="serial")

        search = search_cls(problem, evaluator)

        res = search.search(max_evals=10)
        self.assertEqual(len(res), 10)

        # Test "max_evals" and "timeout" stopping criterias
        # evaluator = Evaluator.create(run_debug_slow, method="serial")

        # search = search_cls(problem, evaluator)

        # with pytest.raises(TypeError):  # timeout should be an int
        #     res = search.search(max_evals=10, timeout=1.0)
        # t1 = time.time()
        # res = search.search(max_evals=10, timeout=1)
        # d = time.time() - t1
        # self.assertAlmostEqual(d, 1, delta=0.1)

    def test_random(self):
        from deephyper.search.nas import Random
        from deephyper.benchmark.nas.linearReg import Problem as linear_reg_problem
        from deephyper.benchmark.nas.linearRegHybrid import (
            Problem as linear_reg_hybrid_problem,
        )

        self.evaluate_search(Random, linear_reg_problem)
        self.evaluate_search(Random, linear_reg_hybrid_problem)

    def test_regevo(self):
        from deephyper.search.nas import RegularizedEvolution
        from deephyper.benchmark.nas.linearReg import Problem as linear_reg_problem

        self.evaluate_search(RegularizedEvolution, linear_reg_problem)

    def test_regevomixed(self):
        from deephyper.search.nas import RegularizedEvolutionMixed
        from deephyper.benchmark.nas.linearReg import Problem as linear_reg_problem
        from deephyper.benchmark.nas.linearRegHybrid import (
            Problem as linear_reg_hybrid_problem,
        )

        self.evaluate_search(RegularizedEvolutionMixed, linear_reg_problem)
        self.evaluate_search(RegularizedEvolutionMixed, linear_reg_hybrid_problem)

    def test_ambsmixed(self):
        from deephyper.search.nas import AMBSMixed
        from deephyper.benchmark.nas.linearReg import Problem as linear_reg_problem
        from deephyper.benchmark.nas.linearRegHybrid import (
            Problem as linear_reg_hybrid_problem,
        )

        self.evaluate_search(AMBSMixed, linear_reg_problem)
        self.evaluate_search(AMBSMixed, linear_reg_hybrid_problem)

    def test_agebo(self):
        from deephyper.search.nas import AgEBO
        from deephyper.benchmark.nas.linearReg import Problem as linear_reg_problem
        from deephyper.benchmark.nas.linearRegHybrid import (
            Problem as linear_reg_hybrid_problem,
        )

        with pytest.raises(ValueError):
            self.evaluate_search(AgEBO, linear_reg_problem)

        self.evaluate_search(AgEBO, linear_reg_hybrid_problem)
