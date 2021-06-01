import os
import json
from random import random, seed

from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch

dhlogger = util.conf_logger("deephyper.search.nas.random")


class Random(NeuralArchitectureSearch):
    """Search class to run a full random neural architecture search. The search is filling every available nodes as soon as they are detected. The master job is using only 1 MPI rank.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.nas.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
    """

    def __init__(self, problem, run, evaluator, **kwargs):

        super().__init__(problem=problem, run=run, evaluator=evaluator, **kwargs)

        seed(self.problem.seed)

        self.free_workers = self.evaluator.num_workers

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        return parser

    def saved_keys(self, val: dict):
        res = {
            "id": val["id"],
            "arch_seq": str(val["arch_seq"])
        }
        return res

    def main(self):

        # Setup
        space = self.problem.space
        search_space = self.problem.build_search_space()

        len_arch = search_space.num_nodes

        def gen_arch():
            return [random() for _ in range(len_arch)]

        num_evals_done = 0
        available_workers = self.free_workers

        def gen_batch(size):
            batch = []
            for _ in range(size):
                cfg = space.copy()
                cfg["arch_seq"] = gen_arch()
                batch.append(cfg)
            return batch

        # Filling available nodes at start
        self.evaluator.add_eval_batch(gen_batch(size=available_workers))

        # Main loop
        while num_evals_done < self.max_evals:
            results = self.evaluator.get_finished_evals()

            num_received = num_evals_done
            for _ in results:
                num_evals_done += 1
            num_received = num_evals_done - num_received

            # Filling available nodes
            if num_received > 0:
                self.evaluator.dump_evals(saved_keys=self.saved_keys)

                if num_evals_done < self.max_evals:
                    self.evaluator.add_eval_batch(gen_batch(size=num_received))


if __name__ == "__main__":
    args = Random.parse_args()
    search = Random(**vars(args))
    search.main()
