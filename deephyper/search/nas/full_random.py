import os
from random import random

from deephyper.search import Search

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Random(Search):
    """Search class to run a full random neural architecture search. The search is filling every available nodes as soon as they are detected. The master job is using only 1 MPI rank.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
    """

    def __init__(self, problem, run, evaluator, **kwargs):

        if MPI is None:
            self.free_workers = 1
        else:
            nranks = MPI.COMM_WORLD.Get_size()
            if evaluator == 'balsam':  # TODO: async is a kw
                balsam_launcher_nodes = int(
                    os.environ.get('BALSAM_LAUNCHER_NODES', 1))
                deephyper_workers_per_node = int(
                    os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))
                n_free_nodes = balsam_launcher_nodes - nranks  # Number of free nodes
                self.free_workers = n_free_nodes * \
                    deephyper_workers_per_node  # Number of free workers
            else:
                self.free_workers = 1

        super().__init__(problem, run, evaluator, **kwargs)

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument("--problem",
                            default="deephyper.benchmark.nas.linearReg.Problem",
                            help="Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem)."
                            )
        parser.add_argument("--run",
                            default="deephyper.search.nas.model.run.quick",
                            help="Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.quick)."
                            )
        parser.add_argument('--max-evals', type=int, default=1e10,
                            help='maximum number of evaluations.')
        return parser

    def main(self):

        # Setup
        space = self.problem.space
        cs_kwargs = space['create_structure'].get('kwargs')
        if cs_kwargs is None:
            structure = space['create_structure']['func']()
        else:
            structure = space['create_structure']['func'](**cs_kwargs)

        len_arch = structure.max_num_ops
        def gen_arch(): return [random() for _ in range(len_arch)]

        num_evals_done = 0
        available_workers = self.free_workers

        def gen_batch(size):
            batch = []
            for _ in range(size):
                cfg = space.copy()
                cfg['arch_seq'] = gen_arch()
                batch.append(cfg)
            return batch

        # Filling available nodes at start
        self.evaluator.add_eval_batch(gen_batch(size=available_workers))

        # Main loop
        while num_evals_done < self.args.max_evals:
            results = self.evaluator.get_finished_evals()

            num_received = num_evals_done
            for _ in results:
                num_evals_done += 1
            num_received = num_evals_done - num_received

            # Filling available nodes
            if num_received > 0:
                self.evaluator.add_eval_batch(gen_batch(size=num_received))


if __name__ == "__main__":
    args = Random.parse_args()
    search = Random(**vars(args))
    search.main()
