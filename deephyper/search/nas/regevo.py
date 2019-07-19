import os
import collections
import random
import numpy as np
import time

from deephyper.search import Search, util
from deephyper.core.logs.logging import JsonMessage as jm

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

dhlogger = util.conf_logger(
    'deephyper.search.nas.regevo')

class RegularizedEvolution(Search):
    """Regularized evolution.

    https://arxiv.org/abs/1802.01548

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

        # Setup
        self.pb_dict = self.problem.space
        cs_kwargs = self.pb_dict['create_structure'].get('kwargs')
        if cs_kwargs is None:
            structure = self.pb_dict['create_structure']['func']()
        else:
            structure = self.pb_dict['create_structure']['func'](**cs_kwargs)

        self.space_list = [(0, vnode.num_ops-1) for vnode in structure.variable_nodes]
        self.population_size = self.args.population_size
        # self.cycles = self.args.cycles
        self.sample_size = self.args.sample_size

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument("--problem",
                            default="deephyper.benchmark.nas.linearReg.Problem",
                            help="Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem)."
                            )
        parser.add_argument("--run",
                            default="deephyper.search.nas.model.run.alpha.run",
                            help="Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.alpha.run)."
                            )
        parser.add_argument("--max-evals", type=int, default=1e10,
                            help="maximum number of evaluations.")
        parser.add_argument("--population-size", type=int, default=100,
                            help="the number of individuals to keep in the population.")
        # parser.add_argument("--cycles", type=int, default=1e10,
        #                     hel='the number of cycles the algorithm should run for.')
        parser.add_argument("--sample-size", type=int, default=10,
                            help="the number of individuals that should participate in each tournament.")
        return parser

    def main(self):

        num_evals_done = 0
        # num_cyles_done = 0
        population = collections.deque(maxlen=self.population_size)

        # Filling available nodes at start
        self.evaluator.add_eval_batch(self.gen_random_batch(size=self.free_workers))


        # Main loop
        while num_evals_done < self.args.max_evals:

            # Collecting finished evaluations
            new_results = list(self.evaluator.get_finished_evals())


            if len(new_results) > 0:
                population.extend(new_results)
                stats = {
                    'num_cache_used': self.evaluator.stats['num_cache_used'],
                }
                dhlogger.info(jm(type='env_stats', **stats))
                self.evaluator.dump_evals(saved_key='arch_seq')

                num_received = len(new_results)
                num_evals_done += num_received

                if len(population) == self.population_size:
                    children_batch = []
                    for _ in range(len(new_results)):
                        # select_sample
                        indexes = np.random.choice(self.population_size, self.sample_size, replace=False)
                        sample = [population[i] for i in indexes]
                        # select_parent
                        parent = self.select_parent(sample)
                        # copy_mutate_parent
                        children_batch.append(self.copy_mutate_arch(parent))
                    # submit_childs
                    self.evaluator.add_eval_batch(children_batch)
                else:
                    self.evaluator.add_eval_batch(self.gen_random_batch(size=len(new_results)))

    def select_parent(self, sample: list):
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg['arch_seq']

    def gen_random_batch(self, size):
        batch = []
        for _ in range(size):
            cfg = self.pb_dict.copy()
            cfg['arch_seq'] = self.random_architecture()
            batch.append(cfg)
        return batch

    def random_architecture(self):
        return [np.random.choice(b+1) for (_,b) in self.space_list]

    def copy_mutate_arch(self, parent_arch):
        i = np.random.choice(len(parent_arch))
        child_arch = parent_arch[:]
        child_arch[i] = np.random.choice(self.space_list[i][1]+1)
        cfg = self.pb_dict.copy()
        cfg['arch_seq'] = child_arch
        return cfg

    def regularized_evolution(self, cycles, population_size, sample_size):
        """Algorithm for regularized evolution (i.e. aging evolution).

        Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
        Classifier Architecture Search".

        Args:
            cycles: the number of cycles the algorithm should run for.
            population_size: the number of individuals to keep in the population.
            sample_size: the number of individuals that should participate in each
                tournament.

        Returns:
            history: a list of `Model` instances, representing all the models computed
                during the evolution experiment.
        """
        population = collections.deque()
        history = []  # Not used by the algorithm, only used to report results.

        # Initialize the population with random models.
        while len(population) < population_size:
            model = Model()
            model.arch = self.random_architecture()
            model.accuracy = train_and_eval(model.arch)
            population.append(model)
            history.append(model)

        # Carry out evolution in cycles. Each cycle produces a model and removes
        # another.
        while len(history) < cycles:
            # Sample randomly chosen models from the current population.
            sample = []
            while len(sample) < sample_size:
                # Inefficient, but written this way for clarity. In the case of neural
                # nets, the efficiency of this line is irrelevant because training neural
                # nets is the rate-determining step.
                candidate = random.choice(list(population))
                sample.append(candidate)

            # The parent is the best model in the sample.
            parent = max(sample, key=lambda i: i.accuracy)

            # Create the child model and store it.
            child_arch = self.mutate_arch(parent_arch)
            child.accuracy = train_and_eval(child.arch)
            population.append(child)
            history.append(child)

            # Remove the oldest model.
            population.popleft()

        return history

class Model:
    def __init__(self, arch, accuracy):
        self.arch = arch
        self.accuracy = accuracy


if __name__ == "__main__":
    args = RegularizedEvolution.parse_args()
    search = RegularizedEvolution(**vars(args))
    search.main()
