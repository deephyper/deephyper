import os

from deephyper.search import util
from deephyper.search.nas.nas_search import NeuralArchitectureSearch
from deephyper.evaluator._balsam import BalsamEvaluator  # TODO: async kw

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


dhlogger = util.conf_logger('deephyper.search.nas.nas_search')


class Ppo(NeuralArchitectureSearch):
    def __init__(self, problem, run, evaluator, network, **kwargs):
        if MPI is None:
            nenvs = 1
        else:
            nranks = MPI.COMM_WORLD.Get_size()
            if isinstance(evaluator, BalsamEvaluator):  # TODO: async kw
                balsam_launcher_nodes = int(
                    os.environ.get('BALSAM_LAUNCHER_NODES', 1))
                deephyper_workers_per_node = int(
                    os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))
                nagents = nranks  # No parameter server here
                n_free_nodes = balsam_launcher_nodes - nranks  # Number of free nodes
                free_workers = n_free_nodes * deephyper_workers_per_node  # Number of free workers
                nenvs = free_workers // nagents
            else:
                nenvs = 1

        dhlogger.info(f'nenvs: {nenvs}')

        super().__init__(problem, run, evaluator,
                         alg="ppo2",
                         network=network,
                         num_envs=nenvs,
                         **kwargs)


if __name__ == "__main__":
    args = Ppo.parse_args()
    search = Ppo(**vars(args))
    search.main()
