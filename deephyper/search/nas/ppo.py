import os


from deephyper.search.nas.nas_search import NeuralArchitectureSearch

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Ppo(NeuralArchitectureSearch):
    """Search class to run a proximal policy optimization search. The search is launching as many agents as the number of MPI ranks. Each agent is batch synchronous on a number of DNN evaluations. This number of parallel evaluation is equal for all agents, and automaticaly computed based on the number of available workers when using `evaluator='balsam'`. For other evaluators it will be set to 1.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
        network: (str): policy network for the search, value in [
            'ppo_lstm_128',
            'ppo_lnlstm_128',
            'ppo_lstm_64',
            'ppo_lnlstm_64',
            'ppo_lstm_32',
            'ppo_lnlstm_32'
            ].
    """

    def __init__(self, problem, run, evaluator, network, **kwargs):
        if MPI is None:
            nenvs = 1
        else:
            nranks = MPI.COMM_WORLD.Get_size()
            if evaluator == 'balsam':  # TODO: async is a kw
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

        super().__init__(problem, run, evaluator,
                         alg="ppo2",
                         network=network,
                         num_envs=nenvs,
                         **kwargs)

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        parser.add_argument("--cliprange",
                            type=float,
                            default=0.2,
                            help="Clipping parameter of PPO."
                            )
        parser.add_argument("--ent-coef",
                            type=float,
                            default=0.01,
                            help="Entropy parameter for PPO. Adding entropy helps to avoid convergence to a local optimum. To increase the entropy parameter is to increase exploration."
                            )
        parser.add_argument("--gamma",
                            type=float,
                            default=1.,
                            help="Gamma parameter for advantage function in RL, discounting factor for rewards.")
        parser.add_argument("--lam",
                            type=float,
                            default=0.95,
                            help="Lambda parameter for advantage function in RL, advantage estimation discounting factor (lambda in the paper).")
        parser.add_argument("--nminibatches",
                            type=int,
                            default=1,
                            help="Number of minibatches per environments. Here it's directly the number of batch of architectures.")
        parser.add_argument("--noptepochs",
                            type=int,
                            default=10,
                            help="Number of optimization steps to do per epochs. Basicaly it means the number of time you want to use learning data.")
        parser.add_argument('--max-evals', type=int, default=1e10,
                            help='maximum number of learning update.')
        parser.add_argument('--network', type=str, default='ppo_lstm_128',
                            choices=[
                                'ppo_lstm_128',
                                'ppo_lnlstm_128',
                                'ppo_lstm_64',
                                'ppo_lnlstm_64',
                                'ppo_lstm_32',
                                'ppo_lnlstm_32'
                            ],
                            help='Policy-Value network.')
        return parser


if __name__ == "__main__":
    args = Ppo.parse_args()
    search = Ppo(**vars(args))
    search.main()
