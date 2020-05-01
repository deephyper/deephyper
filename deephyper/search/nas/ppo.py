import os

from deephyper.core.parser import add_arguments_from_signature
from deephyper.search.nas.rl import ReinforcementLearningSearch

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Ppo(ReinforcementLearningSearch):
    """Search class to run a proximal policy optimization search. The search is launching as many agents as the number of MPI ranks. Each agent is batch synchronous on a number of DNN evaluations. This number of parallel evaluation is equal for all agents, and automaticaly computed based on the number of available workers when using `evaluator='balsam'`. For other evaluators it will be set to 1.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.alpha.run).
        evaluator (str): value in ['balsam', 'ray', 'subprocess', 'processPool', 'threadPool']. Default to 'ray'.
        cliprange (float, optional): Clipping parameter of PPO. Defaults to 0.2.
        ent_coef (float, optional): Entropy parameter for PPO. Adding entropy helps to avoid convergence to a local optimum. To increase the entropy parameter is to increase exploration. Defaults to 0.01.
        gamma (float, optional): Gamma parameter for advantage function in RL, discounting factor for rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for advantage function in RL, advantage estimation discounting factor (lambda in the paper). Defaults to 0.95.
        nminibatches (int, optional): Number of minibatches per environments. Here it's directly the number of batch of search_spaces. Defaults to 1.
        noptepochs (int, optional): Number of optimization steps to do per epochs. Basicaly it means the number of time you want to use learning data. Defaults to 10.
        network (str): policy network for the search, value in [
            'ppo_lstm_128',
            'ppo_lnlstm_128',
            'ppo_lstm_64',
            'ppo_lnlstm_64',
            'ppo_lstm_32',
            'ppo_lnlstm_32'
            ].
        env (str): Gym environment used among ['NasEnv1', 'NasEnv2'].
    """

    def __init__(
        self,
        problem,
        run,
        evaluator,
        network="ppo_lnlstm_128",
        cliprange=0.2,
        ent_coef=0.01,
        gamma=1.0,
        lam=0.95,
        nminibatches=1,
        noptepochs=10,
        env="NasEnv2",
        **kwargs
    ):

        super().__init__(
            problem,
            run,
            alg="ppo2",
            evaluator=evaluator,
            network=network,
            cliprange=cliprange,
            ent_coef=ent_coef,
            gamma=gamma,
            lam=lam,
            nminibatches=nminibatches,
            noptepochs=noptepochs,
            env=env,
            **kwargs
        )

    @staticmethod
    def _extend_parser(parser):
        ReinforcementLearningSearch._extend_parser(parser)
        add_arguments_from_signature(parser, Ppo)
        return parser


if __name__ == "__main__":
    args = Ppo.parse_args()
    search = Ppo(**vars(args))
    search.main()
