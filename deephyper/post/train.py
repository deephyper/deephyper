"""
The module ``deephyper.post.train`` aims to run post-training using an already defined Problem and the results of a finished search (e.g. list of best search_spaces).

Create a post-training Balsam application:

.. code-block:: console
    :caption: bash

    balsam app --name POST --exe "$(which python) -m deephyper.post.train"

Collect a list of 50 best search_spaces:

.. code-block:: console
    :caption: bash

    deephyper-analytics json best -n 50 -p /projects/datascience/regele/experiments/cfd/cls_mlp_turb/exp_0/cls_mlp_turb_exp_0_2019-05-25_15.json

Create a Balsam job to run your post-training using the previously created list of best search_spaces:

.. code-block:: console
    :caption: bash

    balsam job --name post_cls_0 --workflow post_cls_0 --app POST --args '--evaluator balsam --problem cfdpb.cls_mlp_turbulence.problem_0.Problem --fbest /projects/datascience/regele/cfdpb/cfdpb/cls_mlp_turbulence/exp/exp_0/best_archs.json'

Submit a Theta job:

.. code-block:: console
    :caption: bash

    balsam submit-launch -n 8 -q debug-cache-quad -t 60 -A datascience --job-mode mpi --wf-filter post_cls_0
"""

import argparse
import json
import logging
from pprint import pformat
import os

from deephyper.evaluator.evaluate import Evaluator
from deephyper.post.pipeline import train
from deephyper.search import util

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = logging.getLogger(__name__)


class Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class Manager:
    """Class to manage post-training submission.

        Args:
            problem (Problem): problem related to post-training.
            p_f_best (str): path to the `.json` file containing the list of best search_spaces.
            evaluator (str): name of evaluator to use for the post-training.
        """

    def __init__(self, problem, fbest, evaluator, **kwargs):

        if MPI is None:
            self.free_workers = 1
        else:
            nranks = MPI.COMM_WORLD.Get_size()
            if evaluator == 'balsam': 
                balsam_launcher_nodes = int(
                    os.environ.get('BALSAM_LAUNCHER_NODES', 1))
                deephyper_workers_per_node = int(
                    os.environ.get('DEEPHYPER_WORKERS_PER_NODE', 1))
                n_free_nodes = balsam_launcher_nodes - nranks  # Number of free nodes
                self.free_workers = n_free_nodes * \
                    deephyper_workers_per_node  # Number of free workers
            else:
                self.free_workers = 1

        _args = vars(self.parse_args(''))
        kwargs['problem'] = problem
        kwargs['p_f_best'] = fbest
        kwargs['evaluator'] = evaluator
        _args.update(kwargs)
        self.args = Namespace(**_args)
        self.problem = util.generic_loader(problem, 'Problem')
        if kwargs.get('cache_key') is None:
            self.evaluator = Evaluator.create(
                run_function=train, method=evaluator)
        else:
            self.evaluator = Evaluator.create(
                run_function=train, method=evaluator, cache_key=kwargs['cache_key'])
        self.num_workers = self.evaluator.num_workers

        logger.info(f'Options: '+pformat(self.args.__dict__, indent=4))
        logger.info('Problem definition: ' +
                    pformat(self.problem.space, indent=4))
        logger.info(f'Created {self.args.evaluator} evaluator')
        logger.info(f'Evaluator: num_workers is {self.num_workers}')

    def main(self):
        with open(self.args.p_f_best, 'r') as f_best:
            data = json.load(f_best)

        cursor = 0
        batch = []

        for _ in range(min(self.num_workers, len(data['arch_seq']))):
            cfg = self.problem.space.copy()
            cfg['arch_seq'] = data['arch_seq'][cursor]
            cfg['id'] = cursor
            batch.append(cfg)
            cursor += 1

        self.evaluator.add_eval_batch(batch)

        num_evals_done = 0
        # Main loop
        while cursor < len(data['arch_seq']):
            results = self.evaluator.get_finished_evals()

            num_received = num_evals_done
            for _ in results:
                num_evals_done += 1
            num_received = num_evals_done - num_received

            # Filling available nodes
            if num_received > 0:
                batch = []
                for _ in range(min(num_received, len(data['arch_seq']))):
                    cfg = self.problem.space.copy()
                    cfg['arch_seq'] = data['arch_seq'][cursor]
                    cfg['id'] = cursor
                    batch.append(cfg)
                    cursor += 1
                self.evaluator.add_eval_batch(batch)

    @classmethod
    def parse_args(cls, arg_str=None):
        parser = cls._base_parser()
        try:
            parser = cls._extend_parser(parser)
        except NotImplementedError:
            pass
        if arg_str is not None:
            return parser.parse_args(arg_str)
        else:
            return parser.parse_args()

    @staticmethod
    def _extend_parser(base_parser):
        raise NotImplementedError

    @staticmethod
    def _base_parser():
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument("--problem",
                            help="Module path to the Problem instance you want to use for the post-training (e.g. deephyper.benchmark.hps.polynome2.Problem)."
                            )
        parser.add_argument("--fbest",
                            help="Path to the 'json' file containing the list of best search_spaces. "
                            )
        parser.add_argument("--backend",
                            default='tensorflow',
                            help="Keras backend module name"
                            )
        parser.add_argument('--evaluator',
                            default='subprocess',
                            choices=['balsam', 'subprocess'],
                            help="The evaluator is an object used to run the model."
                            )
        return parser


if __name__ == "__main__":
    args = Manager.parse_args()
    search = Manager(**vars(args))
    search.main()
