import argparse
from pprint import pformat
import logging
from deephyper.search import util
from deephyper.evaluators import Evaluator

logger = logging.getLogger(__name__)

class Search:
    def __init__(self):
        self.args = self.parse_args()
        self.problem = util.load_attr_from(self.args.problem)()
        run_func = util.load_attr_from(self.args.run)
        self.evaluator = Evaluator.create(run_func, method=self.args.evaluator)
        self.num_workers = self.evaluator.num_workers

        logger.info('Hyperparameter space definition: '+pformat(self.problem.space, indent=4))
        logger.info('Evaluator will execute the function: '+self.args.run)
        logger.info(f'Created {self.args.evaluator} evaluator')
        logger.info(f'Evaluator: num_workers is {self.num_workers}')

    def run(self):
        raise NotImplementedError

    def parse_args(self):
        base_parser = self._base_parser()
        parser = self._extend_parser(base_parser)
        return parser.parse_args()

    def _extend_parser(self, base_parser):
        raise NotImplementedError

    @staticmethod
    def _base_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--problem", 
            default="deephyper.benchmarks.rosen2.problem.Problem"
        )
        parser.add_argument("--run", 
            default="deephyper.benchmarks.rosen2.rosenbrock2.run"
        )
        parser.add_argument("--backend", 
            default='tensorflow',
            help="Keras backend module name"
        )
        parser.add_argument('--max-evals', 
            type=int, default=100,
            help='maximum number of evaluations'
        )
        parser.add_argument('--eval-timeout-minutes', 
            type=int, 
            default=-1, 
            help="Kill evals that take longer than this"
        )
        parser.add_argument('--evaluator', 
            default='local', help="'balsam' or 'local'"
        )
        return parser
