import argparse
from pprint import pformat
import logging
from deephyper.search import util
from deephyper.evaluators import Evaluator

logger = logging.getLogger(__name__)

class Namespace:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__dict__[k] = v

class Search:
    def __init__(self, problem=None, evaluator='local', **kwargs):
        self.args = Namespace(**kwargs)
        self.problem = util.load_attr_from(self.args.problem)()
        self.num_workers = self.evaluator.num_workers
        self.evaluator = Evaluator.create(self.run_func, method=self.args.evaluator)

        logger.info('Hyperparameter space definition: '+pformat(self.problem.space, indent=4))
        logger.info(f'Created {self.args.evaluator} evaluator')
        logger.info(f'Evaluator: num_workers is {self.num_workers}')

    def run_func(self):
        raise NotImplementedError

    def main(self):
        raise NotImplementedError

    @classmethod
    def parse_args(cls):
        base_parser = cls._base_parser()
        parser = cls._extend_parser(base_parser)
        return parser.parse_args()

    @staticmethod
    def _extend_parser(base_parser):
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
