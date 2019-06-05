import signal
import logging
import argparse
from mpi4py import MPI
from deephyper.search import util
from deephyper.evaluator.evaluate import Evaluator
from .search import Search
from .hps.ambs import AMBS, on_exit as ambs_on_exit
from .hps.ga import GA, on_exit as ga_on_exit

logger = logging.getLogger(__name__)

class MPIWorker():

    def __init__(self, comm, problem, run, evaluator, **kwargs):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.problem = util.generic_loader(problem, 'Problem')
        self.run_func = util.generic_loader(run, 'run')
        self.evaluator = Evaluator.create(self.run_func, method=evaluator)

    def _exec_eval(self, x):
        self.evaluator.add_eval(x)
        for (x,y) in self.evaluator.get_finished_evals():
            return y

    def run(self):
        while(True):
            exec_info = comm.recv(source=0, tag=0)
            if('exit' in exec_info):
                break
            y = self._exec_eval(exec_info['args'])
            comm.send(y, dest=0, tag=0)

class MPIManager():

    def __init__(self, comm, **kwargs):
        self.comm = comm
        self.args = kwargs
        self.args['evaluator'] = '__mpiPool'
        search_cls = self._get_search_cls(**self.args)
        self.search = search_cls(**self.args)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--search',
            choices=["AMBS", "GA"],
            help='type of HPS search method to use'
            )
        args, remaining_cmd =  parser.parse_known_args()
        search_cls = MPIManager._get_search_cls(**vars(args))
        remaining_args = search_cls.parse_args(remaining_cmd)
        setattr(remaining_args, 'search', args.search)
        return remaining_args

    def run(self):
        if(isinstance(self.search, AMBS)):
            signal.signal(signal.SIGINT, ambs_on_exit)
            signal.signal(signal.SIGTERM, ambs_on_exit)
            self.search.main()
        elif(isinstance(self.search, GA)):
            self.search.run()

    @staticmethod
    def _get_search_cls(**kwargs):
        search_cls = None
        if(kwargs['search'] == 'AMBS'):
            search_cls = AMBS
        elif(kwargs['search'] == 'GA'):
            search_cls = GA
        return search_cls

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = MPIManager.parse_args()
    if(rank == 0):
        master = MPIManager(comm, **vars(args))
        master.run()
    else:
        worker = MPIWorker(comm, **vars(args))
        worker.run()
