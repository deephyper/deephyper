import signal
import logging
from mpi4py import MPI
from deephyper.search import util
from deephyper.evaluator.evaluate import Evaluator
from .hps.ambs import AMBS, on_exit

logger = logging.getLogger(__name__)

class MPIWorker():

    def __init__(self, comm, problem, run, evaluator, **kwargs):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.problem = util.generic_loader(problem, 'Problem')
        self.run_func = util.generic_loader(run, 'run')
        logger.info('Evaluator in MPI worker will execute the function: '+run)
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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = AMBS.parse_args()
    if(rank == 0):
        args.evaluator = '__mpiPool'
        search = AMBS(**vars(args))
        signal.signal(signal.SIGINT, on_exit)
        signal.signal(signal.SIGTERM, on_exit)
        search.main()
    else:
        worker = MPIWorker(MPI.COMM_WORLD, **vars(args))
        worker.run()
