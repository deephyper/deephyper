from collections import OrderedDict
import csv
import logging
import json
import os
import sys
import signal
import time
from importlib import import_module
from importlib.util import find_spec
from numpy import integer, floating, ndarray

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
package = os.path.basename(os.path.dirname(HERE)) # 'deephyper'
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)
os.environ['KERAS_BACKEND']='tensorflow'

from skopt import Optimizer
from deephyper.search.ExtremeGradientBoostingQuantileRegressor import ExtremeGradientBoostingQuantileRegressor
from deephyper.search import util

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
logger = util.conf_logger('deephyper.search.amls_mpi')


class Config:
    class Encoder(json.JSONEncoder):
        '''JSON dump of numpy data'''
        def default(self, obj):
            if isinstance(obj, integer): return int(obj)
            elif isinstance(obj, floating): return float(obj)
            elif isinstance(obj, ndarray): return obj.tolist()
            else: return super(Encoder, self).default(obj)

    def __init__(self, args):
        self.__dict__ = vars(args)
        
        # for example, the default value of args.benchmark is "b1.addition_rnn"
        benchmark_directory = args.benchmark.split('.')[0] # "b1"
        self.benchmark = args.benchmark
        problem_module_name = f'{package}.benchmarks.{benchmark_directory}.problem'
        problem_module = import_module(problem_module_name)

        # get the path of the b1/addition_rnn.py file here:
        self.benchmark_module_name = f'{package}.benchmarks.{args.benchmark}'
        self.benchmark_filename = find_spec(self.benchmark_module_name).origin
        
        # create a problem instance and configure the skopt.Optimizer
        instance = problem_module.Problem()
        self.params = list(instance.params)
        self.starting_point = instance.starting_point
        
        spaceDict = instance.space
        self.space = [spaceDict[key] for key in self.params]

    def encode(self, x):
        return json.dumps(x, cls=self.Encoder)

    def decode(self, x):
        return json.loads(x)

    def to_param_dict(self, x, with_paths=True):
        if isinstance(x, str): x = self.decode(x)
        param_dict = {k : v for k, v in zip(self.params, x)}
        if with_paths:
            param_dict['model_path'] = self.model_path
            param_dict['data_source'] = self.data_source
            param_dict['stage_in_destination'] = self.stage_in_destination
        return param_dict

    def load_benchmark(self):
        return import_module(self.benchmark_module_name)

    def init_optimizer(self):
        kappa = 1.7 + 0.005*comm.size
        random_state = rank * 12345
        n_init = comm.size * 2

        if self.learner in ["RF", "ET", "GBRT", "DUMMY"]:
            optimizer = Optimizer(
                self.space,
                base_estimator=self.learner,
                acq_optimizer='sampling',
                acq_func='LCB',
                acq_func_kwargs={'kappa':kappa},
                random_state=random_state,
                n_initial_points = n_init
            )
        elif self.learner == "XGB":
            optimizer = Optimizer(
                self.space,
                base_estimator=ExtremeGradientBoostingQuantileRegressor(),
                acq_optimizer='sampling',
                acq_func='LCB',
                acq_func_kwargs={'kappa':kappa},
                random_state=random_state,
                n_initial_points = n_init
            )
        else:
            raise ValueError(f"Unknown learner type {self.learner}")
        logger.info(f"Rank {rank} Optimizer: {self.learner} random_state {random_state}")
        return optimizer

    def dump_evals(self, evals, timing, quit=False):
        basename = f"results_{self.benchmark}_{self.learner}"
        with open(basename+'.json', 'w') as fp:
            json.dump(evals, fp, indent=4, sort_keys=False)

        resultsList = []

        for key in evals:
            resultDict = self.to_param_dict(key, with_paths=False)
            resultDict['objective'] = evals[key]
            resultsList.append(resultDict)

        if resultsList:
            with open(basename+'.csv', 'w') as fp:
                columns = resultsList[0].keys()
                writer = csv.DictWriter(fp, columns)
                writer.writeheader()
                writer.writerows(resultsList)
        if timing:
            with open(basename+'_evalStats', 'w') as fp:
                fp.write(f'# {comm.size-1} MPI workers\n')
                fp.write(f'# {"Elapsed time / sec":20} {"Eval count":12} {"Min(objective)":16}\n')
                for elapsed, neval, best in timing:
                    fp.write(f'{elapsed:20.3f} {neval:12} {best:16.6f}\n')
        if quit:
            sys.exit(0)

def master_main(config):
    eval_dict = OrderedDict()
    timing_data = []
    workers_known_eval_index = [0 for i in range(comm.size)]
    status = MPI.Status()

    handler = lambda a, b: config.dump_evals(eval_dict, timing_data, quit=True)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    CHKPOINT_INTVAL = 5
    chkpoint_counter = 0
    start_time = time.time()
    best_eval = sys.float_info.max

    while True:
        new_eval = comm.recv(source=MPI.ANY_SOURCE, status=status)
        elapsed = time.time() - start_time
        x, y = new_eval
        if x in eval_dict: logger.info(f"Received duplicate evaluation of point {x}")
        if y < best_eval: best_eval = y
        eval_dict[x] = y
        chkpoint_counter += 1
        source = status.source
        logger.info(f"received x={x} y={y} from rank {source}")
        timing_data.append((elapsed, len(eval_dict), best_eval))

        update_index = workers_known_eval_index[source]
        update_keys = list(eval_dict.keys())[update_index:]
        update = {x : eval_dict[x] for x in update_keys}
        workers_known_eval_index[source] = len(eval_dict)
        req = comm.isend(update, dest=source)

        if chkpoint_counter == CHKPOINT_INTVAL:
            logger.info(f"Checkpointing {len(eval_dict)} evals to disk")
            config.dump_evals(eval_dict, timing_data)
            chkpoint_counter = 0
        req.wait()

def worker_main(config):
    bench_module =  config.load_benchmark()
    optimizer = config.init_optimizer()

    handler = lambda a, b : sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    while True:
        if rank == 1 and config.starting_point:
            x = config.starting_point
            config.starting_point = None
        else:
            x = optimizer.ask()

        params = config.to_param_dict(x)
        try:
            result = bench_module.run(params)
        except Exception as e:
            logger.warning(f'run() of {bench_module} with {params} caused Exception:\n{e}\nSetting result as float_max')
            result = sys.float_info.max

        new_eval = (config.encode(x), result)
        comm.send(new_eval, dest=0)
        updates = comm.recv(source=0)
        points = [(config.decode(x), updates[x]) for x in updates]
        XX, YY = zip(*points)
        optimizer.tell(XX, YY, fit=True)
        logger.debug(f"Rank {rank} got {len(points)} new evals from master; now my model has fit to {len(optimizer.yi)} evals")
       
if __name__ == "__main__":
    if rank == 0:
        args = util.create_parser().parse_args()
        args = comm.bcast(args, root=0)
        config = Config(args)
        master_main(config)
    else:
        args = None
        args = comm.bcast(args, root=0)
        config = Config(args)
        worker_main(config)
