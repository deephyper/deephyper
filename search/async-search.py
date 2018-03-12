import logging
import os
import pickle
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.search import evaluate, util

from skopt import Optimizer
from deephyper.search.ExtremeGradientBoostingQuantileRegressor import ExtremeGradientBoostingQuantileRegressor

masterLogger = util.conf_logger()
logger = logging.getLogger(__name__)

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 30    # How many jobs to complete between optimizer checkpoints
SEED = 12345

def submit_next_points(opt_config, optimizer, evaluator):
    '''Query optimizer for the next set of points to evaluate'''
    if evaluator.counter >= opt_config.max_evals:
        logger.debug("Reached max_evals; no longer starting new runs")
        return

    if opt_config.starting_point is not None:
        XX = [self.starting_point]
        opt_config.starting_point = None
        additional_pts = optimizer.ask(n_points=evaluator.num_workers-1)
        XX.extend(additional_pts)
        logger.debug("Generating starting points")
    elif evaluator.num_free_workers() > 0:
        XX = optimizer.ask(n_points=1)
        logger.debug("Generating one point")
    else:
        XX = []
        logger.info("No free workers; waiting")

    if not opt_config.repeat_evals:
        XX = [x for x in XX 
              if evaluator.encode(x) not in evaluator.evals
              ]

    for x in XX:
        evaluator.add_eval(x)
        logger.info(f"Submitted eval of {x}")


def save_checkpoint(opt_config, optimizer, evaluator):
    data = {}
    data['opt_config'] = opt_config
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator
    
    fname = f'{opt_config.benchmark}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")

def load_checkpoint(chk_path):
    assert os.path.exists(chk_path), "No such checkpoint file"
    with open(chk_path, 'rb') as fp: data = pickle.load(fp)
    
    cfg, opt, evaluator = data['opt_config'], data['optimizer'], data['evaluator']

    cfg.num_workers = args.num_workers
    logger.info(f"Resuming from checkpoint in {chkdir}")
    logger.info(f"On eval {evaluator.counter}")
    return cfg, opt, evaluator


def main(args):
    '''Service loop: add jobs; read results; drive optimizer'''

    # Initialize optimizer
    if args.from_checkpoint:
        chk_path = args.from_checkpoint
        cfg, optimizer, evaluator = load_checkpoint(chk_path)
    else:
        cfg = util.OptConfig(args)
        optimizer = Optimizer(
            cfg.space,
            base_estimator=ExtremeGradientBoostingQuantileRegressor(),
            acq_optimizer='sampling',
            acq_func='LCB',
            acq_func_kwargs={'kappa':0},
            random_state=SEED)
        evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run with {cfg.benchmark_module_name}")

    timer = util.elapsed_timer(max_runtime_minutes=None, service_period=SERVICE_PERIOD)
    chkpoint_counter = 0

    # Gracefully handle shutdown
    def handler(signum, stack):
        logger.info('Received SIGINT/SIGTERM')
        save_checkpoint(cfg, optimizer, evaluator)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # MAIN LOOP
    logger.info("Hyperopt driver starting")

    for elapsed_seconds in timer:
        logger.info("\nElapsed time:", util.pretty_time(elapsed_seconds))
        if len(evaluator.evals) == cfg.max_evals: break

        for (x, y) in evaluator.get_finished_evals():
            optimizer.tell(x, y)
            chkpoint_counter += 1
            if y == sys.float_info.max:
                logger.warning(f"WARNING: {job.cute_id} cost was not found or NaN")
        
        submit_next_points(cfg, optimizer, evaluator)

        if chkpoint_counter >= CHECKPOINT_INTERVAL:
            save_checkpoint(cfg, optimizer, evaluator)
            chkpoint_counter = 0
        sys.stdout.flush()
    
    # EXIT
    logger.info('Hyperopt driver finishing')
    save_checkpoint(cfg, optimizer, evaluator)

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)
