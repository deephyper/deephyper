import logging
import os
import pickle
import signal
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import evaluate
from deephyper.search import util


masterLogger = util.conf_logger()
logger = logging.getLogger('deephyper.search.async-search')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 10    # How many jobs to complete between optimizer checkpoints
SEED = 12345

def submit_next_points(opt_config, optimizer, evaluator):
    '''Query optimizer for the next set of points to evaluate'''
    if evaluator.counter >= opt_config.max_evals:
        logger.debug("Reached max_evals; no longer starting new runs")
        return False

    if opt_config.starting_point is not None:
        XX = [opt_config.starting_point]
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
        evaluator.add_eval(x, re_evaluate=opt_config.repeat_evals)
        logger.info(f"Submitted eval of {x}")
    if XX:
        return True
    else:
        return False


def save_checkpoint(opt_config, optimizer, evaluator):
    if evaluator.counter == 0: return
    data = {}
    data['opt_config'] = opt_config
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator

    if evaluator.evals:
        best = min(evaluator.evals.items(), key=lambda x: x[1])
        data['best'] = best
        logger.info(f'best point: {best}')
    
    fname = f'{opt_config.benchmark}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")

def load_checkpoint(chk_path):
    chk_path = os.path.abspath(os.path.expanduser(chk_path))
    assert os.path.exists(chk_path), "No such checkpoint file"
    with open(chk_path, 'rb') as fp: data = pickle.load(fp)
    
    cfg, opt, evaluator = data['opt_config'], data['optimizer'], data['evaluator']

    cfg.num_workers = args.num_workers
    logger.info(f"Resuming from checkpoint in {chk_path}")
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
        optimizer = util.sk_optimizer_from_config(cfg, SEED)
        evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run with {cfg.benchmark_module_name}")

    timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
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

    for elapsed_str in timer:
        logger.info(f"Elapsed time: {elapsed_str}")
        if len(evaluator.evals) == cfg.max_evals: break

        for (x, y) in evaluator.get_finished_evals():
            optimizer.tell(x, y)
            chkpoint_counter += 1
            if y == sys.float_info.max:
                logger.warning(f"WARNING: {job.cute_id} cost was not found or NaN")
        
        submitted_any = submit_next_points(cfg, optimizer, evaluator)
        timer.delay = not submitted_any # no idling while there's evals to submit

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
