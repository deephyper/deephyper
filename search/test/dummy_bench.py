import time
import logging
import argparse

logger = logging.getLogger(__name__)

def run(param_dict):
    x, y, sleep = param_dict['x'], param_dict['y'], param_dict['sleep']
    logger.info(f'run: (x={x}, y={y})')
    time.sleep(sleep)
    print("OUTPUT:", x+y)
    return x+y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=1)
    parser.add_argument('--y', type=float, default=1)
    parser.add_argument('--sleep', type=float, default=0)
    args = parser.parse_args()
    x = args.x
    y = args.y
    sleep = args.sleep
    param_dict = vars(args)
    run(param_dict)
