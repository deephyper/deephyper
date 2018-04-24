import os
from pprint import pprint
import sys
import argparse

NDIM = 30

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

from deephyper.benchmarks import util

def run(param_dict):
    x = [param_dict[f'x{i}'] for i in range(1, 1+NDIM)]
    #assert len(x) == NDIM and all(type(xi) is float for xi in x)
    pprint(x)

    result = 0.0
    for i in range(NDIM-1):
        y = 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        result += y
    print("OUTPUT:", result)
    return result


def augment_parser(parser):
    for i in range(1, NDIM+1):
        parser.add_argument(f'--x{i}', type=float, default=2.0+i)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    run(param_dict)
