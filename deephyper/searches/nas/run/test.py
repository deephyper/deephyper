import random
import time
from pprint import pformat

from deephyper.searches.nas.run.nas_cmdline import create_parser


def run(param_dict):
    print(f"run param_dict: {pformat(param_dict, indent=4)}")
    x = param_dict['x']
    time.sleep(x/100)
    output = param_dict['idx']

    if param_dict.get('fail') == True:
        raise RuntimeError
    else:
        print("OUTPUT:", output)
        return output

if __name__ == '__main__':
    parser = create_parser()
    cmdline_args = parser.parse_args()
    param_dict = cmdline_args.config
    run(param_dict)
