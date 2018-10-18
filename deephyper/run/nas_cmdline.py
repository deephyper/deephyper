import argparse
import json
from pprint import pprint

class LoadsJsonAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(LoadsJsonAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f'parser = {parser}')
        print(f'namespace = {namespace}')
        print(f'values = {values}')
        print(f'values type = {type(values)}')
        print(f'json loads = {json.loads(values)}')
        setattr(namespace, self.dest, json.loads(values))

def create_parser(): # TODO
    'command line parser for NAS'
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', action=LoadsJsonAction)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    pprint(param_dict)
