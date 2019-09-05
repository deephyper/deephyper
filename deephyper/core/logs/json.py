import sys

def add_subparser(subparsers):
    subparser_name = 'json'
    function_to_call = main

    parser = subparsers.add_parser(subparser_name, help='Tool to analyse a JSON file produced by the "parse" tool.')
    subparsers = parser.add_subparsers(help='Kind of analytics.')

    # best search_spaces
    subparser = subparsers.add_parser('best',
        help='Select the best n search_spaces and save them into a JSON file.')
    subparser.add_argument('-n', '--number', type=int,
        help='number of best search_space to select.')
    subparser.add_argument('-p', '--path', type=str,
        help='path to the JSON file.')

    return subparser_name, function_to_call

def main(*args, **kwargs):

    if sys.argv[2] == 'best':
        from deephyper.core.logs.best_arch import main
        main(**kwargs)