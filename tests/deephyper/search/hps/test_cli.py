"""
python test_cli.py --problem foo --run foo --evaluator balsam --evaluator-config "(a,b)"
"""
from pprint import pprint

from deephyper.search.hps import AMBS


parser = AMBS.get_parser()
args = parser.parse_args()

print(args)
# pprint(args)