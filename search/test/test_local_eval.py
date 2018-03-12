import os
import sys
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(os.path.dirname(HERE))) # directory containing deephyper
print(top)
sys.path.append(top)

from deephyper.search import evaluate_local, util
import logging

master_logger = util.conf_logger()
logger = logging.getLogger(__name__)

bench_module = 'deephyper.search.test.dummy_bench'
params_list = ['x', 'y', 'sleep']

evaluator = evaluate_local.LocalEvaluator(params_list, bench_module,
                                          num_workers=8)

evaluator.add_eval([1, 1, 0])
evaluator.add_eval([2, 2, 0])
evaluator.add_eval([3, 3, 4.5])
evaluator.add_eval([4, 4, 0.0])

time.sleep(0.4)
for x, y in evaluator.await_evals([ [1, 1, 0], [3, 3, 4.5] ]):
    print(x, y)

print("Checkpointing!")
with open('evaluator.pkl', 'wb') as fp: pickle.dump(evaluator, fp)
del evaluator
time.sleep(1)

print("Loading from checkpoint:")
with open('evaluator.pkl', 'rb') as fp: evaluator = pickle.load(fp)
print("loaded evaluator evals:", evaluator.evals)
print("loaded evaluator pending:", evaluator.pending_evals)

evaluator.add_eval([5, 5, 0.1])
evaluator.add_eval([6, 6, 0.0])

print(evaluator.counter)
print(evaluator.num_free_workers(), "free workers")

awaiting = [ [2,2,0], [4,4,0.0], [5, 5, 0.1], [6, 6, 0.0] ]

for x, y in evaluator.await_evals(awaiting):
    print(x, y)

print(evaluator.counter)
