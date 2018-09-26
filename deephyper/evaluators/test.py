#!/usr/bin/env python

import sys
from deephyper.evaluators import create_evaluator_nas, EvalFailed, TimeoutError
from importlib import import_module

class OptConfig:
    evaluator = 'balsam'
    bench_package_name = 'deephyper.benchmarks.mnistNas'
    run_module = import_module('deephyper.run.test')
    num_workers = 4
    backend = 'tensorflow'
    model_path = ''
    stage_in_destination = ''

print("creating evaluator")
evaluator = create_evaluator_nas(OptConfig())

print('adding 4 jobs')
r1 = evaluator.apply_async({'x' : 10, 'idx': 1}) # 0.1 seconds: OK
r2 = evaluator.apply_async({'x' : 100, 'idx': 2}) # 1 second: OK
r3 = evaluator.apply_async({'x' : 100, 'fail': True, 'idx': 3}) # 1 second: FAIL
r4 = evaluator.apply_async({'x' : 9000, 'idx': 4}) # 90 seconds: TIMEOUT

results = [r1,r2,r3,r4]

while results:
    for i, r in enumerate(results[:]):
        is_ready = r.ready()
        print(i, "ready?", is_ready)
        if is_ready or len(results) == 1:
            try: 
                value = r.get(5)
            except EvalFailed: 
                print(i, 'got EvalFailed')
            except TimeoutError:
                print(i, 'got TimeoutError')
            else: 
                print(i, 'succesful return:', value)
            finally:
                results.remove(r)
