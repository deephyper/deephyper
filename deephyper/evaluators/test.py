import json
from random import randint
from deephyper.evaluators import Evaluator
import logging

logging.basicConfig(level=logging.DEBUG)

def my_run(d):
    return  d['x1']**2 + d['x2']**2

def my_key(d):
    x1, x2 = d['x1'], d['x2']
    return json.dumps(dict(x1=x1, x2=x2))

if __name__ == "__main__":
    ev = Evaluator.create(my_run, cache_key=my_key, method='local')

    N = 2
    for i in range(N):
        ID = str(i)
        x1,x2 = randint(1, 10), randint(-10, 0)
        d = dict(ID=ID, x1=x1, x2=x2)
        d2 = d.copy()
        d2['ID'] = d['ID'] + '_copy'
        ev.add_eval(d)
        ev.add_eval(d2)

    for x,y in ev.get_finished_evals():
        print(x, y)
