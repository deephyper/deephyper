import json
from pprint import pprint
from search.utils import evaluatePoint
from sys import argv

with open(argv[1]) as fp:
    task = json.loads(fp.read())

print("executing task")
pprint(task)

result = evaluatePoint(task['x'], task['eval_counter'], task['params'],
                       task['prob_dir'], task['jobs_dir'],
                       task['results_dir']
                       )

with open('result.dat', 'w') as fp:
    fp.write(json.dumps(result))
