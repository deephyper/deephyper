import json
import time
def run(d):
    if d.get('fail', False):
        raise RuntimeError("Simulated failure (meant to happen!)")
    sleep = d.get('sleep', 0)
    time.sleep(sleep)
    return  d['x1']**2 + d['x2']**2

def key(d):
    x1, x2, sleep, fail = d['x1'], d['x2'], d.get('sleep', 0), d.get('fail', False)
    return json.dumps(dict(x1=x1, x2=x2, sleep=sleep, fail=fail))
