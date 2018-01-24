import json
from re import findall
import math
import os
import subprocess
import sys

def readResults(fname, x):
    resDict = {}
    resDict['cost'] = sys.float_info.max
    resDict['x'] = x
    with open(fname, 'rt') as fp:
        for linenum, line in enumerate(fp):
            if "OUTPUT:" in line.upper():
                print(line)
                str1 = line.rstrip('\n')
                res = findall('OUTPUT:(.*)', str1)
                rv = float(res[0])
                if math.isnan(rv):
                    rv = sys.float_info.max
                resDict['cost'] = rv
                break
    return resDict


def main():
    '''Run the task specified by input file; dump results in result.dat'''
    with open(sys.argv[1]) as fp:
        task = json.loads(fp.read())

    x = task['x']
    params = task['params']
    benchmark = task['benchmark']
    backend = task['backend']

    args = ' '.join(f"--{p}={v}" for p,v in zip(params, x) if 'hidden' not in p)
    cmd  = f"KERAS_BACKEND={backend} {sys.executable} {benchmark} {args}"
    print(cmd)

    with open(f"benchmark.out", 'w') as fp:
        proc = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT, shell=True)
        retcode = proc.wait()

    print("task done with return code", retcode)
    results = readResults("benchmark.out", x)

    with open('result.dat', 'w') as fp:
        fp.write(json.dumps(results))

if __name__ == "__main__":
    main()
