from string import Template
import re
import os
import sys
import time
import json
import math
import os
import subprocess
import csv

def readResults(fname, evalnum):
    pattern1 = re.compile("START TIME:", re.IGNORECASE)
    pattern2 = re.compile("OUTPUT:", re.IGNORECASE)
    pattern3 = re.compile("END TIME:", re.IGNORECASE)
    pattern4 = re.compile("INPUT:", re.IGNORECASE)
    resDict = {}
    resDict['evalnum'] = evalnum
    resDict['startTime'] = -1
    resDict['endTime'] = -1
    resDict['cost'] = sys.float_info.max
    resDict['x'] = None
    try:
        while True:
            with open(fname, 'rt') as in_file:
                for linenum, line in enumerate(in_file):
                    if pattern1.search(line) is not None:
                        print(line)
                        str1 = line.rstrip('\n')
                        res = re.findall('START TIME:(.*)', str1)
                        resDict['startTime'] = int(res[0])
                    elif pattern2.search(line) is not None:
                        print(line)
                        str1 = line.rstrip('\n')
                        res = re.findall('OUTPUT:(.*)', str1)
                        rv = float(res[0])
                        if math.isnan(rv):
                            rv = sys.float_info.max
                        resDict['cost'] = rv
                    elif pattern3.search(line) is not None:
                        print(line)
                        str1 = line.rstrip('\n')
                        res = re.findall('END TIME:(.*)', str1)
                        resDict['endTime'] = int(res[0])
                    elif pattern4.search(line) is not None:
                        print(line)
                        str1 = line.rstrip('\n')
                        res = re.findall('INPUT:(.*)', str1)
                        resDict['x'] = eval(res[0])
                if len(resDict.keys()) == 5:
                    key = os.path.basename(fname)
                    resDict['key'] = key
                    resDict['status'] = 0
            if 'endTime' in resDict.keys():
                    break
            time.sleep(5)
    except Exception:
        print('Unexpected error:', sys.exc_info()[0])
    print(resDict)
    return(resDict)

def commandLine(x, params):
    cmd = ''
    hlist = []
    for p, v in zip(params, x):
        cmd = cmd + ('--%s %s ') % (p, str(v))
    print(cmd)
    return(cmd)

def evaluate(x, evalCounter, params, prob_dir, job_dir, result_dir):
    cmd = commandLine(x, params)
    jobfile = job_dir+'/%05d.job' % evalCounter
    outputfile = result_dir+'/%05d.dat' % evalCounter
    filein = open(prob_dir+'/job.tmpl')
    src = Template(filein.read())
    inpstr = str(x)
    d = {'outputfile': outputfile, 'inpstr': inpstr, 'cmd': cmd, 'ompn':x[0]}
    result = src.substitute(d)
    with open(jobfile, "w") as jobFile:
        jobFile.write(result)
    status = subprocess.check_output('chmod +x %s' % jobfile, shell=True)
    status = subprocess.call(' sh %s ' % jobfile, shell=True)
    resDict = readResults(outputfile, evalCounter)
    print(resDict)

    return(resDict)

    