import time
import subprocess
import getpass
import signal
import os

def poll_until_returns_true(function, *, args=(), period=1.0, timeout=12.0):
    start = time.time()
    result = False
    while time.time() - start < timeout:
        result = function(*args)
        if result: break
        else: time.sleep(period)
    return result

def cmdline(cmd,envs=None):
    '''Return string output from a command line'''
    if type(cmd) == list:
        cmd = ' '.join(cmd)

    cmd = f'time -p ( {cmd} )'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,env=envs,
                         executable='/bin/bash')
    stdout = p.communicate()[0].decode('utf-8')
    return stdout

def ls_procs(keywords):
    if type(keywords) == str: 
        keywords = keywords.split()

    username = getpass.getuser()
    
    searchcmd = 'ps aux | grep '
    searchcmd += ' | grep '.join(f'"{k}"' for k in keywords) 
    stdout = cmdline(searchcmd)

    processes = [line for line in stdout.split('\n') if 'python' in line and line.split()[0]==username]
    return processes

def sig_processes(process_lines, signal):
    for line in process_lines:
        proc = int(line.split()[1])
        try: 
            os.kill(proc, signal)
        except ProcessLookupError:
            pass

def stop_processes(name):
    processes = ls_procs(name)
    sig_processes(processes, signal.SIGTERM)
    
    def check_processes_done():
        procs = ls_procs(name)
        return len(procs) == 0

    poll_until_returns_true(check_processes_done, period=2, timeout=12)
    processes = ls_procs(name)
    if processes:
        sig_processes(processes, signal.SIGKILL)
        time.sleep(3)

def stop_launcher_processes():
    stop_processes('launcher.py')
    stop_processes('mpi_ensemble.py')
