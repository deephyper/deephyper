import subprocess
import os
import sys
from collections import defaultdict
from itertools import chain
import numpy as np

def grep_times(path):
    taskpaths = os.path.join(path, 'task*/*.out')
    cmd = f'grep --color=never TIMER {taskpaths}'
    print("running", cmd)
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        sys.exit(1)

    output = proc.stdout.decode('utf-8')
    for line in output.split('\n'):
        if 'TIMER' in line:
            dat = line.split('TIMER')[1]
            category, time = dat.split(':')
            time = float(time.split()[0])
            yield (category.strip(), time)

def main(path, do_plot=False):
    data_path = os.path.expanduser(path)
    data_path = os.path.join(data_path, 'data')
    wf_paths = os.listdir(data_path)

    for p in wf_paths:
        data = defaultdict(list)
        print("# Searching tasks in", p)
        wf_path = os.path.join(data_path, p)
        for category, time in grep_times(wf_path):
            data[category].append(time)

        with open(p+'.timing.dat', 'w') as fp:
            for category in data:
                dat = np.asarray(data[category], dtype=np.float64)
                data[category] = dat

                fp.write(category + '\n')
                fp.write('\n'.join(map(str, dat)))
                fp.write('\n')

                print(category)
                print("  mean", dat.mean())
                print("  median", np.percentile(dat, 50))
                print("  10th percentile", np.percentile(dat, 10))
                print("  lower quartile",  np.percentile(dat, 25))
                print("  upper quartile",  np.percentile(dat, 75))
                print("  90th percentile",  np.percentile(dat, 90))

    if do_plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(len(data), sharex=False, sharey=False)
        for ax, cat in zip(axes.flatten(), data):
            upper = max(data[cat].max(), 0.001)
            ax.set_xlim(0, upper)
            ax.hist(data[cat], bins=50, range=(0, upper))
            ax.set_title(cat)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    do_plot = 'plot' in sys.argv
    main(sys.argv[1], do_plot=do_plot)
