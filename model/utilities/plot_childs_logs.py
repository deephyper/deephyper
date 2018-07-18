import re
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import spline
import seaborn as sns

def isfloat(n):
    try:
        n = float(n)
        return True
    except:
        return False

def parsing(fname):
    values = []
    with open(fname) as f:
        for line in f:
            if ('Validation accuracy' in line):
                values.append(float(line.split()[-1][:-2]))
            elif( 'global_step' in line):
                l = [int(s) for s in line.split() if s.isdigit()]
                step = l[-1]
    return step, values

def loop_directory(path_directory, ax):
    #ax.grid(color='k', linestyle='-', linewidth=1)
    all_files = os.listdir(path_directory)
    cpl = sns.color_palette("Blues", len(all_files))
    ax.clear()
    for file_name in all_files:
        try:
            if file_name.startswith('task'):
                step, lY = parsing(f'{path_directory}/{file_name}/deephyper.log')
                lX = np.array([i for i in range(len(lY))])
                xnew = np.linspace(0, len(lY)-1, 100)
                power_smooth = spline(lX, np.array(lY), xnew)
                ax.plot(xnew, power_smooth, color=cpl[step])
        except:
            raise

if __name__ == '__main__':
    fname = '/Users/Deathn0t/Desktop/hpc-edge-service/testdb/data/run_nas1'
    plt.style.use('ggplot')
    plt.ylim(0, 100)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Childs accuracy')
    #animate = lambda i: loop_directory(fname, ax1)
    #ani = animation.FuncAnimation(fig, animate, interval=10000)
    loop_directory(fname, ax1)
    plt.show()
