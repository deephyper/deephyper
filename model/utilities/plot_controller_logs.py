import re
import matplotlib.pyplot as plt

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
            if( 'x:' in line):
                l = [float(s) for s in line.split() if isfloat(s)]
                values.append(l[-1])
    return values

if __name__ == '__main__':
    fname = '/Users/Deathn0t/Desktop/hpc-edge-service/testdb/data/run_nas1/run_nas1_64fc7bf3/'
    fname += 'deephyper.log'
    lY = parsing(fname)
    print(lY)
    fig, ax = plt.subplots()
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.plot([i for i in range(len(lY))], lY)
    ax.set_title('Architectures accuracy')
    plt.ylim(0, 100)
    plt.show()
