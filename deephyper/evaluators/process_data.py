import sys
from terminalplot import plot
from balsam.launcher.dag import BalsamJob

def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl

data = BalsamJob.objects.filter(workflow=sys.argv[1]).values_list('data__reward', flat=True)
data = list(filter(lambda e: e != None, list(data)))

print(f'data: {data}')

plot([i for i in range(len(data))], data)

data = max_list(data)
plot([i for i in range(len(data))], data)
