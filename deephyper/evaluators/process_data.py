import sys
import json
import datetime
from terminalplot import plot
from balsam.launcher.dag import BalsamJob

now = '_'.join(str(datetime.datetime.now(datetime.timezone.utc)).split(" "))

def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl

def rm_none(l):
    return list(filter(lambda e: e != None, list(l)))

def process_data(workflow):
    data = BalsamJob.objects.filter(workflow=workflow).values_list('data__reward', flat=True)
    print(f'data len: {len(data)}')

    raw_rewards = list(filter(lambda e: e != None, rm_none(data)))
    if len(raw_rewards) == 0:
        print(f'no rewards for : {workflow}')
        return -1

    plot([i for i in range(len(raw_rewards))], raw_rewards)

    max_rewards = max_list(raw_rewards)
    plot([i for i in range(len(max_rewards))], max_rewards)

    data = BalsamJob.objects.filter(workflow=workflow).values_list('data__arch_seq', flat=True)
    arch_seq = rm_none(data)

    data = BalsamJob.objects.filter(workflow=workflow).values_list('data__id_worker', flat=True)
    w = rm_none(data)

    filename = f'wf-{workflow}_{now}'
    print(f'filename: {filename}')
    with open('data/'+filename+'.json', "w") as f:
        data = dict(
            fig=filename,
            raw_rewards=raw_rewards,
            max_rewards=max_rewards,
            arch_seq=arch_seq,
            id_worker=w
            )
        json.dump(data, f)
    return 0

for wf in sys.argv[1:]:
    process_data(wf)
