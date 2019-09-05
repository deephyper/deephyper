import os
import sys
import json
import numpy as np
from math import inf

def process_f(file):
    with open(file, 'r') as f:
        data = json.load(f)
    print(f'Json keys are: {list(data.keys())}')
    return data

def process_archseq(data, number):
    arch_seq = data['arch_seq']
    raw_rewards = data['raw_rewards']

    l = list(zip(raw_rewards, list(range(len(raw_rewards))), arch_seq))
    l.sort(reverse=True, key=lambda x: x[0])
    shape = np.shape(np.array(arch_seq))
    print(f'Json contains: {shape[0]} search_space sequences.')
    print(f'Sequences length is {shape[1]}.')

    i = 0
    best_archs = list()
    rewards = list()
    index_list = list()
    while i < len(l) \
            and len(best_archs) < number:
        reward, index, arch = l[i]
        if all([arch != e for e in best_archs]):
                best_archs.append(arch)
                rewards.append(reward)
                index_list.append(index)
        i += 1

    data_best_arch = dict(rewards=rewards, arch_seq=best_archs, index=index_list)
    with open('best_archs.json', 'w') as f:
        json.dump(data_best_arch, f, indent=4)

def main(path, number, *args, **kwargs):
    print(f'Processing: {path}, to collect top {number} best search_spaces.')
    data = process_f(path)
    process_archseq(data, number)
