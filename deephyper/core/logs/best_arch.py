import os
import sys
import json
import numpy as np
from math import inf

def process_f(file):
    with open(file, 'r') as f:
        data = json.load(f)
    print(f'json keys: {list(data.keys())}')
    return data

def process_archseq(data, number):
    arch_seq = data['arch_seq']
    raw_rewards = data['raw_rewards']

    l = list(zip(raw_rewards, list(range(len(raw_rewards))), arch_seq))
    l.sort(reverse=True, key=lambda x: x[0])
    print(f'arch seq shape: {np.shape(np.array(arch_seq))}')

    rewards = [l[i][0] for i in range(number)]
    index_list = [l[i][1] for i in range(number)]
    best_archs = [l[i][2] for i in range(number)]

    data_best_arch = dict(rewards=rewards, arch_seq=best_archs, index=index_list)
    with open('best_archs.json', 'w') as f:
        json.dump(data_best_arch, f, indent=4)

def main(path, number, *args, **kwargs):
    print(f'processing: {path}')
    data = process_f(path)
    process_archseq(data, number)
